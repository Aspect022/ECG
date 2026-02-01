"""
Classical Path: Hierarchical Temporal Feature Pyramid for V2.0 Architecture.

Implements the 3-stage classical processing path:
- Stage 1: Local patterns (high-resolution, P-wave/T-wave)
- Stage 2: Beat patterns (mid-resolution, QRS complex)
- Stage 3: Rhythm patterns (low-resolution, HRV)
- Temporal Fusion Block
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Import from sibling modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantization import QuantizedConv1d, QuantLIFNeuron
from v2.attention import MultiScaleSpikingAttention, GlobalSpikingAttention


class LocalPatternExtractor(nn.Module):
    """
    Stage 1: Local High-Resolution Pattern Extractor.
    
    Captures fine details like P-wave and T-wave morphology.
    Uses depthwise separable convolution with binary weights.
    
    Input: (batch, 12, 5000)
    Output: (batch, 256, 5000)
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 256,
        kernel_size: int = 3,
        weight_bits: int = 1,
        timesteps: int = 4
    ):
        super().__init__()
        self.timesteps = timesteps
        
        # Depthwise convolution (per-channel)
        self.depthwise = QuantizedConv1d(
            in_channels, in_channels, kernel_size,
            stride=1, padding=kernel_size // 2,
            groups=in_channels, weight_bits=weight_bits
        )
        
        # Pointwise convolution (channel mixing)
        self.pointwise = QuantizedConv1d(
            in_channels, out_channels, kernel_size=1,
            weight_bits=weight_bits
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.lif = QuantLIFNeuron(
            threshold=1.0, tau=2.0,
            potential_bits=8, spike_regularization=0.01
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000)
        Returns:
            Tuple of (output, reg_loss)
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x, _, reg_loss = self.lif(x, timesteps=self.timesteps)
        return x, reg_loss


class BeatPatternExtractor(nn.Module):
    """
    Stage 2: Beat-Level Mid-Resolution Pattern Extractor.
    
    Captures QRS complex and RR intervals using larger kernels
    and multi-scale spiking attention.
    
    Input: (batch, 256, 5000)
    Output: (batch, 128, 2500)
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        kernel_size: int = 9,
        weight_bits: int = 1,
        num_heads: int = 3
    ):
        super().__init__()
        
        # Strided convolution for downsampling
        self.conv = QuantizedConv1d(
            in_channels, out_channels, kernel_size,
            stride=2, padding=kernel_size // 2,
            weight_bits=weight_bits
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Multi-scale spiking attention
        self.mssa = MultiScaleSpikingAttention(
            dim=out_channels,
            num_heads=num_heads,
            local_window=64,
            regional_window=256,
            global_pool=16
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 256, 5000)
        Returns:
            (batch, 128, 2500)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.mssa(x)
        return x


class RhythmPatternExtractor(nn.Module):
    """
    Stage 3: Rhythm Low-Resolution Pattern Extractor.
    
    Captures heart rate variability and arrhythmia patterns
    using large kernels and global attention.
    
    Input: (batch, 128, 2500)
    Output: (batch, 64, 625)
    """
    
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 64,
        kernel_size: int = 27,
        weight_bits: int = 1
    ):
        super().__init__()
        
        # Large kernel for rhythm patterns
        self.conv = QuantizedConv1d(
            in_channels, out_channels, kernel_size,
            stride=4, padding=kernel_size // 2,
            weight_bits=weight_bits
        )
        self.bn = nn.BatchNorm1d(out_channels)
        
        # Global attention
        self.global_attn = GlobalSpikingAttention(out_channels, num_heads=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 128, 2500)
        Returns:
            (batch, 64, 625)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.global_attn(x)
        return x


class TemporalFusionBlock(nn.Module):
    """
    Fuse multi-scale features from all stages.
    
    Pools features to common length, concatenates, and projects
    to final embedding dimension.
    
    Inputs: stage1 (256, 5000), stage2 (128, 2500), stage3 (64, 625)
    Output: (batch, 128)
    """
    
    def __init__(
        self,
        stage1_channels: int = 256,
        stage2_channels: int = 128,
        stage3_channels: int = 64,
        fusion_dim: int = 128,
        pool_length: int = 256,
        weight_bits: int = 2
    ):
        super().__init__()
        
        total_channels = stage1_channels + stage2_channels + stage3_channels
        
        # Adaptive pooling to common length
        self.pool = nn.AdaptiveAvgPool1d(pool_length)
        
        # Learnable scale weights
        self.alpha = nn.Parameter(torch.ones(3) / 3.0)
        
        # Projection to fusion dimension
        self.proj = QuantizedConv1d(
            total_channels, fusion_dim, kernel_size=1,
            weight_bits=weight_bits
        )
        self.bn = nn.BatchNorm1d(fusion_dim)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(
        self,
        stage1: torch.Tensor,
        stage2: torch.Tensor,
        stage3: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            stage1: (batch, 256, 5000)
            stage2: (batch, 128, 2500)
            stage3: (batch, 64, 625)
        Returns:
            (batch, 128)
        """
        # Normalize fusion weights
        weights = F.softmax(self.alpha, dim=0)
        
        # Pool to common length
        s1 = self.pool(stage1) * weights[0]  # (batch, 256, pool_length)
        s2 = self.pool(stage2) * weights[1]  # (batch, 128, pool_length)
        s3 = self.pool(stage3) * weights[2]  # (batch, 64, pool_length)
        
        # Concatenate along channel dimension
        fused = torch.cat([s1, s2, s3], dim=1)  # (batch, 448, pool_length)
        
        # Project to fusion dimension
        fused = self.proj(fused)
        fused = self.bn(fused)
        fused = F.elu(fused)
        
        # Global average pooling
        output = self.global_pool(fused).squeeze(-1)  # (batch, 128)
        
        return output


class ClassicalPath(nn.Module):
    """
    Complete Classical Path: Hierarchical Temporal Feature Pyramid.
    
    Processes 12-lead ECG through 3 stages and fuses features.
    
    Input: (batch, 12, 5000)
    Output: (batch, 128), reg_loss
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        stage1_channels: int = 256,
        stage2_channels: int = 128,
        stage3_channels: int = 64,
        fusion_dim: int = 128,
        weight_bits: int = 1,
        timesteps: int = 4
    ):
        super().__init__()
        
        # 3-stage pyramid
        self.stage1 = LocalPatternExtractor(
            in_channels, stage1_channels,
            weight_bits=weight_bits, timesteps=timesteps
        )
        self.stage2 = BeatPatternExtractor(
            stage1_channels, stage2_channels,
            weight_bits=weight_bits
        )
        self.stage3 = RhythmPatternExtractor(
            stage2_channels, stage3_channels,
            weight_bits=weight_bits
        )
        
        # Temporal fusion
        self.fusion = TemporalFusionBlock(
            stage1_channels, stage2_channels, stage3_channels,
            fusion_dim=fusion_dim
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000) - 12-lead ECG
        Returns:
            Tuple of (features, reg_loss)
            - features: (batch, 128)
            - reg_loss: scalar spike regularization loss
        """
        # Stage 1: Local patterns
        s1, reg_loss = self.stage1(x)  # (batch, 256, 5000)
        
        # Stage 2: Beat patterns
        s2 = self.stage2(s1)  # (batch, 128, 2500)
        
        # Stage 3: Rhythm patterns
        s3 = self.stage3(s2)  # (batch, 64, 625)
        
        # Fuse multi-scale features
        features = self.fusion(s1, s2, s3)  # (batch, 128)
        
        return features, reg_loss
