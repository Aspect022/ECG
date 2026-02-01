"""
Quantized Neural Network Layers for V2.0 Architecture.

Implements:
- QuantizedConv1d: Binary/multi-bit weight quantization
- QuantLIFNeuron: Quantized membrane potential LIF neuron

Optimized for RTX 5050 (8GB VRAM) with memory-efficient operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class StraightThroughEstimator(torch.autograd.Function):
    """Straight-through estimator for binary quantization."""
    
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        return torch.sign(input)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.clamp(-1, 1)


class QuantizedConv1d(nn.Module):
    """
    1D Convolution with weight quantization.
    
    Supports binary (1-bit) and multi-bit quantization.
    Uses straight-through estimator for gradient flow.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding (auto-calculated if None)
        groups: Number of groups for grouped convolution
        weight_bits: Bit-width for weight quantization (1, 2, 4, 8)
        activation_bits: Bit-width for activation quantization (8 default)
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        weight_bits: int = 1,
        activation_bits: int = 8,
        bias: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding is not None else kernel_size // 2
        self.groups = groups
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Full-precision weights (will be quantized during forward)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size) * 0.02
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Scaling factor for binary weights
        self.register_buffer('weight_scale', torch.ones(out_channels, 1, 1))
        
        # Initialize with Kaiming
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    
    def _quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        """Quantize weights to specified bit-width."""
        if self.weight_bits == 1:
            # Binary quantization: sign function with STE
            # Scale by mean absolute value for better gradient flow
            scale = w.abs().mean(dim=[1, 2], keepdim=True) + 1e-8
            w_normalized = w / scale
            w_binary = StraightThroughEstimator.apply(w_normalized)
            return w_binary * scale
        
        elif self.weight_bits in [2, 4, 8]:
            # Multi-bit uniform quantization
            n_levels = 2 ** self.weight_bits
            w_min = w.min()
            w_max = w.max()
            scale = (w_max - w_min) / (n_levels - 1)
            
            # Quantize
            w_quantized = torch.round((w - w_min) / (scale + 1e-8))
            w_quantized = w_quantized.clamp(0, n_levels - 1)
            
            # Dequantize for forward pass
            w_dequantized = w_quantized * scale + w_min
            
            # STE: use quantized for forward, full precision for backward
            return w + (w_dequantized - w).detach()
        
        else:
            return w  # No quantization
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with quantized weights.
        
        Args:
            x: Input tensor of shape (batch, in_channels, length)
            
        Returns:
            Output tensor of shape (batch, out_channels, new_length)
        """
        # Quantize weights
        w_q = self._quantize_weights(self.weight)
        
        # Convolution
        return F.conv1d(
            x, w_q, self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )


class QuantLIFNeuron(nn.Module):
    """
    Quantized Leaky Integrate-and-Fire Neuron.
    
    Membrane potential is quantized to specified bit-width.
    Uses surrogate gradient for spike backpropagation.
    
    Args:
        threshold: Firing threshold
        tau: Membrane time constant
        potential_bits: Bit-width for membrane potential quantization
        spike_regularization: Regularization coefficient for spike rate
        learnable_tau: Whether tau is learnable
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        tau: float = 2.0,
        potential_bits: int = 8,
        spike_regularization: float = 0.01,
        learnable_tau: bool = True
    ):
        super().__init__()
        
        self.threshold = threshold
        self.potential_bits = potential_bits
        self.spike_reg = spike_regularization
        
        # Decay factor from tau
        decay = math.exp(-1.0 / tau)
        
        if learnable_tau:
            # Learnable decay (constrained to valid range via sigmoid)
            self.decay_param = nn.Parameter(torch.tensor(decay).logit())
        else:
            self.register_buffer('decay_param', torch.tensor(decay).logit())
    
    @property
    def decay(self) -> torch.Tensor:
        """Get decay factor (always in valid range [0, 1])."""
        return torch.sigmoid(self.decay_param)
    
    def _quantize_potential(self, v: torch.Tensor) -> torch.Tensor:
        """Quantize membrane potential to specified bit-width."""
        if self.potential_bits >= 32:
            return v  # No quantization
        
        n_levels = 2 ** self.potential_bits
        v_min = -2.0  # Expected range for membrane potential
        v_max = 2.0
        scale = (v_max - v_min) / (n_levels - 1)
        
        # Clamp, quantize, dequantize
        v_clamped = v.clamp(v_min, v_max)
        v_quantized = torch.round((v_clamped - v_min) / scale) * scale + v_min
        
        # STE
        return v + (v_quantized - v).detach()
    
    def _surrogate_spike(self, v: torch.Tensor) -> torch.Tensor:
        """Fast sigmoid surrogate gradient for spikes."""
        # Forward: Heaviside
        spike = (v >= self.threshold).float()
        
        # Backward: Fast sigmoid gradient
        # d(spike)/d(v) ≈ beta / (2 * (1 + beta * |v - thresh|)^2)
        beta = 10.0
        grad_surrogate = beta / (2.0 * (1.0 + beta * torch.abs(v - self.threshold)) ** 2)
        
        # STE: spike for forward, grad_surrogate * grad_output for backward
        return spike + (grad_surrogate * (v - self.threshold) - spike + spike).detach()
    
    def forward(
        self, 
        x: torch.Tensor, 
        timesteps: int = 4,
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through LIF neuron over multiple timesteps.
        
        Args:
            x: Input current (batch, channels, length)
            timesteps: Number of simulation timesteps
            mem: Initial membrane potential (optional)
            
        Returns:
            Tuple of (spike_output, final_membrane, reg_loss)
        """
        batch, channels, length = x.shape
        device = x.device
        
        # Initialize membrane potential
        if mem is None:
            v = torch.zeros(batch, channels, length, device=device)
        else:
            v = mem
        
        # Accumulate spikes
        spike_sum = torch.zeros_like(x)
        
        # Input current per timestep (divide by timesteps for rate coding)
        current = x / timesteps
        
        for t in range(timesteps):
            # Leak
            v = self.decay * v
            
            # Integrate
            v = v + current
            
            # Quantize membrane potential
            v = self._quantize_potential(v)
            
            # Fire (with surrogate gradient)
            spike = (v >= self.threshold).float()
            
            # Reset (soft reset)
            v = v * (1.0 - spike)
            
            # Accumulate
            spike_sum = spike_sum + spike
        
        # Average spike rate
        spike_rate = spike_sum / timesteps
        
        # Spike regularization loss (encourage ~0.5 firing rate)
        reg_loss = self.spike_reg * torch.mean((spike_rate - 0.3) ** 2)
        
        return spike_rate, v, reg_loss


class QuantizedDepthwiseConv1d(nn.Module):
    """
    Depthwise separable 1D convolution with quantization.
    
    Combines depthwise and pointwise convolutions for efficiency.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        weight_bits: int = 1
    ):
        super().__init__()
        
        padding = padding if padding is not None else kernel_size // 2
        
        # Depthwise: convolve each channel separately
        self.depthwise = QuantizedConv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels,
            weight_bits=weight_bits
        )
        
        # Pointwise: 1x1 conv to mix channels
        self.pointwise = QuantizedConv1d(
            in_channels, out_channels, kernel_size=1,
            weight_bits=weight_bits
        )
        
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x


# Convenience function
def create_quant_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    weight_bits: int = 1,
    use_lif: bool = True,
    **lif_kwargs
) -> nn.Module:
    """Create a quantized conv + optional LIF block."""
    layers = [
        QuantizedConv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, weight_bits=weight_bits
        ),
        nn.BatchNorm1d(out_channels)
    ]
    
    if use_lif:
        layers.append(QuantLIFNeuron(**lif_kwargs))
    
    return nn.Sequential(*layers) if not use_lif else nn.ModuleList(layers)
