"""
Vision Transformer (ViT) adapted for 1D ECG signals.

Supports four operating modes via constructor flags:
  - Standard ViT     (use_snn=False, use_quantum=False)
  - ViT + SNN        (use_snn=True,  use_quantum=False)
  - ViT + Quantum    (use_snn=False, use_quantum=True)
  - ViT + SNN + Qtm  (use_snn=True,  use_quantum=True)

SNN mode replaces standard MHSA with MultiScaleSpikingAttention.
Quantum mode adds a parallel QuantumPath and GatedFusion.

Input:  (batch, 12, 5000)
Output: Dict with 'logits', 'reg_loss', optionally 'gate'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v2.attention import MultiScaleSpikingAttention
from v2.quantum_path import QuantumPath
from v2.fusion import GatedFusionModule, ClassificationHead


# ──────────────────────── Config ────────────────────────

@dataclass
class ViT1DConfig:
    """Configuration for ViT1D model."""
    # Input
    in_channels: int = 12
    input_length: int = 5000
    num_classes: int = 5

    # Patch embedding
    patch_size: int = 50        # each patch = 50 samples → 100 tokens
    embed_dim: int = 128

    # Transformer
    depth: int = 4              # number of transformer blocks
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

    # Modes
    use_snn: bool = False
    use_quantum: bool = False

    # Quantum (used only when use_quantum=True)
    n_qubits: int = 8
    n_q_layers: int = 3
    quantum_feature_dim: int = 64
    weight_bits: int = 1

    # Classifier
    hidden_dim: int = 64
    cls_dropout: float = 0.3


# ──────────────────────── Patch Embedding ────────────────────────

class PatchEmbedding1D(nn.Module):
    """
    Split the ECG signal into non-overlapping patches and embed them.

    (batch, 12, 5000) → (batch, num_patches, embed_dim)
    """

    def __init__(self, in_channels: int = 12, patch_size: int = 50,
                 embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, length)
        Returns:
            (batch, num_patches, embed_dim)
        """
        x = self.proj(x)           # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)      # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x


# ──────────────────────── Standard Transformer Block ────────────────────────

class TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block with MHSA + FFN."""

    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MHSA + residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


# ──────────────────────── Spiking Transformer Block ────────────────────────

class SpikingTransformerBlock(nn.Module):
    """
    Transformer block where MHSA is replaced by
    MultiScaleSpikingAttention from the existing codebase.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        # MSSA expects (B, C, L) and returns (B, C, L)
        self.mssa = MultiScaleSpikingAttention(
            dim=dim, num_heads=3,
            local_window=64, regional_window=256, global_pool=16,
        )
        # Additional FFN (MSSA already has its own internal FFN,
        # but we add a second one for deeper learning capacity)
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, length, dim)   ← sequence-first for ViT
        Returns:
            (batch, length, dim)
        """
        # MSSA expects (B, C, L), so transpose in and out
        x_conv = x.transpose(1, 2)         # (B, dim, L)
        x_conv = self.mssa(x_conv)          # (B, dim, L)
        x = x + x_conv.transpose(1, 2)     # residual in (B, L, dim)

        # FFN + residual
        x = x + self.ffn(self.norm(x))
        return x


# ──────────────────────── ViT 1D Model ────────────────────────

class ViT1D(nn.Module):
    """
    Vision Transformer for 1D ECG signals.

    Modes:
      Standard  → TransformerBlock
      SNN       → SpikingTransformerBlock  (existing MSSA)
      Quantum   → parallel QuantumPath + GatedFusion
      Hybrid    → SNN blocks + QuantumPath
    """

    def __init__(self, config: Optional[ViT1DConfig] = None):
        super().__init__()
        if config is None:
            config = ViT1DConfig()
        self.config = config

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbedding1D(
            in_channels=config.in_channels,
            patch_size=config.patch_size,
            embed_dim=config.embed_dim,
        )
        num_patches = config.input_length // config.patch_size

        # ── Positional Embedding + CLS token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embed_dim),
        )
        self.pos_drop = nn.Dropout(config.dropout)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Blocks ──
        if config.use_snn:
            self.blocks = nn.ModuleList([
                SpikingTransformerBlock(
                    dim=config.embed_dim,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ])

        self.norm = nn.LayerNorm(config.embed_dim)

        # ── Quantum Path (optional) ──
        self.use_quantum = config.use_quantum
        if self.use_quantum:
            self.quantum_path = QuantumPath(
                in_channels=config.in_channels,
                feature_dim=config.quantum_feature_dim,
                n_qubits=config.n_qubits,
                n_layers=config.n_q_layers,
                weight_bits=config.weight_bits,
            )
            self.fusion = GatedFusionModule(
                classical_dim=config.embed_dim,
                quantum_dim=config.n_qubits,
                gate_hidden=config.hidden_dim,
            )

        # ── Classification Head ──
        self.classifier = ClassificationHead(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.cls_dropout,
        )

    # ──────────────── forward ────────────────

    def forward(
        self, x: torch.Tensor, return_gate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000)
        Returns:
            Dict with 'logits', 'reg_loss', optionally 'gate'
        """
        B = x.shape[0]
        reg_loss = torch.tensor(0.0, device=x.device)

        # ── ViT backbone ──
        tokens = self.patch_embed(x)                    # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)        # (B, N+1, D)
        tokens = self.pos_drop(tokens + self.pos_embed)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        vit_features = tokens[:, 0]                     # CLS token → (B, D)

        # ── Quantum branch ──
        gate_value = torch.zeros(B, 1, device=x.device)

        if self.use_quantum:
            quantum_features, q_reg_loss = self.quantum_path(x)
            reg_loss = reg_loss + q_reg_loss
            vit_features, gate_value = self.fusion(
                vit_features, quantum_features,
            )

        # ── Classifier ──
        logits = self.classifier(vit_features)

        output: Dict[str, torch.Tensor] = {
            'logits': logits,
            'reg_loss': reg_loss,
        }
        if return_gate:
            output['gate'] = gate_value

        return output

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────── Self-test ────────────────

if __name__ == "__main__":
    x = torch.randn(4, 12, 5000)

    for tag, snn, qtm in [
        ("Standard ViT", False, False),
        ("ViT + SNN",    True,  False),
        ("ViT + Quantum", False, True),
        ("ViT + Hybrid",  True,  True),
    ]:
        cfg = ViT1DConfig(use_snn=snn, use_quantum=qtm)
        model = ViT1D(cfg)
        out = model(x, return_gate=True)
        print(f"{tag:18s} | params={model.num_parameters:>9,} | "
              f"logits={out['logits'].shape} | gate={out['gate'].mean():.4f}")

    print("[OK] All ViT1D variants passed!")
