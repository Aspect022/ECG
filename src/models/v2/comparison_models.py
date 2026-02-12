"""
Comparison Model Factory.

Creates any of the five model variants from a single config dict:
  - resnet        → ResNet1D baseline
  - vit           → Standard ViT 1D
  - vit_snn       → ViT with Spiking attention
  - vit_quantum   → ViT with Quantum path
  - vit_hybrid    → ViT with Spiking + Quantum

All models share the same forward() interface:
    output = model(x, return_gate=True)
    output['logits']   → (B, num_classes)
    output['reg_loss'] → scalar
    output['gate']     → (B, 1)
"""

from typing import Dict
import torch.nn as nn

from v2.resnet import ResNet1D
from v2.vit import ViT1D, ViT1DConfig


MODEL_REGISTRY = {
    'resnet':      'ResNet1D Baseline',
    'vit':         'ViT Standard',
    'vit_snn':     'ViT + SNN',
    'vit_quantum': 'ViT + Quantum',
    'vit_hybrid':  'ViT + SNN + Quantum (Hybrid)',
}


def create_comparison_model(config: dict, model_type: str) -> nn.Module:
    """
    Factory that builds the requested model variant.

    Args:
        config: Full YAML config dict (same format as hybrid_v2.yaml).
        model_type: One of MODEL_REGISTRY keys.

    Returns:
        nn.Module with the unified forward() interface.
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(MODEL_REGISTRY.keys())}"
        )

    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    quantum_cfg = model_cfg.get('quantum', {})
    classifier_cfg = model_cfg.get('classifier', {})

    num_classes = data_cfg.get('num_classes', 5)
    in_channels = model_cfg.get('in_channels', 12)
    input_length = data_cfg.get('input_length', 5000)

    # ── ResNet ──
    if model_type == 'resnet':
        return ResNet1D(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=model_cfg.get('resnet', {}).get('base_channels', 64),
            dropout=classifier_cfg.get('dropout', 0.3),
        )

    # ── ViT variants ──
    vit_cfg_dict = model_cfg.get('vit', {})

    vit_config = ViT1DConfig(
        in_channels=in_channels,
        input_length=input_length,
        num_classes=num_classes,
        patch_size=vit_cfg_dict.get('patch_size', 50),
        embed_dim=vit_cfg_dict.get('embed_dim', 128),
        depth=vit_cfg_dict.get('depth', 4),
        num_heads=vit_cfg_dict.get('num_heads', 4),
        mlp_ratio=vit_cfg_dict.get('mlp_ratio', 4.0),
        dropout=vit_cfg_dict.get('dropout', 0.1),
        use_snn=(model_type in ('vit_snn', 'vit_hybrid')),
        use_quantum=(model_type in ('vit_quantum', 'vit_hybrid')),
        n_qubits=quantum_cfg.get('n_qubits', 8),
        n_q_layers=quantum_cfg.get('n_layers', 3),
        quantum_feature_dim=quantum_cfg.get('feature_dim', 64),
        weight_bits=model_cfg.get('classical', {}).get('weight_bits', 1),
        hidden_dim=classifier_cfg.get('hidden_dim', 64),
        cls_dropout=classifier_cfg.get('dropout', 0.3),
    )

    return ViT1D(vit_config)
