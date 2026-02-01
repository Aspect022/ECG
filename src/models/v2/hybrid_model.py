"""
Hybrid Quantum-Classical ECG Model V2.0

Complete dual-path architecture combining:
- Classical Path: Hierarchical Temporal Feature Pyramid
- Quantum Path: Vectorized Variational Quantum Circuit
- Gated Fusion: Learned classical/quantum balance

Optimized for RTX 5050 (8GB VRAM).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from v2.classical_path import ClassicalPath
from v2.quantum_path import QuantumPath
from v2.fusion import GatedFusionModule, ClassificationHead


@dataclass
class HybridModelConfig:
    """Configuration for Hybrid Quantum-Classical Model."""
    # Input
    in_channels: int = 12
    input_length: int = 5000
    num_classes: int = 5
    
    # Classical path
    stage1_channels: int = 256
    stage2_channels: int = 128
    stage3_channels: int = 64
    fusion_dim: int = 128
    
    # Quantum path
    n_qubits: int = 8
    n_layers: int = 3
    quantum_feature_dim: int = 64
    
    # Quantization
    weight_bits: int = 1
    
    # SNN
    timesteps: int = 4
    
    # Classifier
    hidden_dim: int = 64
    dropout: float = 0.3


class HybridQuantumClassicalECG(nn.Module):
    """
    Hybrid Quantum-Classical ECG Classifier V2.0.
    
    Dual-path architecture:
    - Classical Path: 3-stage hierarchical temporal pyramid
    - Quantum Path: Compressed features → VQC → Measurement
    - Gated Fusion: Learned weighting of paths
    - Classification Head: MLP for disease prediction
    
    Args:
        config: HybridModelConfig with all hyperparameters
    """
    
    def __init__(self, config: Optional[HybridModelConfig] = None):
        super().__init__()
        
        if config is None:
            config = HybridModelConfig()
        
        self.config = config
        
        # ===== CLASSICAL PATH =====
        self.classical_path = ClassicalPath(
            in_channels=config.in_channels,
            stage1_channels=config.stage1_channels,
            stage2_channels=config.stage2_channels,
            stage3_channels=config.stage3_channels,
            fusion_dim=config.fusion_dim,
            weight_bits=config.weight_bits,
            timesteps=config.timesteps
        )
        
        # ===== QUANTUM PATH =====
        self.quantum_path = QuantumPath(
            in_channels=config.in_channels,
            feature_dim=config.quantum_feature_dim,
            n_qubits=config.n_qubits,
            n_layers=config.n_layers,
            weight_bits=config.weight_bits
        )
        
        # ===== GATED FUSION =====
        self.fusion = GatedFusionModule(
            classical_dim=config.fusion_dim,
            quantum_dim=config.n_qubits,
            gate_hidden=config.hidden_dim
        )
        
        # ===== CLASSIFICATION HEAD =====
        self.classifier = ClassificationHead(
            input_dim=config.fusion_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hybrid model.
        
        Args:
            x: (batch, 12, 5000) - 12-lead ECG signal
            return_gate: Whether to return gate values for monitoring
        
        Returns:
            Dict with:
            - 'logits': (batch, num_classes) classification logits
            - 'reg_loss': scalar regularization loss
            - 'gate': (batch, 1) gate values (if return_gate=True)
        """
        # ===== CLASSICAL PATH =====
        classical_features, classical_reg_loss = self.classical_path(x)
        # classical_features: (batch, 128)
        
        # ===== QUANTUM PATH =====
        quantum_features, quantum_reg_loss = self.quantum_path(x)
        # quantum_features: (batch, n_qubits)
        
        # ===== FUSION =====
        fused_features, gate_value = self.fusion(
            classical_features, quantum_features
        )
        # fused_features: (batch, 128)
        
        # ===== CLASSIFICATION =====
        logits = self.classifier(fused_features)
        # logits: (batch, num_classes)
        
        # Total regularization loss
        total_reg_loss = classical_reg_loss + quantum_reg_loss
        
        output = {
            'logits': logits,
            'reg_loss': total_reg_loss
        }
        
        if return_gate:
            output['gate'] = gate_value
        
        return output
    
    def get_classical_only_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Get predictions using only classical path (for ablation)."""
        classical_features, _ = self.classical_path(x)
        # Use fusion with zero quantum features
        dummy_quantum = torch.zeros(
            x.shape[0], self.config.n_qubits, device=x.device
        )
        fused, _ = self.fusion(classical_features, dummy_quantum)
        return self.classifier(fused)
    
    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @property
    def model_size_mb(self) -> float:
        """Estimated model size in MB."""
        # Binary weights are 1-bit, other params are 32-bit
        total_bits = 0
        for name, param in self.named_parameters():
            if 'weight' in name and 'QuantizedConv' in str(type(param)):
                total_bits += param.numel() * self.config.weight_bits
            else:
                total_bits += param.numel() * 32
        return total_bits / 8 / 1024 / 1024


def create_model(
    num_classes: int = 5,
    n_qubits: int = 8,
    weight_bits: int = 1,
    **kwargs
) -> HybridQuantumClassicalECG:
    """
    Factory function to create hybrid model.
    
    Args:
        num_classes: Number of output classes
        n_qubits: Number of qubits in quantum circuit
        weight_bits: Bit-width for weight quantization
        **kwargs: Additional config overrides
    
    Returns:
        Configured HybridQuantumClassicalECG model
    """
    config = HybridModelConfig(
        num_classes=num_classes,
        n_qubits=n_qubits,
        weight_bits=weight_bits,
        **kwargs
    )
    return HybridQuantumClassicalECG(config)


# Test block
if __name__ == "__main__":
    # Quick sanity check
    print("Creating model...")
    model = create_model(num_classes=5, n_qubits=8)
    print(f"Parameters: {model.num_parameters:,}")
    
    # Test forward pass
    print("Testing forward pass...")
    x = torch.randn(4, 12, 5000)
    output = model(x, return_gate=True)
    
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Gate mean: {output['gate'].mean().item():.4f}")
    print(f"Reg loss: {output['reg_loss'].item():.6f}")
    print("✅ Model test passed!")
