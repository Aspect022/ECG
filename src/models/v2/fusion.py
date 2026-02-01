"""
Gated Fusion Module for V2.0 Architecture.

Combines classical and quantum features with learned gating.
Allows model to dynamically balance classical robustness with quantum enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GatedFusionModule(nn.Module):
    """
    Gated Fusion of Classical and Quantum Features.
    
    Learns a gate value g ∈ [0, 1] that controls the contribution
    of quantum features to the final representation:
    
    h_fused = (1 - g) * h_classical + g * h_quantum_expanded
    
    Gate interpretation:
    - g ≈ 0: Model relies purely on classical features
    - g ≈ 1: Model relies on quantum-enhanced features
    - 0 < g < 1: Balanced fusion (typical case)
    
    Args:
        classical_dim: Dimension of classical features (128)
        quantum_dim: Dimension of quantum features (8)
        gate_hidden: Hidden dimension for gate network
    """
    
    def __init__(
        self,
        classical_dim: int = 128,
        quantum_dim: int = 8,
        gate_hidden: int = 64
    ):
        super().__init__()
        
        self.classical_dim = classical_dim
        self.quantum_dim = quantum_dim
        
        # Expand quantum features to match classical dimension
        self.quantum_expansion = nn.Sequential(
            nn.Linear(quantum_dim, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, classical_dim)
        )
        
        # Gate network: predicts fusion weight
        self.gate_network = nn.Sequential(
            nn.Linear(classical_dim + quantum_dim, gate_hidden),
            nn.ELU(),
            nn.Linear(gate_hidden, 1),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(classical_dim)
    
    def forward(
        self,
        classical_features: torch.Tensor,
        quantum_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse classical and quantum features.
        
        Args:
            classical_features: (batch, 128)
            quantum_features: (batch, 8)
        
        Returns:
            Tuple of (fused_features, gate_value)
            - fused_features: (batch, 128)
            - gate_value: (batch, 1) for monitoring
        """
        # Compute gate value
        combined = torch.cat([classical_features, quantum_features], dim=1)
        gate = self.gate_network(combined)  # (batch, 1)
        
        # Expand quantum features
        quantum_expanded = self.quantum_expansion(quantum_features)  # (batch, 128)
        
        # Gated fusion
        fused = (1 - gate) * classical_features + gate * quantum_expanded
        
        # Normalize
        fused = self.norm(fused)
        
        return fused, gate


class ClassificationHead(nn.Module):
    """
    Classification head for ECG disease prediction.
    
    Simple MLP with BatchNorm and Dropout for regularization.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 5,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, num_classes) logits
        """
        return self.classifier(x)
