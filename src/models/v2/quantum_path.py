"""
Quantum Path for V2.0 Architecture.

Implements:
- Feature Compressor: ECG (12, 5000) -> 64 compressed features
- Quantum Encoding Layer: Classical -> Quantum state
- QuantumPath: Full quantum processing pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quantization import QuantizedConv1d, QuantLIFNeuron
from v2.quantum_circuit import VectorizedQuantumCircuit, QuantumMeasurement


class FeatureCompressor(nn.Module):
    """
    Compress 12-lead ECG to quantum-encodable feature vector.
    
    Pipeline:
    1. Quantized Conv: (12, 5000) -> (64, 1000)
    2. LIF neuron layer
    3. MaxPool: (64, 1000) -> (64, 62)
    4. Global average pool + linear -> 64 features
    
    Input: (batch, 12, 5000)
    Output: (batch, 64)
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        hidden_channels: int = 64,
        output_features: int = 64,
        weight_bits: int = 1
    ):
        super().__init__()
        
        # Downsample convolution
        self.conv1 = QuantizedConv1d(
            in_channels, hidden_channels,
            kernel_size=5, stride=5, padding=2,
            weight_bits=weight_bits
        )
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        
        # LIF neuron
        self.lif = QuantLIFNeuron(
            threshold=1.0, tau=2.0,
            potential_bits=8
        )
        
        # MaxPool for further compression
        self.pool = nn.MaxPool1d(kernel_size=16, stride=16)
        
        # Calculate intermediate size: 5000 // 5 = 1000 -> 1000 // 16 = 62
        self.fc = nn.Linear(hidden_channels * 62, output_features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000)
        Returns:
            Tuple of (features, reg_loss)
            - features: (batch, 64)
        """
        # Conv + BN
        x = self.conv1(x)  # (batch, 64, 1000)
        x = self.bn1(x)
        
        # LIF neuron
        x, _, reg_loss = self.lif(x, timesteps=4)  # (batch, 64, 1000)
        
        # MaxPool
        x = self.pool(x)  # (batch, 64, 62)
        
        # Flatten and project
        x = x.flatten(1)  # (batch, 64*62)
        x = self.fc(x)  # (batch, 64)
        
        return x, reg_loss


class QuantumEncodingLayer(nn.Module):
    """
    Encode classical features into quantum-ready angles.
    
    Projects features to qubit dimension and normalizes to [0, π].
    
    Input: (batch, feature_dim)
    Output: (batch, n_qubits) angles in [0, π]
    """
    
    def __init__(self, feature_dim: int = 64, n_qubits: int = 8):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Project to qubit dimension
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ELU(),
            nn.Linear(32, n_qubits)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_dim)
        Returns:
            (batch, n_qubits) angles in [0, π]
        """
        x = self.projection(x)
        # Normalize to [0, π] using sigmoid
        x = torch.sigmoid(x) * np.pi
        return x


class QuantumPath(nn.Module):
    """
    Complete Quantum Processing Path.
    
    Pipeline:
    1. Feature compression: ECG -> 64 features
    2. Quantum encoding: 64 features -> 8 angles
    3. Variational circuit: Amplitude encoding + variational layers
    4. Measurement: Pauli-Z expectations -> 8 quantum features
    
    Input: (batch, 12, 5000)
    Output: (batch, n_qubits), reg_loss
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        feature_dim: int = 64,
        n_qubits: int = 8,
        n_layers: int = 3,
        weight_bits: int = 1
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        
        # Feature compression
        self.compressor = FeatureCompressor(
            in_channels=in_channels,
            output_features=feature_dim,
            weight_bits=weight_bits
        )
        
        # Quantum encoding
        self.encoder = QuantumEncodingLayer(
            feature_dim=feature_dim,
            n_qubits=n_qubits
        )
        
        # Variational quantum circuit
        self.circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers
        )
        
        # Measurement
        self.measurement = QuantumMeasurement(n_qubits=n_qubits)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000) - 12-lead ECG
        Returns:
            Tuple of (quantum_features, reg_loss)
            - quantum_features: (batch, n_qubits)
        """
        # Compress features
        compressed, reg_loss = self.compressor(x)  # (batch, 64)
        
        # Encode to quantum angles
        angles = self.encoder(compressed)  # (batch, n_qubits)
        
        # Apply quantum circuit
        quantum_state = self.circuit(angles)  # (batch, 2^n_qubits)
        
        # Measure
        quantum_features = self.measurement(quantum_state)  # (batch, n_qubits)
        
        return quantum_features, reg_loss
