"""
PennyLane Quantum Circuit for V2.0 Architecture.

Replaces the pure-PyTorch VectorizedQuantumCircuit with a PennyLane
QNode wrapped via qml.qnn.TorchLayer for seamless PyTorch integration.

Architecture (mirrors the vectorized version exactly):
- Amplitude Encoding: RY(x_i) on each qubit
- Variational Layers: RY(theta) + Circular CNOT + RZ(omega)
- Measurement: Pauli-Z expectation on each qubit
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

try:
    import pennylane as qml
except ImportError:
    raise ImportError(
        "PennyLane is required for this module. "
        "Install it with: pip install pennylane"
    )


class PennyLaneQuantumCircuit(nn.Module):
    """
    PennyLane Variational Quantum Circuit.

    Uses qml.qnn.TorchLayer to wrap a QNode, making it a drop-in
    replacement for VectorizedQuantumCircuit + QuantumMeasurement.

    Input:  (batch, n_qubits)  — angles in [0, π]
    Output: (batch, n_qubits)  — Pauli-Z expectation values

    Args:
        n_qubits: Number of qubits (default: 8)
        n_layers: Number of variational layers (default: 3)
    """

    def __init__(self, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # PennyLane device (statevector simulator)
        dev = qml.device("default.qubit", wires=n_qubits)

        # Define the QNode
        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, theta, omega):
            """
            Variational quantum circuit.

            Args:
                inputs: (n_qubits,) encoding angles
                theta:  (n_layers, n_qubits) RY rotation params
                omega:  (n_layers, n_qubits) RZ rotation params
            """
            # === Amplitude Encoding ===
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # === Variational Layers ===
            for layer in range(n_layers):
                # RY rotations (trainable)
                for i in range(n_qubits):
                    qml.RY(theta[layer, i], wires=i)

                # Circular entanglement: CNOT(i, (i+1) % n)
                for i in range(n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % n_qubits])

                # RZ rotations (trainable)
                for i in range(n_qubits):
                    qml.RZ(omega[layer, i], wires=i)

            # Measure Pauli-Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Define weight shapes matching the vectorized version
        weight_shapes = {
            "theta": (n_layers, n_qubits),
            "omega": (n_layers, n_qubits),
        }

        # Initialize weights to match vectorized version (mean=0, std=0.1)
        init_method = {
            "theta": lambda t: nn.init.normal_(t, mean=0.0, std=0.1),
            "omega": lambda t: nn.init.normal_(t, mean=0.0, std=0.1),
        }

        # Wrap into TorchLayer
        self.qlayer = qml.qnn.TorchLayer(
            circuit,
            weight_shapes=weight_shapes,
            init_method=init_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode features and apply variational circuit.

        Args:
            x: (batch, n_qubits) classical features in [0, π]

        Returns:
            (batch, n_qubits) Pauli-Z expectation values
        """
        return self.qlayer(x)


# Backwards-compatible aliases so imports don't break
QuantumGates = None  # Not needed for PennyLane backend
VectorizedQuantumCircuit = PennyLaneQuantumCircuit


class QuantumMeasurement(nn.Module):
    """
    No-op measurement module for PennyLane backend.

    Measurement is built into the QNode, so this is an identity pass-through
    to maintain API compatibility with quantum_path.py.
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Identity pass-through (measurement already done in QNode)."""
        return state
