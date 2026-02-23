"""
Qiskit Quantum Circuit for V2.0 Architecture.

Replaces the pure-PyTorch VectorizedQuantumCircuit with a Qiskit
QuantumCircuit wrapped via TorchConnector for seamless PyTorch integration.

Architecture (mirrors the vectorized version exactly):
- Amplitude Encoding: RY(x_i) on each qubit
- Variational Layers: RY(theta) + Circular CNOT + RZ(omega)
- Measurement: Pauli-Z expectation on each qubit
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.primitives import StatevectorEstimator
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
except ImportError:
    raise ImportError(
        "Qiskit and Qiskit Machine Learning are required. "
        "Install with: pip install qiskit qiskit-machine-learning"
    )


class QiskitQuantumCircuit(nn.Module):
    """
    Qiskit Variational Quantum Circuit.

    Uses TorchConnector to wrap an EstimatorQNN, making it a drop-in
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

        # Define Qiskit Parameters
        self.inputs = ParameterVector('input', n_qubits)
        self.theta = ParameterVector('theta', n_layers * n_qubits)
        self.omega = ParameterVector('omega', n_layers * n_qubits)

        # Build Quantum Circuit
        qc = QuantumCircuit(n_qubits)

        # === Amplitude Encoding ===
        for i in range(n_qubits):
            qc.ry(self.inputs[i], i)

        # === Variational Layers ===
        theta_idx = 0
        omega_idx = 0

        for layer in range(n_layers):
            # RY rotations (trainable)
            for i in range(n_qubits):
                qc.ry(self.theta[theta_idx], i)
                theta_idx += 1

            # Circular entanglement: CNOT(i, (i+1) % n)
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)

            # RZ rotations (trainable)
            for i in range(n_qubits):
                qc.rz(self.omega[omega_idx], i)
                omega_idx += 1

        # Define Observables for each qubit: Z_0, Z_1, ..., Z_{n-1}
        # Qiskit uses little-endian ordering, so 'Z' at index i corresponds to qubit i.
        # e.g. for qubit 0 out of 8: "IIIIIIIZ"
        observables = []
        for i in range(n_qubits):
            pauli_string = ['I'] * n_qubits
            pauli_string[n_qubits - 1 - i] = 'Z'
            observables.append(SparsePauliOp("".join(pauli_string)))

        # Create StatevectorEstimator
        estimator = StatevectorEstimator()

        # Build QNN
        qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=self.inputs,
            weight_params=list(self.theta) + list(self.omega),
            input_gradients=True, # Critical for backprop if part of a larger hybrid net
            estimator=estimator
        )

        # Initial weights matching (mean=0, std=0.1) like vectorized
        initial_weights = torch.randn(qnn.num_weights) * 0.1

        # Wrap with TorchConnector
        self.qlayer = TorchConnector(qnn, initial_weights=initial_weights)

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
QuantumGates = None  # Not needed for Qiskit backend
VectorizedQuantumCircuit = QiskitQuantumCircuit


class QuantumMeasurement(nn.Module):
    """
    No-op measurement module for Qiskit backend.

    Measurement is built into the EstimatorQNN, so this is an identity 
    pass-through to maintain API compatibility with quantum_path.py.
    """

    def __init__(self, n_qubits: int = 8):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Identity pass-through (measurement already done in EstimatorQNN)."""
        return state
