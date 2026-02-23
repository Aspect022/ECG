"""
Test suite to verify that all three quantum backends (Vectorized, PennyLane, Qiskit)
produce correctly shaped outputs and support PyTorch's backward pass.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src/models to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

# Import the backends
from v2.quantum_circuit import VectorizedQuantumCircuit, QuantumMeasurement


def test_backend(name: str, circuit: nn.Module, n_qubits: int, batch_size: int):
    print(f"\n--- Testing Backend: {name} ---")
    
    # Dummy input: angles normalized between [0, pi]
    x = torch.rand(batch_size, n_qubits) * 3.14159
    x.requires_grad_(True)
    
    try:
        # Forward Pass
        if name == "Vectorized":
            measurement = QuantumMeasurement(n_qubits)
            state = circuit(x)
            output = measurement(state)
        else:
            # PennyLane and Qiskit have built-in measurement in our wrappers
            output = circuit(x)
            
        print(f"Forward Pass OK. Output shape: {output.shape}")
        assert output.shape == (batch_size, n_qubits), f"Expected shape ({batch_size}, {n_qubits}), got {output.shape}"
        
        # Dummy loss (MSE vs zeros)
        target = torch.zeros_like(output)
        loss = nn.MSELoss()(output, target)
        
        # Backward Pass
        loss.backward()
        
        print(f"Backward Pass OK. Loss: {loss.item():.4f}")
        assert x.grad is not None, "Input gradients not computed!"
        
        print(f"[OK] {name} backend passed.")
        return True
        
    except ImportError as e:
        print(f"⚠️ Skipping {name} backend due to missing package: {e}")
        return False
    except Exception as e:
        print(f"[FAIL] {name} backend FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    n_qubits = 4  # Small for fast local testing
    n_layers = 2
    batch_size = 2
    
    print(f"Testing with n_qubits={n_qubits}, n_layers={n_layers}, batch_size={batch_size}")
    
    # 1. Vectorized (Pure PyTorch)
    test_backend(
        "Vectorized",
        VectorizedQuantumCircuit(n_qubits, n_layers),
        n_qubits,
        batch_size
    )
    
    # 2. PennyLane
    try:
        from pennylane_v2.quantum_circuit import PennyLaneQuantumCircuit
        test_backend(
            "PennyLane",
            PennyLaneQuantumCircuit(n_qubits, n_layers),
            n_qubits,
            batch_size
        )
    except ImportError:
        print("PennyLane not installed. Skipping.")
        
    # 3. Qiskit
    try:
        from qiskit_v2.quantum_circuit import QiskitQuantumCircuit
        test_backend(
            "Qiskit",
            QiskitQuantumCircuit(n_qubits, n_layers),
            n_qubits,
            batch_size
        )
    except ImportError:
        print("Qiskit not installed. Skipping.")
