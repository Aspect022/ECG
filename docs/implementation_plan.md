# Implementation Plan: Cleanup, SNN Fixes, Quantum Repair, and 5-Fold CV

## Goal Description
Clean up the project, fix the broken "Learned Entanglement" in the Quantum Circuit, ensuring the "SNN" is truly spiking (replacing ELU), optimizng for <50k parameters, and implementing robust 5-Fold Cross-Validation.

## User Review Required
- **Critical Quantum Bug**: The "Learned Entanglement" used `argmax`, which broke gradient flow. The quantum circuit was **not learning** its topology. I will replace it with **Standard Circular Entanglement** (CNOTs between neighbors), which is robust and fully differentiable.
- **SNN Fix**: Replacing non-spiking `ELU` with `QuantLIFNeuron` in Stages 2 & 3.
- **Cleanup**: Deleting `baselines`, `heads`, and `scripts`.

## Proposed Changes

### 1. Project Cleanup & Restructuring
- **DELETE**: `src/models/baselines/` (ResNet1d).
- **DELETE**: `src/models/heads/` (Unused).
- **CLEAN**: `scripts/`.

### 2. Quantum Circuit Repair (`src/models/v2/quantum_circuit.py`)
- **Fix**: Remove the non-differentiable `entangle_attn` and `argmax` logic.
- **Implement**: **Circular Entanglement**.
    - For each layer, apply CNOT on $(i, (i+1) \% n_{qubits})$.
    - This creates a strongly entangling ring topology that allows gradients to flow perfectly to the rotation parameters.

### 3. SNN & Model Optimization (`src/models/v2/classical_path.py`)
- **Fix SNN**: Replace `F.elu` in Stages 2 & 3 with `QuantLIFNeuron`.
- **Param Reduction**: Switch to **Depthwise Separable Convolutions** in Stages 2 & 3 to hit **< 50,000 parameters**.

### 4. Data Loader Refactor (`src/data/ptbxl.py`)
- Update `PTBXLDataset` to accept dynamic `folds` list.

### 5. 5-Fold Cross-Validation (`train_hybrid_v2.py`)
- Implement 5-Fold Loop with metric aggregation (Accuracy, F1, Latency, Energy).

## Verification Plan

### Automated Tests
- **Gradient Check**: I will run a small script to verify that `quantum_circuit.parameters()` receive non-zero gradients (proving the fix works).
- **Parameter Check**: Verify params < 50k.

### Manual Verification
- **Training Logs**: Monitor 5-Fold training. Validation accuracy should behave realistically.
