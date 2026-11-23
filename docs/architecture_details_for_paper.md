# Architecture Specifications for Hybrid Quantum-Classical ECG Classifier V2.0

As part of the shift towards "Green AI" in cardiology, deploying models to battery-powered, wearable end-edge devices (e.g., smartwatches, Holter monitors) requires extreme computational efficiency without sacrificing the diagnostic parity achieved by deep learning. The following details the architectural pillars of our framework, completely bypassing heavy, parameter-dense classical models. 

## 1. Core Paradigm: Dual-Path Gated Fusion
The Hybrid V2 architecture relies on a **Dual-Path processing strategy**, combining a Spiking Neural Network (SNN) backbone for computationally sparse temporal processing and an Amplitude-Phase Encoded Variational Quantum Circuit (VQC) for exploring extreme high-dimensional correlations, effectively bound by a Learnable Gated Late-Fusion mechanism. 

### 1.1 The Classical Path: Hierarchical Temporal Spiking Pyramid
Standard ViTs or CNNs execute heavy matrix multiplications continuously, dominating computational budgets. Our approach replaces this standard with a **Hierarchical Temporal Spiking Pyramid**, integrating Leaky Integrate-and-Fire (LIF) neurons.

#### Temporal Stages:
ECG data is inherently composed of multi-scale frequencies. The SNN captures these across three dedicated stages:
1.  **Stage 1: Local Patterns (High-Resolution)**
    *   **Architecture:** Quantized 1D Convolutions (k=3, stride=1) coupled with LIF neurons.
    *   **Function:** Extracts fine-grained morphological details, such as rapid P-wave and T-wave localized alterations, at a sequence length of 5000 timesteps.
    *   **Output:** 256 temporal features.
2.  **Stage 2: Beat-Level Patterns (Mid-Resolution)**
    *   **Architecture:** Multi-Scale Spiking Attention (MSSA, 3 heads) alongside Quantized 1D Convolutions (k=9, stride=2).
    *   **Function:** Focuses on standard cardiac cycles, predominantly capturing the structural integrity of the QRS complex and RR intervals.
    *   **Output:** 128 temporal features.
3.  **Stage 3: Rhythm Patterns (Low-Resolution)**
    *   **Architecture:** Global Spiking Attention alongside Quantized 1D Convolutions (k=27, stride=4).
    *   **Function:** Extracts macroscopic sequence trends, isolating longer-range anomalies such as Heart Rate Variability (HRV) and broader arrhythmic events.
    *   **Output:** 64 temporal features. 

**Neuromorphic Efficiency**: The LIF neurons actively clamp the membrane potentials using 8-bit precision (Binary Quantization). Communication exclusively utilizes discrete binary spikes (1 or 0), mitigating standard matrix multiplication into energy-efficient, sparse addition operations. Weights are simultaneously collapsed utilizing 1-bit quantization steered by the Straight-Through Estimator (STE).

### 1.2 The Quantum Path: Parameterized Variational Quantum Circuit (VQC)
A parallel path independently evaluates the identical input sample via a Hilbert-space embedding. While the SNN path specializes in temporal patterns, the quantum circuit utilizes superposition and quantum entanglement to capture non-linear, inter-lead correlations inherently difficult for classical mechanisms to express with few parameters.

#### Flow of Information:
1.  **Feature Compression:** Standard classical compression (1D Convolutions + Max Pooling) restricts the massive input dimensions down to 64 discrete latent variables.
2.  **State Preparation (Data Encoding):** The 64 compressed features determine the angular trajectory. Using Amplitude and Phase encoding, classical variables execute parallel `RY` and `RZ` rotational bounds onto a simulated 8-qubit register, casting the deterministic input into a complex quantum state vector $|\psi\rangle$.
3.  **Variational Entanglement (Trainable Evolution):** The state progresses through 3 consecutive trainable instances:
    *   **`RY(θ)` Rotations**: Applied iteratively across the 8 qubits based on neural gradients.
    *   **Learned CNOT Entanglement**: Instead of a strict arbitrary topology (e.g., ring or circular), an attention-based mechanism learns to dynamically orchestrate `CNOT` pair alignments. If leads corresponding to subsets of the state demonstrate higher correlations, the parameters shift to selectively entangle respective neighbors.
    *   **`RZ(ω)` Rotations**: Phase rotational tunings based on parameterized weights. 
4.  **Measurement (Expectation Variable):** The collapsed quantum state is recorded via Pauli-Z expectation values across all 8 individual qubits, creating a deterministically classical dense feature vector mapped into dimensions viable for dense alignment.

*(Experimental Verification: We natively replicated these measurements comparing purely vectorized math against the standard PennyLane `default.qubit` simulator framework, confirming mathematically equivalent execution capable of yielding identically robust classifications)*.

### 1.3 The Gated Fusion Module
The terminal phase merges the output from the Classical Spiking Path (128 dimensions) and the Quantum circuit sequence (8 dimensions). In clinical datasets, simple concatenation runs the risk of introducing excessive quantum noise into easily decipherable patterns. 

Our architecture leverages **Learnable Gated Late-Fusion**. 
*   **Gate Definition**: A miniature gate network receives both features concatenated, culminating in a `Sigmoid` activation rendering a scalar $g \in [0, 1]$.
*   **Dynamic Balance**: The ultimate feature representation respects linear interpolation: $h_{fused} = (1 - g) \cdot h_{classical} + g \cdot h_{quantum\_expanded}$.
*   **Interpretability**: On structurally routine ECG samples, the algorithm pushes $g \approx 0$, entirely trusting the resource-efficient SNN path. For uncharacteristic or ambiguous rhythms overlapping inter-lead bounds, $g$ dynamically shifts positively toward 1.0, engaging abstract quantum representation logic to inform final classification.
