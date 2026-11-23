# Detailed Model Analysis: SNN, QNN, and Hybrid Architectures

This document provides a comprehensive analysis of the experimental results and the intricate architectural details of the Spiking Neural Network (SNN), Quantum Neural Network (QNN), and their hybrid fusion variant implemented for the 12-lead ECG classification task.

## 1. Experimental Results Summary

The 5-fold cross-validation evaluation across the different models yielded the following metrics:

| Model                            | Test Accuracy | Mean Accuracy (5-folds) | Parameters | Latency (ms) | Energy (mJ) |
| :------------------------------- | :------------ | :---------------------- | :--------- | :----------- | :---------- |
| **ViT + SNN**                    | 98.98%        | 99.96% ± 0.0001         | 1,944,461  | 0.13         | 1.98        |
| **ViT + Quantum**                | 99.03%        | 98.56% ± 0.0049         | 1,170,655  | 0.26         | 5.90        |
| **ViT + SNN + Quantum (Hybrid)** | 98.17%        | 97.96% ± 0.0025         | 2,222,823  | 0.35         | 8.57        |


---

## 2. Global Training & Implementation Settings

Before diving into the specific paths, several global and shared mechanisms are rigorously enabled across all training paradigms:
*   **Early Stopping:** Enabled with a `patience=15`. Training ceases if validation accuracy does not improve by at least `min_delta=0.0` over 15 epochs.
*   **Automatic Mixed Precision (AMP):** Enabled via `torch.cuda.amp.autocast()`, optimizing VRAM usage and accelerating the training loop on the RTX 5050.
*   **Optimizer & Scheduler:** Uses `AdamW` decoupled weight decay coupled with `CosineAnnealingWarmRestarts` for learning rate scheduling.
*   **Cross Validation:** Strict stratified 5-fold cross-validation leveraging specific arrays of `train_folds` and `val_folds`.
*   **Weight Parameter Control:** Gradient norm clipping is strictly enforced at a maximum norm of `1.0`.

---

## 3. Spiking Neural Network (SNN) Path Details

### SNN Type & Architectural Integration
Instead of a standalone model, the SNN is tightly integrated into the ViT sequence explicitly via the **SpikingTransformerBlock**. This block replaces the standard Multi-Head Self Attention (MHSA) with a **MultiScaleSpikingAttention (MSSA)** module.
*   The architecture processes data over a discrete simulation window of **`timesteps=4`**.
*   The MSSA extracts temporal features across multiple receptive fields: `local_window=64`, `regional_window=256`, and `global_pool=16`.
*   A 3-stage temporal pyramid explicitly captures High-res localized patterns (P/T waves), Mid-rank (QRS complex), and Low-res Rhythm patterns (HRV) using strided convolutions.

### BIT LIF (Membrane Potential Quantization)
The models employ the **`QuantLIFNeuron`**, a highly customized Leaky Integrate-and-Fire neuron optimized for hardware constraints:
*   **`potential_bits=8`**: The membrane potential is actively clamped between `[-2.0, 2.0]` and uniformly quantized to 8 bits.
*   **Reset Mechanism:** A "soft reset" is utilized ($V = V \cdot (1 - S)$) rather than a rigid target reset.
*   **Surrogate Gradient:** It leverages a **Fast Sigmoid** estimator (`beta=10.0`) during backpropagation to overcome the non-differentiability of the Heaviside step function.
*   **Hyperparameters:** Firing threshold is initialized at `threshold=1.0`, with a membrane decay constant `tau=2.0`.
*   **Spike Regularization:** An active L2 regularization loss penalizes deviations from an optimal `0.3` firing rate, maintaining energy-efficient sparsity.

### Theoretical Foundation of SNNs
Spiking Neural Networks are often referred to as the third generation of neural networks, inspired directly by biological neurons. Unlike standard artificial neurons that output continuous floating-point values (like ReLUs), SNNs transmit information via discrete, binary events known as "spikes" over time. 

The core of our SNN is the Leaky Integrate-and-Fire (LIF) neuron. The biological equivalent manages a membrane potential $V(t)$ that integrates incoming current $I(t)$ and "leaks" charge over time, governed by the differential equation: $\tau_m \frac{dV}{dt} = -(V - V_{rest}) + R_m I(t)$. When $V$ reaches a threshold $V_{th}$, a spike is emitted, and the potential resets.

### SNN Implementation Snippet
In our implementation, the `QuantLIFNeuron` utilizes a surrogate gradient (Fast Sigmoid) to remain differentiable during backpropagation. Here is the core operational loop of our LIF neuron over discrete simulation timesteps:

```python
for t in range(timesteps):
    # Leak: potential decays over time
    v = self.decay * v
    
    # Integrate: input current is added
    v = v + current
    
    # Quantize membrane potential to hardware-friendly 8-bit
    v = self._quantize_potential(v)
    
    # Fire: generate a spike using Heaviside (with surrogate gradient for backward pass)
    spike = (v >= self.threshold).float()
    
    # Reset: Soft reset mechanism prevents hard information loss
    v = v * (1.0 - spike)
    
    # Accumulate spikes across discrete timesteps
    spike_sum = spike_sum + spike
```

### BIT Type (Weight Quantization)
*   **`weight_bits=1`**: Convolutional weights are constrained using Binary Quantization. 
*   It utilizes a hardware-efficient **Straight-Through Estimator (STE)**. Instead of raw binary weights, they are strategically scaled by their mean absolute value along the channel axes, boosting gradient flow and representational power even at 1-bit constraints.

---

## 4. Quantum Neural Network (QNN) Path Details

The Quantum block operates concurrently by maintaining a vectorized, fully parallel PyTorch simulation (`VectorizedQuantumCircuit`), natively escaping the bottleneck of PennyLane plugin transfers. 

### Quantum Circuit Layout & Layers Enabled
1.  **Amplitude Encoding:** Classical features are compressed and mapped to rotations in the operational range $[0, \pi]$.
2.  **Variational Layers:** Configured with `n_qubits=8` and `n_layers=3`.
3.  **Measurement:** Standard Pauli-Z expectation value measurements track the state of every individual qubit.

### Rotation Specifications
*   **State Preparation:** Initial amplitude encoding applies pure **`RY`** rotations.
*   **Parameterised Variational Rotations:** Dictated by the configurable `rotation_axes` constraint (usually `'yz'`). During each layer step, the circuit subsequently applies parameterized rotational equivalents (e.g., `RY` followed by `RZ`) with independent, trainable parameters initialized around zero ($N(0, 0.1)$).

### Entanglement Strategy
*   **Type:** The circuit enforces a **`circular`** entanglement topology. 
*   **Mechanism:** Implemented using deterministic cascading `CNOT` operations acting on adjacent topological pairs—control qubit $q$ directly entangled to target qubit $(q+1) \pmod N$. 

### Theoretical Foundation of QNNs
Quantum Neural Networks (QNNs) or Parameterized Quantum Circuits (PQCs) represent operations as unitary transformations in a $2^n$-dimensional complex Hilbert space using $n$ qubits. The exponentially large state space allows quantum circuits to capture highly complex, non-linear relationships with fewer parameters than classical counterparts.

The generic parameterized quantum circuit consists of three main phases:
1. **Data Encoding (State Preparation):** Classical continuous data is encoded into quantum state amplitudes, pivoting the state vector $|\psi\rangle$ on the Bloch sphere.
2. **Trainable Evolution:** Parametrized unitary gates (e.g., $U(\theta)$) are varied during training to minimize the given loss function.
3. **Measurement:** Measuring an observable (like the Pauli-Z operator, $\sigma_z$) collapses the state to provide classical expectation values for the final fully-connected layers.

### Quantum Circuit Implementation Snippet
Instead of relying on slow CPU-bound plugins like PennyLane, our `VectorizedQuantumCircuit` natively simulates quantum operations via pure PyTorch tensors, unlocking massive parallel compute. Here's a glimpse into the vectorized batch Pauli-Z measurement:

```python
def measure(self, state: torch.Tensor) -> torch.Tensor:
    batch_size = state.shape[0]
    expectations = []
    
    for i in range(self.n_qubits):
        # Retrieve the precomputed Pauli-Z observable
        obs = self.pauli_z_obs[i].to(state.device)
        
        # Calculate Expectation: ⟨ψ|Z|ψ⟩ 
        obs_state = torch.matmul(obs, state.T).T
        
        # Inner product: sum over state dimension
        expectation = torch.real(torch.sum(state.conj() * obs_state, dim=1))
        expectations.append(expectation)
        
    return torch.stack(expectations, dim=1) # (batch, n_qubits)
```

---

## 5. The Hybrid Architecture: Gated Fusion

For the **ViT + SNN + Quantum (Hybrid)** model, the architecture maintains an explicit dual-branch execution: 
*   **Path A:** Spiking ViT Backbone (yielding classical/spiking features).
*   **Path B:** Vectorized Quantum Circuit (yielding quantum expectations).

### Fusion Architecture
Upon conclusion of both independent paths, the network integrates the representations within the **`GatedFusionModule`**. 
*   The quantum features are expanded linearly to map against the 128-D dimensions of the classical ViT features.
*   A dense neural gate evaluates a concatenated array of `[classical, quantum]`. It projects down through an ELU activation into a scalar **`Sigmoid`** gate weight ($g \in [0, 1]$).
*   The features are finally blended using linear interpolation: 
    $$h_{fused} = (1 - g) \cdot h_{classical} + g \cdot h_{quantum\_expanded}$$
*   The output undergoes `LayerNorm` stabilization before proceeding to the ML classification head.

### Reason for Fusing
The explicit rationale defining the gated fusion is to mathematically empower the model to **dynamically balance classical robustness with quantum enhancement**. Rather than rigidly forcing both representations on every sample, the learned scalar parameter $g$ dynamically assesses whether local deterministic features (classical SNN/ViT processing) alone are sufficient ($g \approx 0$), or if the abstract representational spaces created by variations in Hilbert space ($g \approx 1$) yield better classification boundaries on ambiguous electrocardiogram signals.

### Theoretical Foundation of Gated Fusion
In clinical environments like ECG analysis, no single processing modality acts impeccably on every signal instance. Some signals present highly canonical morphologies perfect for the classical SNN tree. In contrast, noisy or highly atypical signals may benefit from the abstract correlation spaces explored by the parallel quantum circuit's Hilbert space embedding.

Gated Late Fusion creates a mechanism of **Learned Modality Specialization**. By utilizing a differentiable gating network $g(X)$, the architecture dynamically routes feature selection based on the input signal itself, naturally shifting its attention to the representation space (Classical vs. Quantum) that affords the highest confidence.

### Fusion Implementation Snippet
The implementation of this dynamic differential routing is concise but extremely powerful, maintaining a smooth gradient flow across both distinct modalities:

```python
# Compute dynamic gate value based on concatenated features
combined = torch.cat([classical_features, quantum_features], dim=1)
gate = self.gate_network(combined)  # (batch, 1), normalized via Sigmoid

# Project quantum features into classical vector space
quantum_expanded = self.quantum_expansion(quantum_features)  # (batch, 128)

# Differentiable Gated Interpolation
fused = (1 - gate) * classical_features + gate * quantum_expanded

# Final normalization before the classification head
fused = self.norm(fused)
return fused, gate
```
