# Understanding Quantum Backends in the Hybrid ECG Model

This document explains the purpose, differences, and implementation details of the three different quantum backends (`v2/`, `pennylane_v2/`, and `qiskit_v2/`) built into this project.

---

## 1. Do You Need Both PennyLane and Qiskit?

**No, one is absolutely enough.** 

You only need **one** of these backends to run the quantum portion of the neural network. We implemented three separate versions to give you **options** depending on your experimental goals. All three versions perform the exact same mathematical operations and will output the same tensor shapes to the classical parts of the model.

---

## 2. What Do These Frameworks Actually Do?

In a standard classical neural network, data flows through dense linear layers using matrix multiplication. 

In a **Hybrid Quantum-Classical Network**, we replace some of those dense layers with a **Parametrized Quantum Circuit (PQC)**:
1. The classical network (CNN/SNN) extracts features from the ECG signal.
2. These features are converted into angles (between 0 and $\pi$).
3. The quantum framework uses these angles to rotate simulated subatomic particles called **Qubits** (this is called *Amplitude Encoding*).
4. The framework then applies trainable quantum gates (like `RY` rotations and `CNOT` entanglements) to process the overlapping probabilities of these qubits.
5. Finally, it "measures" the qubits (calculating the `Pauli-Z` expectation value), turning the quantum states back into classical numbers.

PyTorch alone is not designed to simulate quantum physics natively. **PennyLane** and **Qiskit** are specialized scientific libraries that simulate these quantum mechanics accurately and compute the complex gradients required to train them alongside your PyTorch neural network.

---

## 3. What is Their Use? (Which one should I use?)

| Backend | Best Used For | Description |
| :--- | :--- | :--- |
| **Vectorized (`v2/`)** | **Maximum Training Speed** | Uses pure PyTorch tensor operations to mathematically "fake" the quantum simulation. It is highly optimized for GPUs. Use this for rapid iteration, debugging, and initial training. |
| **PennyLane (`pennylane_v2/`)** | **Quantum Machine Learning Research** | PennyLane is built specifically by Xanadu for hybrid QML. It makes testing different embedding strategies incredibly easy and can route your code to actual quantum hardware (like Amazon Braket) if needed. |
| **Qiskit (`qiskit_v2/`)** | **Industry-Standard Quantum Computing** | Developed by IBM, Qiskit is the industry standard for quantum computing. Use this backend if your ultimate goal is to deploy the ECG model onto a real, physical IBM Quantum Computer via the cloud. |

---

## 4. How Are They Implemented?

To ensure the model learns identically regardless of backend, all three implement the exact same quantum architecture:
*   **Encoding:** `RY` rotations on every qubit.
*   **Variational Layers:** A repeating sequence of trainable `RY` rotations $\rightarrow$ Circular `CNOT` entanglement $\rightarrow$ Trainable `RZ` rotations.
*   **Measurement:** `Pauli-Z` expectation values on all qubits.

### Implementation Specifics:

*   **PennyLane (`pennylane_v2/quantum_circuit.py`)** 
    PennyLane treats quantum circuits conceptually like Python functions. We define a `qml.qnode` that executes the quantum gates sequentially on a `default.qubit` simulator. We then wrap this QNode using `qml.qnn.TorchLayer`. To PyTorch, this layer looks and behaves exactly like a standard `nn.Linear` layer—it accepts a loss gradient and updates its internal quantum rotation angles during `.backward()`.
    
*   **Qiskit (`qiskit_v2/quantum_circuit.py`)**
    Qiskit requires building a literal circuit diagram using `QuantumCircuit`. We use `ParameterVector`s to represent the incoming data and the trainable weights. We observe the circuit using `SparsePauliOp` observables and evaluate it using a `StatevectorEstimator`. Finally, we package the circuit into an `EstimatorQNN` and attach a `TorchConnector`, forcing Qiskit to talk to PyTorch's auto-differentiation engine.

### Why Isolated Folders?
Heavy quantum libraries like Qiskit can introduce loading overhead or versioning conflicts. By isolating the implementations into exact copies of the `v2` directory (`pennylane_v2/` and `qiskit_v2/`), you can seamlessly switch your `import` statements at the top level to swap the backend without ever breaking your baseline, lightning-fast Vectorized model.
