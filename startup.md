# Onboarding & Startup Guide: Hybrid Quantum-Spiking ECG Classification

This repository contains the implementation of a journal-grade **Hybrid Quantum-Classical-Spiking Neural Network (SNN)** for multi-label ECG classification. 

The architecture leverages a dual-path design:
1. **Classical-Spiking Path**: Extracts temporal-spectral features using a 1D Convolutional network, Leaky Integrate-and-Fire (LIF) spiking neurons, and a Transformer encoder.
2. **Quantum Path**: Encodes signal features into quantum states and uses a Vectorized Variational Quantum Circuit (VQC) with Pauli-Z measurements.
3. **Gated Cross-Fusion**: Merges features dynamically using a learnable gating mechanism to optimize predictions.

---

## 💻 System & Hardware Prerequisites

* **Operating System**: Windows / Linux
* **GPU**: NVIDIA GeForce RTX 5050 (typical 4GB/6GB VRAM) or higher. CUDA 11.8 or 12.1+ compatible.
* **Python**: Version `3.9` to `3.11` recommended.

---

## ⚙️ Installation & Setup

Follow these steps to set up the environment on the target training device (RTX 5050):

### 1. Clone & Navigate to Repository
Open your terminal/Command Prompt and navigate to the project directory:
```bash
cd path/to/ECG
```

### 2. Create and Activate Virtual Environment
* **On Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
* **On Linux**:
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### 3. Install PyTorch with CUDA Support
Before installing other packages, install the CUDA-enabled version of PyTorch corresponding to your GPU driver:
* **CUDA 11.8**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```
* **CUDA 12.1**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### 4. Install Project Dependencies
Install the remaining packages required for spiking networks, quantum circuits, and dataset processing:
```bash
pip install -r requirements.txt
```
*(Note: If `torchquantum` or `spikingjelly` issues occur, verify they align with the versions specified in the `requirements.txt` / `safe_requirements.txt`)*

---

## 📊 Dataset Acquisition & Placement

The codebase is pre-configured to train on the **PTB-XL ECG Dataset** using the **superclass** classification task (5 classes: `NORM`, `MI`, `STTC`, `CD`, `HYP`).

### 1. Download PTB-XL
Download the dataset from PhysioNet:
* **Link**: [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
* **Direct ZIP Download**: [ptb-xl-a-large-publicly-available-electrocardiography-database-1.0.3.zip](https://physionet.org/content-ptb-xl-1.0.3.zip) (approx. 1.7 GB)

### 2. Create the `data` Folder
In the root of the project directory, create a folder named `data/ptbxl`:
```bash
mkdir -p data/ptbxl
```

### 3. Extract the Dataset
Extract the contents of the downloaded ZIP archive directly into `data/ptbxl`. Ensure the directory structure looks **exactly** like this:

```text
d:\Projects\ECG\
├── configs/
├── src/
├── data/
│   └── ptbxl/
│       ├── ptbxl_database.csv
│       ├── scp_statements.csv
│       ├── records100/
│       │   ├── 00000/
│       │   │   ├── 00001_lr.dat
│       │   │   ├── 00001_lr.hea
│       │   │   └── ...
│       │   └── ...
│       └── records500/
│           ├── 00000/
│           │   ├── 00001_hr.dat
│           │   ├── 00001_hr.hea
│           │   └── ...
│           └── ...
```

---

## 🔍 Step-by-Step Execution Guide

Once the dataset is placed correctly, run the following verification and training scripts.

### Step 1: Verify Gradient Flow (No Dataset Required)
Run this lightweight test script to verify that all SNN custom surrogate gradients, VQC layers, and Gated Fusion layers propagate gradients cleanly without any autograd mutations or crashes:
```bash
python verify_gradient_flow.py
```
* **Expected Output**: `VERIFICATION SUCCESSFUL: ALL CRASHES FIXED & GRADIENT FLOW FULLY RESTORED!`

### Step 2: Verify Dataset Consistency (Dataset Required)
Verify that the PTB-XL dataset parses correctly, handles empty multi-labels appropriately, and maintains class mappings consistently between train and validation splits:
```bash
python verify_fixes.py
```
* **Expected Output**: `[PASS] Train and Val classes match exactly.` followed by `Verification Complete`.

### Step 3: Run the Unified Comparison Pipeline
To train and evaluate the baselines alongside the main model under identical training conditions, run:
```bash
python run_pipeline.py --epochs 50
```
This runs the training sequentially for:
1. **ViT + SNN** (Baseline 1)
2. **ViT + Quantum** (Baseline 2)
3. **Hybrid V2 Model** (Our main dual-path fused model)

*(Use `--debug` to run a fast check with small batches and 2 epochs: `python run_pipeline.py --epochs 2 --debug`)*

---

## ⚡ NVIDIA RTX 5050 Performance Tuning

Since the RTX 5050 is a mobile GPU typically equipped with **4GB or 6GB of VRAM**, use these optimizations in the configuration files if you encounter Out-Of-Memory (OOM) errors:

1. **Reduce Batch Size**:
   Open `configs/hybrid_v2.yaml` or `configs/comparison.yaml` and reduce `batch_size`:
   ```yaml
   data:
     batch_size: 16  # (Reduced from 32/64)
   ```
2. **Gradient Accumulation**:
   For training stable models at smaller batch sizes, use gradient accumulation in `configs/comparison.yaml`:
   ```yaml
   data:
     accumulation_steps: 4  # Simulates effective batch size of (batch_size * accumulation_steps)
   ```
3. **Number of Dataloader Workers**:
   Ensure `num_workers: 4` is set (both in configs and when executing) to balance CPU pre-fetching with GPU execution without bottlenecks.
4. **Use AMP (Automatic Mixed Precision)**:
   Ensure `use_amp: true` remains enabled in configs. This runs operations in FP16, saving ~50% GPU memory and speeding up execution on Tensor Cores.

---

## 📈 Metric Tracking & TensorBoard

All experiments log training loss, class-wise accuracies, precision, recall, F1 scores, ROC-AUC, and gating coefficient distributions.

To visualize metrics in real-time during training:
1. Run TensorBoard:
   ```bash
   tensorboard --logdir runs
   ```
2. Open your browser and navigate to: [http://localhost:6006](http://localhost:6006)

---

## 💾 Model Checkpoints

Model checkpoints are stored automatically during training:
* **Hybrid V2 Checkpoints**: Saved in `experiments/` (only the best validation performance and the last epoch are saved to conserve disk space).
* **Comparison Checkpoints**: Saved in `experiments/comparison/{timestamp}/`.
