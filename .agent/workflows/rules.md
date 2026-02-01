# ECG Classification Project - Agent Guidelines

## Project Context
This is a **journal-level research project** for ECG classification using deep learning. All code must meet publication standards with rigorous documentation, reproducibility, and comprehensive metric tracking.

---

## Core Principles

### 0. Hardware Environment (CPU-Only)

**Target Hardware**: AMD Ryzen 5000 Series (6-8 cores, no GPU)

```python
# ALWAYS include this at the start of training scripts
import torch
import os

def configure_cpu_environment():
    """Configure optimal settings for AMD Ryzen 5000 CPU training."""
    # Set thread count to physical cores (not hyperthreads)
    PHYSICAL_CORES = int(os.environ.get('CPU_CORES', 6))
    torch.set_num_threads(PHYSICAL_CORES)
    os.environ['OMP_NUM_THREADS'] = str(PHYSICAL_CORES)
    os.environ['MKL_NUM_THREADS'] = str(PHYSICAL_CORES)
    
    # Enable MKL-DNN (works well on AMD)
    torch.backends.mkldnn.enabled = True
    
    # For quantization, use fbgemm backend
    torch.backends.quantized.engine = 'fbgemm'
    
    return PHYSICAL_CORES
```

**Batch Size Guidelines**:
- Default: 16 (reduced from 32)
- For large models (ViT, Spiking ViT): 8
- For small models (CNN baseline): 32

**Training Time Expectations**:
- Quick experiments: 1-2 hours
- Full training: 3-8 hours (overnight acceptable)
- Hyperparameter search: Run one config at a time

### 1. Code Quality Standards

- **Docstrings**: Every function, class, and module MUST have comprehensive docstrings
  ```python
  def train_model(model, train_loader, config):
      """
      Train the model for one epoch.
      
      Args:
          model: PyTorch model to train
          train_loader: DataLoader for training data
          config: Configuration dict with training params
      
      Returns:
          dict: Training metrics including loss and accuracy
          
      Example:
          >>> metrics = train_model(model, loader, config)
          >>> print(metrics['accuracy'])
          0.92
      """
  ```

- **Type Hints**: Required for all function signatures
  ```python
  def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
  ```

- **Logging**: Use Python's logging module, never print statements in production code

### 2. Reproducibility Requirements

- **ALL experiments MUST set random seeds**:
  ```python
  def set_seed(seed: int = 42):
      random.seed(seed)
      np.random.seed(seed)
      torch.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  ```

- **Save complete configuration** with every checkpoint
- **Log all hyperparameters** at the start of each experiment
- **Record Git commit hash** if available

### 3. Metric Tracking Protocol

**Every training run MUST log the following:**

#### Classification Metrics (per-class AND macro):
- Accuracy
- Precision
- Recall  
- F1-Score
- ROC-AUC
- PR-AUC
- Specificity
- Sensitivity
- Matthews Correlation Coefficient (MCC)

#### Efficiency Metrics:
- Parameter count (total and trainable)
- FLOPs / MACs
- Model size in MB (float32)
- Model size in MB (quantized, if applicable)
- Training time per epoch
- Inference latency (ms, averaged over 100 runs)
- Throughput (samples/second)

#### Quantization Metrics:
- Accuracy drop from quantization
- Size reduction ratio
- Speedup factor
- Energy savings (for SNN)

### 4. File Naming Conventions

```
Checkpoints:
  {model_name}_{dataset}_{metric:.4f}_ep{epoch}.pt
  Example: cnn_snn_transformer_ptbxl_0.9234_ep45.pt

Logs:
  {model_name}_{dataset}_{timestamp}.csv
  Example: vit_ecg_images_20241214_093000.csv

Figures:
  {metric_type}_{model_name}_{dataset}.png
  Example: confusion_matrix_snn_ptbxl.png
  Example: roc_curve_vit_mitbih.png

Results Tables:
  results_{experiment_name}_{timestamp}.csv
```

### 5. Experiment Organization

```
results/
├── checkpoints/
│   ├── cnn_baseline/
│   ├── snn/
│   ├── vit/
│   ├── hybrid/
│   └── quantized/
├── logs/
│   ├── tensorboard/
│   └── csv/
├── figures/
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── comparison_plots/
└── reports/
    ├── model_comparison.csv
    ├── quantization_analysis.csv
    └── final_results.csv
```

---

## Implementation Rules

### Model Implementation

1. **Base classes**: All models inherit from a common `BaseModel` class
2. **Forward returns**: Always return logits (not softmax)
3. **Configurable**: All hyperparameters passed via config dict
4. **Modular**: Each component (backbone, attention, head) is separately testable

### Quantization Implementation

1. **Never quantize these layers**:
   - LIF neurons (non-differentiable spike function)
   - Layer normalization
   - Positional embeddings
   
2. **Quantization order**:
   - Train full-precision model → Save checkpoint
   - Apply QAT preparation → Fine-tune 10-20 epochs
   - Convert to quantized → Evaluate
   - Save both models for comparison

3. **Mixed precision support**:
   ```python
   # Critical layers: Keep at FP16/FP32
   # General layers: Quantize to INT8
   # Optional: INT4 for FC layers only
   ```

### Training Protocol

1. **Learning rate warmup**: First 5-10% of epochs
2. **Gradient clipping**: max_norm=5.0
3. **Early stopping**: patience=10-15 epochs
4. **Checkpoint saving**: 
   - Best validation accuracy
   - Best validation F1
   - Last epoch
   - Every 10 epochs

### Evaluation Protocol

1. **Statistical significance**:
   - Run each experiment with 3 different seeds
   - Report mean ± std
   - Perform paired t-tests for model comparisons
   
2. **Ablation studies**:
   - Each component should be removable
   - Track contribution of each module

---

## Code Snippets for Common Tasks

### Loading Configuration
```python
import yaml

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
```

### Model Checkpoint Saving
```python
def save_checkpoint(model, optimizer, epoch, metrics, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': model.config,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
```

### Metrics Logging
```python
def log_metrics(writer, metrics, epoch, prefix='train'):
    for name, value in metrics.items():
        writer.add_scalar(f'{prefix}/{name}', value, epoch)
```

---

## Error Handling

1. **Data loading errors**: Catch and log, skip corrupted samples
2. **CUDA OOM**: Implement gradient accumulation fallback
3. **NaN loss**: Detect and checkpoint recovery
4. **Quantization failures**: Fall back to float model gracefully

---

## Dependencies Version Pinning

Always verify compatible versions:
- PyTorch: 2.0+ (for torch.compile support)
- SpikingJelly: 0.0.0.0.12+ (for SNN)
- PennyLane: 0.41+ (for quantum)
- TorchQuantum: Latest stable
- timm: 0.6.12+ (for pretrained models)

---

## Journal-Specific Requirements

### Tables Format
- Use 4 decimal places for metrics (0.9234, not 0.92)
- Bold best results
- Include std for multi-run experiments
- Include model size and inference time

### Figure Requirements  
- DPI: 300 minimum
- Format: PNG for raster, PDF for vector
- Font size: Readable when printed
- Colors: Color-blind friendly palette
- Include legends and axis labels

### Ablation Studies Required
1. Impact of each architectural component
2. Effect of different attention mechanisms
3. Quantization bit-width comparison
4. Training hyperparameter sensitivity
