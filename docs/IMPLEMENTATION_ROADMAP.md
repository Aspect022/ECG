# IMPLEMENTATION ROADMAP V2.0 - DUAL-PATH QUANTUM-ENHANCED ECG CLASSIFIER

## OVERVIEW

This document provides a practical, step-by-step implementation guide for the **Dual-Path Quantum-Enhanced ECG Classifier (V2.0 Architecture)**.

> [!NOTE]
> This roadmap aligns with the **REVISED HYBRID QUANTUM-CLASSICAL ARCHITECTURE V2.0** specifications. Key innovations include: Hierarchical Temporal Feature Pyramid, Learned Entanglement VQC, and Gated Fusion.

**Timeline**: 10 weeks  
**Target**: Publication-ready paper + open-source code  
**Hardware**: RTX 5050 (8GB VRAM)
**Architecture Reference**: See [SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)

---

## WEEK 1-2: PHASE 0 - BASELINES & INFRASTRUCTURE

### Goals
- ✅ Setup development environment
- ✅ Implement EEGNet baseline (92.5% target)
- ✅ Establish benchmarks for comparison
- ✅ Validate RTX 5050 compatibility

### Tasks

#### Day 1-2: Environment Setup

```bash
# Create conda environment
conda create -n quantum-ecg python=3.10
conda activate quantum-ecg

# Install core dependencies
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.1.0
pip install wandb==0.16.0
pip install wfdb==4.1.0  # For PTB-XL dataset
pip install scikit-learn pandas numpy matplotlib seaborn

# Create project structure
mkdir -p quantum_ecg/{models,data,utils,configs,experiments}
touch quantum_ecg/__init__.py
```

#### Day 3-5: Data Pipeline

```python
# quantum_ecg/data/ptbxl_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import wfdb
import pandas as pd
import numpy as np

class PTBXLDataset(Dataset):
    """PTB-XL ECG Dataset Loader"""
    
    def __init__(self, data_path='./data/ptbxl', 
                 sampling_rate=500, 
                 split='train',
                 transform=None):
        self.data_path = data_path
        self.sampling_rate = sampling_rate
        self.transform = transform
        
        # Load metadata
        self.metadata = pd.read_csv(f'{data_path}/ptbxl_database.csv', index_col='ecg_id')
        
        # Load diagnostic statements
        self.statements = pd.read_csv(f'{data_path}/scp_statements.csv', index_col=0)
        
        # Filter by split
        if split == 'train':
            self.metadata = self.metadata[self.metadata.strat_fold < 9]
        elif split == 'val':
            self.metadata = self.metadata[self.metadata.strat_fold == 9]
        elif split == 'test':
            self.metadata = self.metadata[self.metadata.strat_fold == 10]
        
        # Aggregate diagnostic superclasses
        self.metadata['diagnostic_superclass'] = self.metadata.scp_codes.apply(
            lambda x: self._aggregate_diagnostic(eval(x))
        )
        
        # Filter out samples with no diagnosis
        self.metadata = self.metadata[self.metadata.diagnostic_superclass != 'NORM']
        
    def _aggregate_diagnostic(self, scp_codes):
        """Convert SCP codes to superclass"""
        # Implementation details...
        # Returns one of: NORM, MI, STTC, CD, HYP
        pass
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load ECG signal
        record = wfdb.rdsamp(f'{self.data_path}/{row.filename_hr}')
        signal = record[0]  # (5000, 12)
        
        # Transpose to (12, 5000)
        signal = torch.FloatTensor(signal.T)
        
        # Normalize per lead
        signal = (signal - signal.mean(dim=1, keepdim=True)) / (signal.std(dim=1, keepdim=True) + 1e-8)
        
        # Label
        label = self._diagnostic_to_idx(row.diagnostic_superclass)
        
        if self.transform:
            signal = self.transform(signal)
        
        return signal, label
    
    def _diagnostic_to_idx(self, diagnostic):
        """Convert diagnostic superclass to index"""
        classes = ['MI', 'STTC', 'CD', 'HYP']
        return classes.index(diagnostic)

# Create DataLoaders
def create_dataloaders(batch_size=128, num_workers=4):
    train_dataset = PTBXLDataset(split='train')
    val_dataset = PTBXLDataset(split='val')
    test_dataset = PTBXLDataset(split='test')
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

#### Day 6-10: EEGNet Baseline

```python
# quantum_ecg/models/eegnet_baseline.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """EEGNet baseline for ECG classification"""
    
    def __init__(self, num_classes=4, channels=12, samples=5000, 
                 F1=8, D=2, F2=16, dropout=0.25):
        super().__init__()
        
        # Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Depthwise convolution (spatial)
        self.depthwise = nn.Conv2d(F1, F1*D, (channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1*D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Separable convolution
        self.separable1 = nn.Conv2d(F1*D, F1*D, (1, 16), padding='same', groups=F1*D, bias=False)
        self.separable2 = nn.Conv2d(F1*D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Classifier
        self.flatten = nn.Flatten()
        # Calculate flattened size
        self.fc_size = F2 * (samples // 32)
        self.fc = nn.Linear(self.fc_size, num_classes)
        
    def forward(self, x):
        # x: (batch, 12, 5000)
        x = x.unsqueeze(1)  # (batch, 1, 12, 5000)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Block 2
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.separable1(x)
        x = self.separable2(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.fc(x)
        
        return x
```

#### Day 11-14: Training Script with Lightning

```python
# quantum_ecg/train_baseline.py

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, Precision, Recall

class EEGNetLightning(pl.LightningModule):
    def __init__(self, num_classes=4, lr=0.001):
        super().__init__()
        self.model = EEGNet(num_classes=num_classes)
        self.lr = lr
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        acc = self.train_acc(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        # Log metrics
        acc = self.val_acc(logits, y)
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        
        acc = self.test_acc(logits, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

# Main training loop
if __name__ == '__main__':
    # Initialize model
    model = EEGNetLightning(num_classes=4, lr=0.001)
    
    # Data
    train_loader, val_loader, test_loader = create_dataloaders(batch_size=128)
    
    # Logger
    wandb_logger = WandbLogger(project='quantum-ecg', name='eegnet-baseline')
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=3,
        filename='eegnet-{epoch:02d}-{val_acc:.4f}'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        patience=15,
        mode='max'
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',  # Mixed precision for RTX 5050
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        gradient_clip_val=1.0
    )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Test
    trainer.test(model, test_loader, ckpt_path='best')
```

### Expected Outcomes (Week 1-2)

- ✅ Working data pipeline (PTB-XL loaded)
- ✅ EEGNet baseline trained
- ✅ **Target**: 92-93% accuracy on PTB-XL
- ✅ Training time: ~8 min/epoch on RTX 5050
- ✅ Memory usage: ~180 MB/batch

---

## WEEK 3-4: PHASE 1 - CLASSICAL PATH IMPLEMENTATION

### Goals
- Implement Quantized SNN backbone
- Implement Multi-Scale Spiking Attention
- Build 3-stage hierarchical pyramid
- **Target**: 95.0% accuracy

### Tasks

#### Day 15-18: Quantized LIF Neuron

```python
# quantum_ecg/models/quantized_snn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class QuantLIFNeuron(nn.Module):
    """Quantized Leaky Integrate-and-Fire Neuron"""
    
    def __init__(self, threshold=1.0, tau=2.0, potential_bits=8, 
                 spike_regularization=0.01):
        super().__init__()
        self.threshold = threshold
        self.tau = tau
        self.potential_bits = potential_bits
        self.spike_reg = spike_regularization
        
        # Learnable decay parameter
        self.alpha = nn.Parameter(torch.tensor(np.exp(-1/tau)))
    
    def quantize(self, x, bits):
        """Quantize tensor to specified bit-width"""
        scale = 2 ** bits
        x_quantized = torch.round(x * scale) / scale
        return x_quantized
    
    def forward(self, x, timesteps=4):
        """
        x: (batch, channels, length)
        Returns: spikes, potentials
        """
        batch, channels, length = x.shape
        device = x.device
        
        # Initialize membrane potential
        v = torch.zeros(batch, channels, length, device=device)
        
        # Accumulate spikes over timesteps
        spike_sum = torch.zeros_like(x)
        
        for t in range(timesteps):
            # Leak
            v = self.alpha * v
            
            # Integrate
            v = v + x
            
            # Quantize membrane potential
            v = self.quantize(v, self.potential_bits)
            
            # Fire
            spike = (v >= self.threshold).float()
            
            # Reset
            v = v * (1 - spike)
            
            # Accumulate
            spike_sum += spike
        
        # Average spikes over timesteps
        spike_avg = spike_sum / timesteps
        
        # Spike regularization loss
        reg_loss = self.spike_reg * torch.mean((spike_avg - 0.5) ** 2)
        
        return spike_avg, v, reg_loss


class QuantizedConv1d(nn.Module):
    """1D Convolution with Binary Weight Quantization"""
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, groups=1, weight_bits=1):
        super().__init__()
        self.weight_bits = weight_bits
        
        # Standard convolution
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              stride, padding, groups=groups, bias=False)
        
        # Layer normalization for weights
        self.weight_ln = nn.LayerNorm(self.conv.weight.shape)
    
    def binarize_weights(self, w):
        """Binarize weights using sign function"""
        w_normalized = self.weight_ln(w)
        w_binary = torch.sign(w_normalized)
        w_binary = torch.where(w_binary == 0, torch.ones_like(w_binary), w_binary)
        return w_binary
    
    def forward(self, x):
        # Binarize weights
        if self.weight_bits == 1:
            w = self.binarize_weights(self.conv.weight)
        else:
            w = self.conv.weight
        
        # Apply convolution with binarized weights
        return F.conv1d(x, w, bias=None, 
                       stride=self.conv.stride, 
                       padding=self.conv.padding,
                       groups=self.conv.groups)
```

#### Day 19-22: Multi-Scale Spiking Attention

```python
# quantum_ecg/models/mssa.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleSpikingAttention(nn.Module):
    """Multi-Scale Spiking Attention (MSSA) adapted for 1D ECG"""
    
    def __init__(self, dim, num_heads=3, local_window=64, regional_window=256):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.local_window = local_window
        self.regional_window = regional_window
        
        # Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Multi-scale feature extraction
        self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.regional_conv = nn.Conv1d(dim, dim, kernel_size=9, padding=4, 
                                       dilation=2, groups=dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Spiking mechanism
        self.spike_gate = QuantLIFNeuron(threshold=0.5, tau=2.0)
    
    def forward(self, x):
        """
        x: (batch, channels, length)
        """
        batch, channels, length = x.shape
        
        # Transpose for linear layer
        x_t = x.transpose(1, 2)  # (batch, length, channels)
        
        # Q, K, V
        qkv = self.qkv(x_t).reshape(batch, length, 3, channels)
        q, k, v = qkv.unbind(2)  # Each: (batch, length, channels)
        
        # Local attention (fine details)
        local_attn = self._local_attention(q, k, v, self.local_window)
        
        # Regional attention (medium-scale patterns)
        regional_attn = self._regional_attention(q, k, v, self.regional_window)
        
        # Global attention (rhythm trends)
        global_attn = self._global_attention(q, k, v)
        
        # Combine multi-scale features
        multi_scale = (local_attn + regional_attn + global_attn) / 3
        
        # Apply spiking gate
        multi_scale_t = multi_scale.transpose(1, 2)  # (batch, channels, length)
        gated, _, _ = self.spike_gate(multi_scale_t)
        
        # Output projection
        output = self.proj(gated.transpose(1, 2))  # (batch, length, channels)
        output = output.transpose(1, 2)  # (batch, channels, length)
        
        return output
    
    def _local_attention(self, q, k, v, window_size):
        """Local windowed attention"""
        batch, length, channels = q.shape
        
        # Pad sequence
        pad = window_size // 2
        q_pad = F.pad(q, (0, 0, pad, pad))
        k_pad = F.pad(k, (0, 0, pad, pad))
        v_pad = F.pad(v, (0, 0, pad, pad))
        
        # Unfold into windows
        q_windows = q_pad.unfold(1, window_size, 1)  # (batch, length, channels, window)
        k_windows = k_pad.unfold(1, window_size, 1)
        v_windows = v_pad.unfold(1, window_size, 1)
        
        # Attention within windows
        attn_scores = torch.matmul(q.unsqueeze(-2), k_windows.transpose(-2, -1))
        attn_scores = attn_scores / (channels ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v_windows).squeeze(-2)
        
        return output
    
    def _regional_attention(self, q, k, v, window_size):
        """Regional dilated attention"""
        # Similar to local but with dilated windows
        # Implementation details...
        return q  # Placeholder
    
    def _global_attention(self, q, k, v):
        """Lightweight global attention"""
        batch, length, channels = q.shape
        
        # Global pooling
        k_global = k.mean(dim=1, keepdim=True)  # (batch, 1, channels)
        v_global = v.mean(dim=1, keepdim=True)
        
        # Attention
        attn_scores = torch.matmul(q, k_global.transpose(-2, -1))
        attn_scores = attn_scores / (channels ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        output = torch.matmul(attn_weights, v_global).expand(-1, length, -1)
        
        return output
```

#### Day 23-28: Hierarchical Temporal Pyramid

```python
# quantum_ecg/models/classical_path.py

import torch
import torch.nn as nn

class LocalPatternExtractor(nn.Module):
    """Stage 1: Local high-resolution patterns"""
    
    def __init__(self, in_channels=12, out_channels=256):
        super().__init__()
        self.conv = QuantizedConv1d(in_channels, out_channels, 
                                     kernel_size=3, stride=1, padding=1,
                                     groups=in_channels, weight_bits=1)
        self.lif = QuantLIFNeuron(threshold=1.0, tau=2.0, potential_bits=8)
    
    def forward(self, x):
        x = self.conv(x)
        x, _, reg_loss = self.lif(x, timesteps=4)
        return x, reg_loss


class BeatPatternExtractor(nn.Module):
    """Stage 2: Beat-level mid-resolution patterns"""
    
    def __init__(self, in_channels=256, out_channels=128):
        super().__init__()
        self.conv = QuantizedConv1d(in_channels, out_channels, 
                                     kernel_size=9, stride=2, padding=4,
                                     weight_bits=1)
        self.mssa = MultiScaleSpikingAttention(dim=out_channels, num_heads=3)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.mssa(x)
        return x


class RhythmPatternExtractor(nn.Module):
    """Stage 3: Rhythm low-resolution patterns"""
    
    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()
        self.conv = QuantizedConv1d(in_channels, out_channels, 
                                     kernel_size=27, stride=4, padding=13,
                                     weight_bits=1)
        self.global_attn = nn.MultiheadAttention(out_channels, num_heads=1, 
                                                  batch_first=True)
    
    def forward(self, x):
        x = self.conv(x)
        # Apply global attention
        x_t = x.transpose(1, 2)
        x_t, _ = self.global_attn(x_t, x_t, x_t)
        x = x_t.transpose(1, 2)
        return x


class TemporalFusionBlock(nn.Module):
    """Fuse multi-scale features"""
    
    def __init__(self):
        super().__init__()
        # Adaptive pooling to common length
        self.pool = nn.AdaptiveAvgPool1d(1000)
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(3))
        
        # Projection
        self.proj = QuantizedConv1d(256+128+64, 128, kernel_size=1, weight_bits=2)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, stage1, stage2, stage3):
        # Pool to common length
        s1 = self.pool(stage1)
        s2 = self.pool(stage2)
        s3 = self.pool(stage3)
        
        # Concatenate
        fused = torch.cat([s1, s2, s3], dim=1)
        
        # Project
        fused = self.proj(fused)
        
        # Global pool
        fused = self.global_pool(fused).squeeze(-1)
        
        return fused


class ClassicalPath(nn.Module):
    """Complete classical hierarchical temporal pyramid"""
    
    def __init__(self):
        super().__init__()
        self.stage1 = LocalPatternExtractor()
        self.stage2 = BeatPatternExtractor()
        self.stage3 = RhythmPatternExtractor()
        self.fusion = TemporalFusionBlock()
    
    def forward(self, x):
        # x: (batch, 12, 5000)
        s1, reg_loss1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        
        classical_features = self.fusion(s1, s2, s3)
        
        return classical_features, reg_loss1
```

### Expected Outcomes (Week 3-4)

- ✅ Classical path implemented
- ✅ **Target**: 94.5-95.0% accuracy (without quantum)
- ✅ Model size: ~25 MB (binary quantization)
- ✅ Training time: ~9 min/epoch

---

## WEEK 5-6: PHASE 2 - QUANTUM PATH IMPLEMENTATION

### Goals
- Implement vectorized quantum circuit in PyTorch
- Build feature compressor
- Validate 50× speedup vs PennyLane
- **Target**: Quantum features ready for fusion

### Tasks

#### Day 29-32: Vectorized Quantum Operations

```python
# quantum_ecg/models/quantum_circuit.py

import torch
import torch.nn as nn
import numpy as np

class VectorizedQuantumCircuit(nn.Module):
    """Vectorized Quantum Circuit (no PennyLane)"""
    
    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state_dim = 2 ** n_qubits
        
        # Trainable parameters
        self.theta = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.omega = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        
        # Learned entanglement (attention-based)
        self.entangle_attn = nn.Linear(n_qubits, n_qubits)
        
        # Precompute gate matrices
        self._precompute_gates()
    
    def _precompute_gates(self):
        """Precompute Pauli matrices and basis gates"""
        # Pauli-Y
        self.pauli_y = torch.tensor([
            [0, -1j],
            [1j, 0]
        ], dtype=torch.complex64)
        
        # Pauli-Z
        self.pauli_z = torch.tensor([
            [1, 0],
            [0, -1]
        ], dtype=torch.complex64)
        
        # Identity
        self.identity = torch.eye(2, dtype=torch.complex64)
    
    def ry_gate(self, angle):
        """Create RY rotation gate"""
        cos_half = torch.cos(angle / 2)
        sin_half = torch.sin(angle / 2)
        
        gate = torch.stack([
            torch.stack([cos_half, -sin_half]),
            torch.stack([sin_half, cos_half])
        ])
        
        return gate
    
    def rz_gate(self, angle):
        """Create RZ rotation gate"""
        phase = torch.exp(1j * angle / 2)
        
        gate = torch.stack([
            torch.stack([phase.conj(), torch.zeros_like(phase)]),
            torch.stack([torch.zeros_like(phase), phase])
        ])
        
        return gate
    
    def apply_single_qubit_gate(self, state, qubit_idx, gate):
        """Apply single-qubit gate using Kronecker products"""
        batch_size = state.shape[0]
        
        # Build full gate: I ⊗ ... ⊗ G ⊗ ... ⊗ I
        full_gate = self.identity
        for i in range(self.n_qubits):
            if i == 0:
                full_gate = gate if i == qubit_idx else self.identity
            else:
                current_gate = gate if i == qubit_idx else self.identity
                full_gate = torch.kron(full_gate, current_gate)
        
        # Apply to state
        # state: (batch, 2^n)
        # full_gate: (2^n, 2^n)
        new_state = torch.matmul(state, full_gate.t())
        
        return new_state
    
    def cnot_gate(self, state, control, target):
        """Apply CNOT gate"""
        batch_size = state.shape[0]
        
        # Build CNOT matrix
        # CNOT = |0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X
        # Efficient implementation using index manipulation
        
        # Reshape state to expose qubit structure
        state_reshaped = state.view(batch_size, *([2]*self.n_qubits))
        
        # Swap target qubit based on control qubit
        # (Implementation uses advanced indexing)
        
        # Placeholder - full implementation in actual code
        return state
    
    def forward(self, x):
        """
        x: (batch, n_qubits) - classical features normalized to [0, π]
        Returns: (batch, state_dim) - quantum state
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize |0⟩^⊗n
        state = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j
        
        # Amplitude encoding with RY gates
        for i in range(self.n_qubits):
            gate = self.ry_gate(x[:, i])
            state = self.apply_single_qubit_gate(state, i, gate)
        
        # Variational layers
        for layer in range(self.n_layers):
            # RY rotations
            for i in range(self.n_qubits):
                gate = self.ry_gate(self.theta[layer, i])
                state = self.apply_single_qubit_gate(state, i, gate)
            
            # Learned entanglement
            entangle_weights = torch.softmax(self.entangle_attn.weight, dim=1)
            for i in range(self.n_qubits):
                target = torch.argmax(entangle_weights[i]).item()
                if target != i:
                    state = self.cnot_gate(state, control=i, target=target)
            
            # RZ rotations
            for i in range(self.n_qubits):
                gate = self.rz_gate(self.omega[layer, i])
                state = self.apply_single_qubit_gate(state, i, gate)
        
        return state
    
    def measure(self, state):
        """Measure Pauli-Z expectation for each qubit"""
        batch_size = state.shape[0]
        expectations = []
        
        for i in range(self.n_qubits):
            # Create Pauli-Z observable for qubit i
            observable = self._create_pauli_z_observable(i)
            
            # Expectation: ⟨ψ|Z_i|ψ⟩
            expectation = torch.real(
                torch.sum(state.conj() * torch.matmul(state, observable.t()), dim=1)
            )
            expectations.append(expectation)
        
        quantum_features = torch.stack(expectations, dim=1)
        return quantum_features
    
    def _create_pauli_z_observable(self, qubit_idx):
        """Create Pauli-Z observable for specific qubit"""
        observable = self.identity
        for i in range(self.n_qubits):
            if i == 0:
                observable = self.pauli_z if i == qubit_idx else self.identity
            else:
                current = self.pauli_z if i == qubit_idx else self.identity
                observable = torch.kron(observable, current)
        return observable
```

#### Day 33-36: Complete Quantum Path

```python
# quantum_ecg/models/quantum_path.py

import torch
import torch.nn as nn

class FeatureCompressor(nn.Module):
    """Compress 12-lead ECG to quantum-encodable features"""
    
    def __init__(self, n_features=64):
        super().__init__()
        self.conv1 = QuantizedConv1d(12, 64, kernel_size=5, stride=5)
        self.lif = QuantLIFNeuron(threshold=1.0, tau=2.0)
        self.pool = nn.MaxPool1d(kernel_size=16, stride=16)
        self.fc = nn.Linear(64 * 62, n_features)
    
    def forward(self, x):
        # x: (batch, 12, 5000)
        x = self.conv1(x)  # (batch, 64, 1000)
        x, _, _ = self.lif(x)  # Spiking
        x = self.pool(x)  # (batch, 64, 62)
        
        x = x.flatten(1)  # (batch, 64*62)
        x = self.fc(x)  # (batch, 64)
        
        return x


class QuantumEncodingLayer(nn.Module):
    """Encode classical features into quantum state"""
    
    def __init__(self, n_features=64, n_qubits=8):
        super().__init__()
        self.projection = nn.Linear(n_features, n_qubits)
    
    def forward(self, x):
        # Project and normalize to [0, π]
        x = self.projection(x)
        x = torch.sigmoid(x) * np.pi
        return x


class QuantumPath(nn.Module):
    """Complete quantum processing path"""
    
    def __init__(self, n_qubits=8, n_layers=3):
        super().__init__()
        self.compressor = FeatureCompressor(n_features=64)
        self.encoder = QuantumEncodingLayer(n_features=64, n_qubits=n_qubits)
        self.circuit = VectorizedQuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
    
    def forward(self, x):
        # x: (batch, 12, 5000)
        compressed = self.compressor(x)  # (batch, 64)
        encoded = self.encoder(compressed)  # (batch, 8) in [0, π]
        
        quantum_state = self.circuit(encoded)  # (batch, 256)
        quantum_features = self.circuit.measure(quantum_state)  # (batch, 8)
        
        return quantum_features
```

### Expected Outcomes (Week 5-6)

- ✅ Vectorized quantum circuit implemented
- ✅ 50-100× speedup verified vs PennyLane
- ✅ Quantum features (8-dim) generated
- ✅ Gradients flow correctly through quantum circuit

---

## WEEK 7-8: PHASE 3 - INTEGRATION & OPTIMIZATION

### Goals
- Implement gated fusion
- Integrate full hybrid model
- Hyperparameter tuning
- **Target**: 95.5-95.8% accuracy

### Tasks

#### Day 37-40: Gated Fusion

```python
# quantum_ecg/models/fusion.py

import torch
import torch.nn as nn

class GatedFusionModule(nn.Module):
    """Gated fusion of classical and quantum features"""
    
    def __init__(self, classical_dim=128, quantum_dim=8):
        super().__init__()
        
        # Expand quantum features
        self.quantum_expand = nn.Sequential(
            nn.Linear(quantum_dim, 64),
            nn.ELU(),
            nn.Linear(64, classical_dim)
        )
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(classical_dim + quantum_dim, 64),
            nn.ELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, classical_features, quantum_features):
        # classical_features: (batch, 128)
        # quantum_features: (batch, 8)
        
        # Compute gate
        combined = torch.cat([classical_features, quantum_features], dim=1)
        gate_value = self.gate(combined)  # (batch, 1)
        
        # Expand quantum
        quantum_expanded = self.quantum_expand(quantum_features)
        
        # Gated fusion
        fused = (1 - gate_value) * classical_features + gate_value * quantum_expanded
        
        return fused, gate_value
```

#### Day 41-44: Full Hybrid Model

```python
# quantum_ecg/models/hybrid_model.py

import torch
import torch.nn as nn
import pytorch_lightning as pl

class HybridQuantumClassicalECG(pl.LightningModule):
    """Complete dual-path hybrid model"""
    
    def __init__(self, num_classes=4, n_qubits=8, lr=0.001):
        super().__init__()
        
        # Classical path
        self.classical_path = ClassicalPath()
        
        # Quantum path
        self.quantum_path = QuantumPath(n_qubits=n_qubits, n_layers=3)
        
        # Fusion
        self.fusion = GatedFusionModule(classical_dim=128, quantum_dim=n_qubits)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
        self.lr = lr
        self.save_hyperparameters()
    
    def forward(self, x):
        # Classical path
        classical_features, reg_loss = self.classical_path(x)
        
        # Quantum path
        quantum_features = self.quantum_path(x)
        
        # Fusion
        fused_features, gate_value = self.fusion(classical_features, quantum_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, reg_loss, gate_value
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, reg_loss, gate_value = self(x)
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, y)
        
        # Total loss
        loss = ce_loss + reg_loss
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_ce_loss', ce_loss)
        self.log('train_reg_loss', reg_loss)
        self.log('gate_mean', gate_value.mean())
        
        return loss
    
    # ... (validation, test, configure_optimizers similar to baseline)
```

### Expected Outcomes (Week 7-8)

- ✅ Full hybrid model working
- ✅ **Target**: 95.5-95.8% accuracy
- ✅ Training time: ~10-11 min/epoch
- ✅ Gate value converges (shows quantum contribution)

---

## WEEK 9-10: PHASE 4 - EXPERIMENTS & PAPER

### Goals
- Run ablation studies
- Cross-dataset validation
- Generate all figures/tables
- Write paper draft
- **Target**: Publication-ready manuscript

### Tasks

#### Day 45-49: Ablation Studies

```python
# experiments/ablations.py

# 1. Architecture Ablation
ablation_configs = [
    {'name': 'EEGNet', 'model': EEGNet()},
    {'name': 'Q-SNN', 'model': QuantizedSNN()},
    {'name': 'Q-SNN+Pyramid', 'model': ClassicalPath()},
    {'name': 'Q-SNN+Pyramid+Quantum', 'model': HybridQuantumClassicalECG()},
]

# 2. Quantum Configuration
qubit_counts = [4, 8, 12, 16]
layer_counts = [1, 2, 3, 4]

# 3. Fusion Strategy
fusion_types = ['concat', 'weighted_sum', 'gated']  # Gated best

# Run all ablations and log to wandb
```

#### Day 50-53: Cross-Dataset Validation

```python
# experiments/cross_dataset.py

# Datasets
datasets = [
    'PTB-XL',      # Training dataset
    'MIT-BIH',     # Test generalization
    'PhysioNet'    # Test generalization
]

# Train on PTB-XL, test on others
# Report accuracy, F1, precision, recall
```

#### Day 54-56: Figure Generation

```python
# experiments/generate_figures.py

# Figure 1: Architecture diagram
# Figure 2: Training curves
# Figure 3: Ablation results (bar charts)
# Figure 4: Confusion matrices
# Figure 5: Gate value evolution
# Figure 6: Quantum feature visualization (t-SNE)
# Figure 7: Cross-dataset results
```

#### Day 57-70: Paper Writing

```markdown
# Paper Outline

1. **Title**: Dual-Path Quantum-Enhanced Spiking Neural Network...
2. **Abstract**: 250 words
3. **Introduction** (1.5 pages):
   - ECG importance
   - Deep learning for ECG
   - Limitations of current methods
   - Quantum ML potential
   - Our contribution
4. **Related Work** (2 pages):
   - SNNs for ECG
   - Quantum ML
   - Hybrid quantum-classical models
5. **Methods** (4 pages):
   - Classical path design
   - Quantum path design
   - Gated fusion
   - Training procedure
6. **Experiments** (3 pages):
   - Datasets
   - Implementation details
   - Baseline comparisons
   - Ablation studies
7. **Results** (2 pages):
   - Main results table
   - Ablation results
   - Cross-dataset validation
   - Qualitative analysis
8. **Discussion** (1 page):
   - Why quantum helps
   - Learned entanglement analysis
   - Limitations
9. **Conclusion** (0.5 pages)
10. **References** (2 pages)

**Total**: ~16-18 pages
```

### Expected Outcomes (Week 9-10)

- ✅ All experiments completed
- ✅ 10+ figures/tables generated
- ✅ Paper draft ready for submission
- ✅ Code open-sourced on GitHub

---

## CRITICAL SUCCESS METRICS

### Performance Targets

| Metric | Target | Achieved? |
|--------|--------|-----------|
| Accuracy | ≥95.5% | TBD |
| Model Size | ≤30 MB | TBD |
| Training Time | ≤12 min/epoch | TBD |
| Memory Usage | ≤200 MB/batch | TBD |
| Inference Speed | ≥8 ECG/s | TBD |

### Deliverables Checklist

- [ ] Working hybrid model (all components)
- [ ] Trained on PTB-XL (95.5%+ accuracy)
- [ ] Cross-dataset validated (MIT-BIH, PhysioNet)
- [ ] Ablation studies (6+ experiments)
- [ ] Paper draft (16-18 pages)
- [ ] Code repository (GitHub, documented)
- [ ] Pretrained weights (HuggingFace)

---

## TROUBLESHOOTING GUIDE

### Common Issues

1. **RTX 5050 OOM**:
   - Reduce batch size (128 → 64)
   - Increase gradient accumulation (4 → 8)
   - Use more aggressive quantization (int8 → int4)

2. **Quantum circuit too slow**:
   - Verify vectorization (no loops over qubits)
   - Use torch.jit.script for gates
   - Reduce state dimension (8 qubits max on GPU)

3. **Training instability**:
   - Reduce learning rate (0.001 → 0.0001)
   - Add warmup (10 epochs)
   - Clip gradients (1.0)
   - Check gate value (should be 0.1-0.9, not 0 or 1)

4. **Quantum not helping**:
   - Increase qubits (8 → 12)
   - Increase layers (3 → 4)
   - Check learned entanglement (should not be identity)
   - Verify features are normalized correctly

---

## CONCLUSION

This roadmap provides a complete, week-by-week guide to implementing the dual-path quantum-enhanced ECG classifier. The phased approach ensures:

1. **Safety**: Classical baseline first
2. **Modularity**: Components can be developed independently
3. **Validation**: Each phase has clear success criteria
4. **Flexibility**: Can adjust if targets not met

**Start with Week 1-2 baselines this week. Good luck! 🚀**
