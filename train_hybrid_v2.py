"""
Hybrid Quantum-Classical ECG Trainer V2.0

PyTorch Lightning training script for the dual-path architecture.
Optimized for RTX 5050 (8GB VRAM) with:
- Mixed Precision (FP16)
- Gradient Accumulation
- TensorBoard Logging
- Early Stopping & Checkpointing
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

import yaml
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.ptbxl import PTBXLDataset
from src.models.v2 import HybridQuantumClassicalECG, HybridModelConfig


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_fold_dataloaders(config: dict, train_folds: list, val_folds: list) -> tuple:
    """Create train and val dataloaders for a specific fold split.
    
    Args:
        config: Configuration dictionary.
        train_folds: List of strat_fold values for training (e.g., [1, 2, 3, 4]).
        val_folds: List of strat_fold values for validation (e.g., [5]).
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_cfg = config['data']
    
    train_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=train_folds,
        input_length=data_cfg['input_length'],
        augment=data_cfg.get('augment', True)
    )
    
    val_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=val_folds,
        input_length=data_cfg['input_length'],
        augment=False
    )
    
    exp_cfg = config['experiment']
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory']
    )
    
    return train_loader, val_loader


def create_test_dataloader(config: dict) -> DataLoader:
    """Create test dataloader (fold 10, held out for all K-fold runs)."""
    data_cfg = config['data']
    exp_cfg = config['experiment']
    
    test_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=[10],  # Always fold 10 for test
        input_length=data_cfg['input_length'],
        augment=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory']
    )
    
    return test_loader


def create_model(config: dict) -> HybridQuantumClassicalECG:
    """Create hybrid model from config."""
    model_cfg = config['model']
    classical_cfg = model_cfg.get('classical', {})
    quantum_cfg = model_cfg.get('quantum', {})
    snn_cfg = model_cfg.get('snn', {})
    classifier_cfg = model_cfg.get('classifier', {})
    
    model_config = HybridModelConfig(
        in_channels=model_cfg.get('in_channels', 12),
        num_classes=config['data'].get('num_classes', 5),
        
        # Classical path
        stage1_channels=classical_cfg.get('stage1_channels', 256),
        stage2_channels=classical_cfg.get('stage2_channels', 128),
        stage3_channels=classical_cfg.get('stage3_channels', 64),
        fusion_dim=classical_cfg.get('fusion_dim', 128),
        weight_bits=classical_cfg.get('weight_bits', 1),
        
        # Quantum path
        n_qubits=quantum_cfg.get('n_qubits', 8),
        n_layers=quantum_cfg.get('n_layers', 3),
        quantum_feature_dim=quantum_cfg.get('feature_dim', 64),
        
        # SNN
        timesteps=snn_cfg.get('timesteps', 4),
        
        # Classifier
        hidden_dim=classifier_cfg.get('hidden_dim', 64),
        dropout=classifier_cfg.get('dropout', 0.3)
    )
    
    return HybridQuantumClassicalECG(model_config)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_ce_loss = 0.0
    total_reg_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_values = []
    
    accumulation_steps = config['data'].get('accumulation_steps', 1)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        signal = batch['signal'].to(device)
        label = batch['label'].to(device)
        
        # Mixed precision forward
        with autocast(enabled=config['experiment'].get('use_amp', True)):
            output = model(signal, return_gate=True)
            logits = output['logits']
            reg_loss = output['reg_loss']
            
            # Loss calculation
            if config['training'].get('criterion', 'cross_entropy') == 'bce_with_logits':
                ce_loss = F.binary_cross_entropy_with_logits(logits, label)
            else:
                # For single-label, convert to class indices
                if label.dim() > 1:
                    label_idx = label.argmax(dim=1)
                else:
                    label_idx = label
                ce_loss = F.cross_entropy(logits, label_idx)
            
            loss = ce_loss + reg_loss
            loss = loss / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            total_loss += loss.item() * accumulation_steps
            total_ce_loss += ce_loss.item()
            total_reg_loss += reg_loss.item()
            
            # Accuracy
            if label.dim() > 1:
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == label).all(dim=1).sum().item()
            else:
                preds = logits.argmax(dim=1)
                correct = (preds == label).sum().item()
            
            total_correct += correct
            total_samples += signal.shape[0]
            
            # Gate values
            gate_values.append(output['gate'].mean().item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'acc': f'{total_correct / total_samples:.4f}',
            'gate': f'{np.mean(gate_values[-10:]):.3f}'
        })
    
    # Log to TensorBoard
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    accuracy = total_correct / total_samples
    avg_gate = np.mean(gate_values)
    
    writer.add_scalar('Train/Loss', avg_loss, epoch)
    writer.add_scalar('Train/CE_Loss', avg_ce_loss, epoch)
    writer.add_scalar('Train/Reg_Loss', avg_reg_loss, epoch)
    writer.add_scalar('Train/Accuracy', accuracy, epoch)
    writer.add_scalar('Train/Gate_Mean', avg_gate, epoch)
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'reg_loss': avg_reg_loss,
        'accuracy': accuracy,
        'gate_mean': avg_gate
    }


class PowerMeter:
    """Monitor GPU power usage."""
    def __init__(self, device_idx: int = 0):
        self.device_idx = device_idx
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.available = True
        except ImportError:
            print("pynvml not found. Energy monitoring disabled.")
        except Exception as e:
            print(f"NVML init failed: {e}. Energy monitoring disabled.")

    def get_power(self) -> float:
        """Get current power usage in Watts."""
        if not self.available:
            return 0.0
        try:
            import pynvml
            # Returns milliwatts
            return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except:
            return 0.0


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    power_meter: PowerMeter = None
) -> dict:
    """Validate model with Latency and Energy tracking."""
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Latency tracking
    import time
    latencies = []
    
    # Energy tracking
    total_energy = 0.0
    
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        signal = batch['signal'].to(device)
        label = batch['label'].to(device)
        batch_size = signal.shape[0]
        
        # Measure power before
        p_start = power_meter.get_power() if power_meter else 0.0
        
        # Measure time
        t_start = time.time()
        
        with autocast(enabled=config['experiment'].get('use_amp', True)):
            output = model(signal)
            logits = output['logits']
            
            if config['training'].get('criterion', 'cross_entropy') == 'bce_with_logits':
                loss = F.binary_cross_entropy_with_logits(logits, label)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == label).all(dim=1).sum().item()
            else:
                if label.dim() > 1:
                    label_idx = label.argmax(dim=1)
                else:
                    label_idx = label
                loss = F.cross_entropy(logits, label_idx)
                preds = logits.argmax(dim=1)
                correct = (preds == label_idx).sum().item()
        
        # Synchronize for exact timing if CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        t_end = time.time()
        
        # Measure power after (approximate average)
        p_end = power_meter.get_power() if power_meter else 0.0
        avg_power = (p_start + p_end) / 2.0
        
        # Metrics calculation
        batch_latency = (t_end - t_start) * 1000.0  # ms
        batch_energy = avg_power * (t_end - t_start)  # Joules (Watts * seconds)
        
        latencies.append(batch_latency / batch_size)  # ms per sample
        total_energy += batch_energy
        
        total_loss += loss.item()
        total_correct += correct
        total_samples += signal.shape[0]
    
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    avg_latency = np.mean(latencies)
    avg_energy_per_sample = (total_energy / total_samples) * 1000.0 # mJ per sample
    
    writer.add_scalar('Val/Loss', avg_loss, epoch)
    writer.add_scalar('Val/Accuracy', accuracy, epoch)
    writer.add_scalar('Val/Latency_ms', avg_latency, epoch)
    writer.add_scalar('Val/Energy_mJ', avg_energy_per_sample, epoch)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'latency_ms': avg_latency,
        'energy_mj': avg_energy_per_sample
    }


def train(config_path: str, debug: bool = False, k_folds: int = 5):
    """Main training function with K-Fold Cross-Validation.
    
    Trains `k_folds` models, each on a different train/val split.
    Aggregates metrics across all folds for robust evaluation.
    """
    # Load config
    config = load_config(config_path)
    
    if debug:
        config['training']['epochs'] = 2
        config['data']['batch_size'] = 4
        config['experiment']['num_workers'] = 0
        k_folds = 2  # Faster debugging
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(config['experiment']['output_dir']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # TensorBoard
    log_dir = Path(config['logging']['log_dir']) / timestamp
    writer = SummaryWriter(log_dir)
    
    # Power Meter
    power_meter = PowerMeter(device_idx=0 if device.type == 'cuda' else -1)
    
    # K-Fold Setup: Use folds 1-9 for CV, fold 10 is always test
    available_folds = list(range(1, 10))  # Folds 1-9
    fold_size = len(available_folds) // k_folds
    
    fold_results = []
    
    print(f"\n{'=' * 80}")
    print(f"Starting {k_folds}-Fold Cross-Validation")
    print(f"{'=' * 80}\n")
    
    for fold_idx in range(k_folds):
        print(f"\n{'=' * 40}")
        print(f"FOLD {fold_idx + 1}/{k_folds}")
        print(f"{'=' * 40}")
        
        # Determine train/val folds for this iteration
        # Val fold: rotate through available folds
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size
        val_folds = available_folds[val_start:val_end]
        train_folds = [f for f in available_folds if f not in val_folds]
        
        print(f"Train folds: {train_folds}, Val fold(s): {val_folds}")
        
        # Create dataloaders for this fold
        train_loader, val_loader = create_fold_dataloaders(config, train_folds, val_folds)
        print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
        
        # Create FRESH model for each fold
        model = create_model(config)
        model = model.to(device)
        print(f"Model parameters: {model.num_parameters:,}")
        
        # Optimizer
        opt_cfg = config['optimizer']
        optimizer = AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999))
        )
        
        # Scheduler
        sched_cfg = config['scheduler']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get('T_0', 10),
            T_mult=sched_cfg.get('T_mult', 2),
            eta_min=sched_cfg.get('eta_min', 1e-6)
        )
        
        # Mixed precision
        scaler = GradScaler(enabled=config['experiment'].get('use_amp', True))
        
        # Early stopping (reset per fold)
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 15)
        )
        
        # Training loop for this fold
        best_val_acc = 0.0
        train_cfg = config['training']
        
        for epoch in range(1, train_cfg['epochs'] + 1):
            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, scaler,
                config, device, epoch, writer
            )
            
            # Validate
            val_metrics = validate(
                model, val_loader, config, device, epoch, writer, power_meter
            )
            
            # Scheduler step
            scheduler.step()
            
            # Print epoch summary
            print(f"  Epoch {epoch:03d} | "
                  f"Train/Val Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
                  f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
                  f"Gate: {train_metrics['gate_mean']:.3f}")
            
            # Save best model for this fold
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                torch.save({
                    'fold': fold_idx + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': best_val_acc,
                    'config': config
                }, output_dir / f'best_model_fold_{fold_idx + 1}.pt')
            
            # Early stopping
            if early_stopping(val_metrics['accuracy']):
                print(f"  Early stopping triggered at epoch {epoch}")
                break
        
        # Store fold results
        fold_results.append({
            'fold': fold_idx + 1,
            'best_val_acc': best_val_acc,
            'final_val_metrics': val_metrics
        })
        print(f"Fold {fold_idx + 1} Complete: Best Val Acc = {best_val_acc:.4f}")
    
    # ===== Aggregate Results =====
    print(f"\n{'=' * 80}")
    print("K-Fold Cross-Validation Results")
    print(f"{'=' * 80}")
    
    accuracies = [r['best_val_acc'] for r in fold_results]
    print(f"Fold Accuracies: {[f'{a:.4f}' for a in accuracies]}")
    print(f"Mean Accuracy:   {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    # Save summary
    with open(output_dir / 'kfold_summary.yaml', 'w') as f:
        yaml.dump({
            'k_folds': k_folds,
            'fold_results': fold_results,
            'mean_accuracy': float(np.mean(accuracies)),
            'std_accuracy': float(np.std(accuracies))
        }, f)
    
    # ===== Final Test on Fold 10 =====
    print("\nRunning final test on held-out Fold 10...")
    
    # Use best fold model for test
    best_fold = np.argmax(accuracies) + 1
    print(f"Using best fold model: Fold {best_fold}")
    
    test_loader = create_test_dataloader(config)
    model.load_state_dict(
        torch.load(output_dir / f'best_model_fold_{best_fold}.pt')['model_state_dict']
    )
    
    test_metrics = validate(
        model, test_loader, config, device, 0, writer, power_meter
    )
    
    print(f"\nTest Results (Fold 10):")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Latency:  {test_metrics['latency_ms']:.2f} ms/sample")
    print(f"  Energy:   {test_metrics['energy_mj']:.2f} mJ/sample")
    
    writer.close()
    
    return np.mean(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hybrid Quantum-Classical ECG Model')
    parser.add_argument(
        '--config', type=str,
        default='configs/hybrid_v2.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Run in debug mode (2 epochs, small batch)'
    )
    
    args = parser.parse_args()
    
    train(args.config, debug=args.debug)
