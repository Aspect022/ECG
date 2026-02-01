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


def create_dataloaders(config: dict) -> tuple:
    """Create train, val, test dataloaders."""
    data_cfg = config['data']
    
    train_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        mode='train',
        input_length=data_cfg['input_length'],
        augment=data_cfg.get('augment', True)
    )
    
    val_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        mode='val',
        input_length=data_cfg['input_length'],
        augment=False
    )
    
    test_dataset = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        mode='test',
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory']
    )
    
    return train_loader, val_loader, test_loader


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


def train(config_path: str, debug: bool = False):
    """Main training function."""
    # Load config
    config = load_config(config_path)
    
    if debug:
        config['training']['epochs'] = 2
        config['data']['batch_size'] = 4
        config['experiment']['num_workers'] = 0
    
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
    
    # Data
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Model
    print("Creating model...")
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
    
    # Power Meter
    power_meter = PowerMeter(device_idx=0 if device.type == 'cuda' else -1)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['training'].get('patience', 15)
    )
    
    # Training loop
    best_val_acc = 0.0
    train_cfg = config['training']
    
    print(f"\nStarting training for {train_cfg['epochs']} epochs...")
    print("=" * 80)
    
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
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch:03d} | "
              f"Train/Val Acc: {train_metrics['accuracy']:.4f}/{val_metrics['accuracy']:.4f} | "
              f"Loss: {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} | "
              f"Gate: {train_metrics['gate_mean']:.3f} | "
              f"Lat: {val_metrics['latency_ms']:.2f}ms | "
              f"E: {val_metrics['energy_mj']:.2f}mJ")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'config': config
            }, output_dir / 'best_model.pt')
            print(f"  ✓ New best model saved (acc: {best_val_acc:.4f})")
        
        # Save periodic checkpoint
        if epoch % config['experiment'].get('save_every', 5) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch:03d}.pt')
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print("=" * 80)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    
    # Final test
    print("\nRunning final test...")
    model.load_state_dict(
        torch.load(output_dir / 'best_model.pt')['model_state_dict']
    )
    test_metrics = validate(
        model, test_loader, config, device, epoch, writer, power_meter
    )
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Latency: {test_metrics['latency_ms']:.2f} ms/sample")
    print(f"Test Energy: {test_metrics['energy_mj']:.2f} mJ/sample")
    
    writer.close()
    
    return best_val_acc


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
