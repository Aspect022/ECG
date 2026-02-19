import torch

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True

torch.backends.cudnn.allow_tf32 = True

torch.set_float32_matmul_precision("high")

"""
Unified Comparison Training Script.

Trains and evaluates multiple ECG model architectures:
  resnet       – ResNet1D baseline
  vit          – Standard Vision Transformer (1D)
  vit_snn      – ViT with Spiking attention (SNN)
  vit_quantum  – ViT with Quantum circuit path
  vit_hybrid   – ViT with SNN + Quantum (full hybrid)

Usage
-----
  # Train a single variant:
  python train_comparison.py --model_type vit_snn

  # Train ALL variants sequentially:
  python train_comparison.py --run_all

  # Quick smoke-test (2 epochs, small batch):
  python train_comparison.py --run_all --debug
"""

import os
import sys
import copy
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

# ── path setup ──
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.data.ptbxl import PTBXLDataset
from src.models.v2.comparison_models import create_comparison_model, MODEL_REGISTRY


# ════════════════════════════════════════════════════════
# Helpers (reused from train_hybrid_v2.py)
# ════════════════════════════════════════════════════════

class EarlyStopping:
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


class PowerMeter:
    def __init__(self, device_idx: int = 0):
        self.available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            self.available = True
        except Exception:
            pass

    def get_power(self) -> float:
        if not self.available:
            return 0.0
        try:
            import pynvml
            return pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
        except Exception:
            return 0.0


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ════════════════════════════════════════════════════════
# Data
# ════════════════════════════════════════════════════════

# Fix: Define fixed classes to ensure consistent mapping across all folds
FIXED_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def create_fold_dataloaders(config: dict, train_folds: list,
                            val_folds: list) -> tuple:
    data_cfg = config['data']
    exp_cfg = config['experiment']

    train_ds = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=train_folds,
        input_length=data_cfg['input_length'],
        augment=data_cfg.get('augment', True),
    )
    # Fix: Inject classes into train dataset
    train_ds.classes = FIXED_CLASSES
    # Re-process labels with fixed classes
    # removed broken call

    val_ds = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=val_folds,
        input_length=data_cfg['input_length'],
        augment=False,
    )
    # Fix: Inject classes into val dataset
    val_ds.classes = FIXED_CLASSES
    # Re-process labels with fixed classes
# removed broken label processing

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory'],
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory'],
    )
    
    # --- LOG CLASS DISTRIBUTION ---
    print(f"  Class Distribution (Filtered):")
    train_labels = np.array(train_ds.labels)
    counts = train_labels.sum(axis=0)
    for i, cls in enumerate(FIXED_CLASSES):
        print(f"    {cls:5s}: {int(counts[i]):>5d} samples")
        
    if counts.sum() == 0:
        print("  [CRITICAL] All labels are ZERO after filtering! Check mapping.")
    elif (counts > 0).sum() < 2:
        print("  [WARNING] Only ONE class has samples. Accuracy will be 1.0 trivially.")

    return train_loader, val_loader


def create_test_dataloader(config: dict) -> DataLoader:
    data_cfg = config['data']
    exp_cfg = config['experiment']

    test_ds = PTBXLDataset(
        data_path=data_cfg['data_path'],
        sampling_rate=data_cfg['sampling_rate'],
        task=data_cfg['task'],
        folds=[10],
        input_length=data_cfg['input_length'],
        augment=False,
    )
    # Fix: Inject classes into test dataset
    test_ds.classes = FIXED_CLASSES
    # Re-process labels with fixed classes
# removed broken label processing

    return DataLoader(
        test_ds,
        batch_size=data_cfg.get('test_batch_size', data_cfg['batch_size']),
        shuffle=False,
        num_workers=exp_cfg['num_workers'],
        pin_memory=exp_cfg['pin_memory'],
    )


# ════════════════════════════════════════════════════════
# Train / Validate
# ════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scaler, config, device,
                epoch, writer, prefix):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    gate_values = []

    accum = config['data'].get('accumulation_steps', 1)
    max_gn = config['training'].get('max_grad_norm', 1.0)

    pbar = tqdm(loader, desc=f'[{prefix}] Epoch {epoch}', leave=False)
    optimizer.zero_grad()

    # Diagnostic flag
    first_batch = True

    for idx, batch in enumerate(pbar):
        signal = batch['signal'].to(device)
        label = batch['label'].to(device)

        if first_batch and epoch == 1:
            # Check for label triviality
            if label.std() == 0:
                print(f"\n  [WARNING] First batch labels are identical! Value={label[0].tolist()}")
            first_batch = False

        with autocast(enabled=config['experiment'].get('use_amp', True)):
            output = model(signal, return_gate=True)
            logits = output['logits']
            reg_loss = output['reg_loss']

            if config['training'].get('criterion') == 'bce_with_logits':
                ce = F.binary_cross_entropy_with_logits(logits, label)
            else:
                lbl = label.argmax(dim=1) if label.dim() > 1 else label
                ce = F.cross_entropy(logits, lbl)

            loss = (ce + reg_loss) / accum

        scaler.scale(loss).backward()

        if (idx + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        with torch.no_grad():
            total_loss += loss.item() * accum
            
            # Fix: Match accuracy calculation to loss function
            if config['training'].get('criterion') == 'bce_with_logits':
                # Multi-label (Sigmoid)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == label).all(dim=1).sum().item()
            else:
                # Single-label (Argmax) - Matches CrossEntropy
                lbl = label.argmax(dim=1) if label.dim() > 1 else label
                preds = logits.argmax(dim=1)
                correct = (preds == lbl).sum().item()
            
            total_correct += correct
            total_samples += signal.shape[0]
            gate_values.append(output['gate'].mean().item())

        pbar.set_postfix(loss=f'{loss.item()*accum:.4f}',
                         acc=f'{total_correct/total_samples:.4f}')

    n = len(loader)
    avg_loss = total_loss / n
    accuracy = total_correct / total_samples
    avg_gate = float(np.mean(gate_values))

    writer.add_scalar(f'{prefix}/Train/Loss', avg_loss, epoch)
    writer.add_scalar(f'{prefix}/Train/Accuracy', accuracy, epoch)
    writer.add_scalar(f'{prefix}/Train/Gate_Mean', avg_gate, epoch)

    return {'loss': avg_loss, 'accuracy': accuracy, 'gate_mean': avg_gate}


@torch.no_grad()
def validate(model, loader, config, device, epoch, writer, prefix,
             power_meter=None):
    import time
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    latencies = []
    total_energy = 0.0

    for batch in tqdm(loader, desc=f'[{prefix}] Val', leave=False):
        signal = batch['signal'].to(device)
        label = batch['label'].to(device)
        bs = signal.shape[0]

        p0 = power_meter.get_power() if power_meter else 0.0
        t0 = time.time()

        with autocast(enabled=config['experiment'].get('use_amp', True)):
            output = model(signal)
            logits = output['logits']

            if config['training'].get('criterion') == 'bce_with_logits':
                loss = F.binary_cross_entropy_with_logits(logits, label)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct = (preds == label).all(dim=1).sum().item()
            else:
                lbl = label.argmax(dim=1) if label.dim() > 1 else label
                loss = F.cross_entropy(logits, lbl)
                # Fix: Single-label accuracy (Argmax)
                preds = logits.argmax(1)
                correct = (preds == lbl).sum().item()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.time()

        p1 = power_meter.get_power() if power_meter else 0.0
        avg_p = (p0 + p1) / 2.0
        latencies.append((t1 - t0) * 1000.0 / bs)
        total_energy += avg_p * (t1 - t0)

        total_loss += loss.item()
        total_correct += correct
        total_samples += bs

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples
    avg_lat = float(np.mean(latencies))
    energy_mj = (total_energy / total_samples) * 1000.0

    writer.add_scalar(f'{prefix}/Val/Loss', avg_loss, epoch)
    writer.add_scalar(f'{prefix}/Val/Accuracy', accuracy, epoch)
    writer.add_scalar(f'{prefix}/Val/Latency_ms', avg_lat, epoch)
    writer.add_scalar(f'{prefix}/Val/Energy_mJ', energy_mj, epoch)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'latency_ms': avg_lat,
        'energy_mj': energy_mj,
    }


# ════════════════════════════════════════════════════════
# Single-model training loop (K-Fold)
# ════════════════════════════════════════════════════════

def train_single_model(config: dict, model_type: str, output_root: Path,
                       writer: SummaryWriter, debug: bool = False,
                       k_folds: int = 5):
    """Train one model variant with K-Fold CV."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    power_meter = PowerMeter(0 if device.type == 'cuda' else -1)
    prefix = model_type  # TensorBoard prefix

    model_dir = output_root / model_type
    model_dir.mkdir(parents=True, exist_ok=True)

    if debug:
        config = copy.deepcopy(config)
        config['training']['epochs'] = 2
        config['data']['batch_size'] = 4
        config['experiment']['num_workers'] = 0
        k_folds = 2

    available_folds = list(range(1, 10))
    fold_size = len(available_folds) // k_folds
    fold_results = []

    print(f"\n{'='*70}")
    print(f"  MODEL: {MODEL_REGISTRY[model_type]}  ({model_type})")
    print(f"{'='*70}\n")

    for fi in range(k_folds):
        val_start = fi * fold_size
        val_folds = available_folds[val_start:val_start + fold_size]
        train_folds = [f for f in available_folds if f not in val_folds]

        print(f"  Fold {fi+1}/{k_folds}  train={train_folds}  val={val_folds}")

        train_loader, val_loader = create_fold_dataloaders(
            config, train_folds, val_folds)

        model = create_comparison_model(config, model_type).to(device)
        print(f"  Parameters: {model.num_parameters:,}")

        opt_cfg = config['optimizer']
        optimizer = AdamW(
            model.parameters(),
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            betas=(opt_cfg.get('beta1', 0.9), opt_cfg.get('beta2', 0.999)),
        )

        sched_cfg = config['scheduler']
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get('T_0', 10),
            T_mult=sched_cfg.get('T_mult', 2),
            eta_min=sched_cfg.get('eta_min', 1e-6),
        )

        scaler = GradScaler(enabled=config['experiment'].get('use_amp', True))
        es = EarlyStopping(patience=config['training'].get('patience', 15))
        best_val_acc = 0.0
        epochs = config['training']['epochs']

        for epoch in range(1, epochs + 1):
            tm = train_epoch(model, train_loader, optimizer, scaler,
                             config, device, epoch, writer, prefix)
            vm = validate(model, val_loader, config, device, epoch,
                          writer, prefix, power_meter)
            scheduler.step()

            print(f"    Ep {epoch:03d} | "
                  f"Acc {tm['accuracy']:.4f}/{vm['accuracy']:.4f} | "
                  f"Loss {tm['loss']:.4f}/{vm['loss']:.4f} | "
                  f"Gate {tm['gate_mean']:.3f}")

            if vm['accuracy'] > best_val_acc:
                best_val_acc = vm['accuracy']
                torch.save({
                    'model_type': model_type,
                    'fold': fi + 1,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_accuracy': best_val_acc,
                }, model_dir / f'best_fold_{fi+1}.pt')

            if es(vm['accuracy']):
                print(f"    Early stopping at epoch {epoch}")
                break

        fold_results.append({'fold': fi+1, 'best_val_acc': best_val_acc,
                             'final_val': vm})
        print(f"  Fold {fi+1} best: {best_val_acc:.4f}")

    # ── Aggregate ──
    accs = [r['best_val_acc'] for r in fold_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    print(f"\n  {model_type} Mean Acc: {mean_acc:.4f} ± {std_acc:.4f}")

    # ── Test on fold 10 ──
    best_fold = int(np.argmax(accs)) + 1
    test_loader = create_test_dataloader(config)
    ckpt = torch.load(model_dir / f'best_fold_{best_fold}.pt',
                      weights_only=False)
    model = create_comparison_model(config, model_type).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    test_m = validate(model, test_loader, config, device, 0,
                      writer, prefix, power_meter)

    print(f"  Test Acc: {test_m['accuracy']:.4f}  "
          f"Latency: {test_m['latency_ms']:.2f} ms  "
          f"Energy: {test_m['energy_mj']:.2f} mJ")

    # ── Save summary ──
    summary = {
        'model_type': model_type,
        'model_name': MODEL_REGISTRY[model_type],
        'k_folds': k_folds,
        'fold_accuracies': accs,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'test_accuracy': test_m['accuracy'],
        'test_latency_ms': test_m['latency_ms'],
        'test_energy_mj': test_m['energy_mj'],
        'num_parameters': model.num_parameters,
    }
    with open(model_dir / 'summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)

    return summary


# ════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Comparison Study: ResNet vs ViT variants')
    parser.add_argument('--config', type=str,
                        default='configs/comparison.yaml')
    parser.add_argument('--model_type', type=str, default='resnet',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Which model variant to train')
    parser.add_argument('--run_all', action='store_true',
                        help='Train ALL variants sequentially')
    parser.add_argument('--debug', action='store_true',
                        help='Quick 2-epoch smoke test')
    parser.add_argument('--k_folds', type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path(config['experiment']['output_dir']) / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config['logging']['log_dir']) / timestamp
    writer = SummaryWriter(log_dir)

    # Save config
    with open(output_root / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    variants = list(MODEL_REGISTRY.keys()) if args.run_all else [args.model_type]
    all_summaries = []

    for mt in variants:
        summary = train_single_model(
            config, mt, output_root, writer,
            debug=args.debug, k_folds=args.k_folds,
        )
        all_summaries.append(summary)

    # ── Final comparison table ──
    print(f"\n{'='*80}")
    print("  COMPARISON RESULTS")
    print(f"{'='*80}")
    print(f"  {'Model':<30s}  {'Params':>10s}  {'Val Acc':>10s}  "
          f"{'Test Acc':>10s}  {'Lat(ms)':>8s}  {'E(mJ)':>8s}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

    for s in all_summaries:
        print(f"  {s['model_name']:<30s}  "
              f"{s['num_parameters']:>10,}  "
              f"{s['mean_accuracy']:>10.4f}  "
              f"{s['test_accuracy']:>10.4f}  "
              f"{s['test_latency_ms']:>8.2f}  "
              f"{s['test_energy_mj']:>8.2f}")

    print(f"{'='*80}\n")

    # Save comparison
    with open(output_root / 'comparison_results.yaml', 'w') as f:
        yaml.dump(all_summaries, f, default_flow_style=False)

    writer.close()
    print(f"Results saved to: {output_root}")


if __name__ == '__main__':
    main()
