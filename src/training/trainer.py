
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import time

# Assumption: MetricsCalculator imported from peer directory
import sys
# Make sure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluation.metrics import MetricsCalculator

class ECGTrainer:
    """
    Unified Training Pipeline for ECG Classification.
    Handles Mix Precision (AMP), Gradient Clipping, Scheduling, and Metrics.
    """
    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config['optimizer']['lr'], 
            weight_decay=config['optimizer']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['training']['epochs'], 
            eta_min=config['scheduler']['min_lr']
        )
        
        # Loss
        if config['training']['criterion'] == 'bce_with_logits':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        # FP16 Scaler
        self.scaler = GradScaler(enabled=config['experiment']['use_amp'])
        
        # Metrics
        self.metrics = MetricsCalculator(num_classes=config['model']['num_classes'])
        
        # Logging
        self.log_dir = os.path.join(config['experiment']['output_dir'], 'logs')
        self.ckpt_dir = os.path.join(config['experiment']['output_dir'], 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        self.best_metric = 0.0
        
    def train(self):
        print(f"Starting training on {self.device}...")
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            
            # --- TRAIN ---
            train_metrics = self._train_epoch(epoch)
            
            # --- VALIDATE ---
            val_metrics = self._validate_epoch(epoch)
            
            # --- SCHEDULER ---
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # --- LOGGING ---
            duration = time.time() - start_time
            print(f"Epoch {epoch}/{self.config['training']['epochs']} | "
                  f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} AUC: {val_metrics['auroc_macro']:.4f} | "
                  f"LR: {current_lr:.2e} | Time: {duration:.1f}s")
            
            # --- CHECKPOINT ---
            # Save periodic
            if epoch % self.config['experiment']['save_every'] == 0:
                self._save_checkpoint(epoch, val_metrics)
                
            # Save Best (Track Macro AUROC)
            if val_metrics['auroc_macro'] > self.best_metric:
                self.best_metric = val_metrics['auroc_macro']
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                print(f"  >>> New Best Model! AUC: {self.best_metric:.4f}")

    def _train_epoch(self, epoch):
        self.model.train()
        self.metrics.reset()
        epoch_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch}", leave=False)
        
        for batch in pbar:
            inputs = batch['signal'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Optional: Clinical Features
            clinical = None
            if 'clinical_features' in batch:
                clinical = batch['clinical_features'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed Precision Forward
            with autocast(enabled=self.config['experiment']['use_amp']):
                if clinical is not None:
                    outputs = self.model(inputs, clinical)
                else:
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
            
            # Backward
            self.scaler.scale(loss).backward()
            
            # Clip Gradients
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['max_grad_norm'])
            
            # Step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            
            # Update Metrics (convert logits to probabilities/preds)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            self.metrics.update(preds, labels, probs)
            
            pbar.set_postfix({'loss': loss.item()})
            
        avg_metrics = self.metrics.compute()
        avg_metrics['loss'] = epoch_loss / len(self.train_loader)
        return avg_metrics

    def _validate_epoch(self, epoch):
        self.model.eval()
        self.metrics.reset()
        epoch_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['signal'].to(self.device)
                labels = batch['label'].to(self.device)
                clinical = chunk = None
                if 'clinical_features' in batch:
                    clinical = batch['clinical_features'].to(self.device)
                
                if clinical is not None:
                    outputs = self.model(inputs, clinical)
                else:
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                self.metrics.update(preds, labels, probs)
                
        avg_metrics = self.metrics.compute()
        avg_metrics['loss'] = epoch_loss / len(self.val_loader)
        return avg_metrics

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_metrics': metrics,
            'config': self.config
        }
        
        filename = "best_model.pt" if is_best else f"checkpoint_ep{epoch}.pt"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, path)
