import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    balanced_accuracy_score, matthews_corrcoef,
    roc_curve, precision_recall_curve, brier_score_loss
)

class AdvancedMetricsCalculator:
    """
    Computes research-grade medical and architectural metrics for ECG Classification.
    """
    
    def __init__(self, num_classes: int = 2, output_dir: str = 'runs/metrics'):
        self.num_classes = num_classes
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(self, preds: np.ndarray, targets: np.ndarray, probs: np.ndarray):
        self.predictions.append(preds)
        self.targets.append(targets)
        self.probabilities.append(probs)
        
    def _compute_clinical_metrics(self, y_true, y_pred, metrics):
        """Computes TP, TN, FP, FN, Specificity, FAR, Miss Rate, Youden, LRs"""
        if self.num_classes == 1 or len(y_true.shape) == 1 or (len(y_true.shape) == 2 and y_true.shape[1] == 1):
            # Binary case
            try:
                tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
            except ValueError:
                # Handle cases where only one class is present in the batch/dataset
                tn, fp, fn, tp = 0, 0, 0, 0
                
            metrics['TP'] = int(tp)
            metrics['TN'] = int(tn)
            metrics['FP'] = int(fp)
            metrics['FN'] = int(fn)
            
            # Clinical derivations
            sensitivity = tp / (tp + fn + 1e-9)
            specificity = tn / (tn + fp + 1e-9)
            
            metrics['Sensitivity'] = sensitivity
            metrics['Specificity'] = specificity
            metrics['FAR'] = fp / (fp + tn + 1e-9)
            metrics['Miss_Rate'] = fn / (fn + tp + 1e-9)
            metrics['Youden_Index'] = sensitivity + specificity - 1.0
            
            metrics['LR+'] = sensitivity / (1 - specificity + 1e-9)
            metrics['LR-'] = (1 - sensitivity) / (specificity + 1e-9)
        else:
            # Multi-class / Multi-label case
            metrics['Sensitivity'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
            
            # Approximation for macro specificity in multi-label
            tps = (y_true * y_pred).sum(axis=0)
            fps = ((1 - y_true) * y_pred).sum(axis=0)
            fns = (y_true * (1 - y_pred)).sum(axis=0)
            tns = ((1 - y_true) * (1 - y_pred)).sum(axis=0)
            
            spec_per_class = tns / (tns + fps + 1e-9)
            metrics['Specificity'] = float(np.mean(spec_per_class))
            
            far_per_class = fps / (fps + tns + 1e-9)
            metrics['FAR'] = float(np.mean(far_per_class))
            
            miss_per_class = fns / (fns + tps + 1e-9)
            metrics['Miss_Rate'] = float(np.mean(miss_per_class))
            
            metrics['Youden_Index'] = metrics['Sensitivity'] + metrics['Specificity'] - 1.0

    def compute(self, save_plots=False, prefix="val") -> dict:
        if not self.predictions:
            return {}
            
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)
        y_prob = np.concatenate(self.probabilities)
        
        metrics = {}
        
        # 1. Base Classification
        metrics['Accuracy'] = accuracy_score(y_true.ravel(), y_pred.ravel())
        metrics['Balanced_Accuracy'] = balanced_accuracy_score(y_true.ravel(), y_pred.ravel())
        metrics['MCC'] = matthews_corrcoef(y_true.ravel(), y_pred.ravel())
        
        # Macro/Micro
        metrics['Precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['Sensitivity_Macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['F1-score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 2. Clinical Variants
        self._compute_clinical_metrics(y_true, y_pred, metrics)
        
        # 3. Probabilistic Algorithms
        try:
            metrics['AUC-ROC'] = float(roc_auc_score(y_true, y_prob, average='macro'))
            metrics['AUC-PR'] = float(average_precision_score(y_true, y_prob, average='macro'))
        except ValueError:
            metrics['AUC-ROC'] = 0.0
            metrics['AUC-PR'] = 0.0
            
        # 4. Reliability - Expected Calibration Error (Brier Score)
        # Using flat arrays for overall calibration score
        metrics['Brier_Score'] = brier_score_loss(y_true.ravel(), y_prob.ravel())

        if save_plots:
            self.plot_confusion_matrix(os.path.join(self.output_dir, f'{prefix}_confusion_matrix.png'))
            self.plot_roc_curve(os.path.join(self.output_dir, f'{prefix}_roc_curve.png'))
            
        return metrics

    def plot_confusion_matrix(self, path: str, class_names=None):
        """Saves Confusion Matrix."""
        if not self.predictions: return
        y_pred = np.concatenate(self.predictions)
        y_true = np.concatenate(self.targets)
        
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix (Aggregated)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(path)
        plt.close()

    def plot_roc_curve(self, path: str):
        """ROC Curve for Class 0 (representative)"""
        if not self.predictions: return
        y_true = np.concatenate(self.targets)
        y_prob = np.concatenate(self.probabilities)
        
        if self.num_classes > 1 and len(np.unique(y_true[:, 0])) == 2:
            fpr, tpr, _ = roc_curve(y_true[:, 0], y_prob[:, 0])
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (Class 0)')
            plt.legend(loc="lower right")
            plt.savefig(path)
            plt.close()

    def _save_plots(self, y_true, y_pred, y_prob, prefix):
        """Saves Confusion Matrix, ROC, and PR curves."""
        # 1. Confusion Matrix (Aggregate if multi-label)
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true.ravel(), y_pred.ravel())
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{prefix.capitalize()} Confusion Matrix (Aggregated)')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_confusion_matrix.png'))
        plt.close()

        # 2. ROC Curve for Class 0 (representative)
        if self.num_classes > 1 and len(np.unique(y_true[:, 0])) == 2:
            fpr, tpr, _ = roc_curve(y_true[:, 0], y_prob[:, 0])
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{prefix.capitalize()} ROC Curve (Class 0)')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.output_dir, f'{prefix}_roc_curve.png'))
            plt.close()
