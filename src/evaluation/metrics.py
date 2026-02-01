
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class MetricsCalculator:
    """
    Calculates comprehensive metrics for Multi-Label Classification.
    
    Metrics:
    - Exact Match Accuracy (Subset Accuracy)
    - Macro/Micro F1, Precision, Recall
    - Macro/Micro AUROC
    - Average Precision (AP)
    """
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.predictions = []
        self.targets = []
        self.probabilities = []
        
    def update(self, preds: torch.Tensor, targets: torch.Tensor, probs: torch.Tensor):
        """
        Update metrics with batch data.
        Args:
            preds: Binary predictions [Batch, NumClasses]
            targets: Ground truth [Batch, NumClasses]
            probs: Probabilities [Batch, NumClasses]
        """
        self.predictions.append(preds.cpu().numpy())
        self.targets.append(targets.cpu().numpy())
        self.probabilities.append(probs.detach().cpu().numpy())
        
    def compute(self) -> dict:
        """Compute all metrics over the accumulated data."""
        y_pred = np.vstack(self.predictions)
        y_true = np.vstack(self.targets)
        y_prob = np.vstack(self.probabilities)
        
        metrics = {}
        
        # 1. Exact Match Accuracy (all labels must match)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 2. Hamming Loss (fraction of wrong labels)
        metrics['hamming_accuracy'] = (y_pred == y_true).mean()
        
        # 3. Macro metrics (averaged over classes)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # 4. Micro metrics (averaged over samples)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # 5. ROC-AUC (handling cases where a class is not present)
        try:
            metrics['auroc_macro'] = roc_auc_score(y_true, y_prob, average='macro')
            metrics['auroc_micro'] = roc_auc_score(y_true, y_prob, average='micro')
        except ValueError:
            metrics['auroc_macro'] = 0.0
            metrics['auroc_micro'] = 0.0
            
        # 6. Average Precision (PR Area)
        try:
            metrics['ap_macro'] = average_precision_score(y_true, y_prob, average='macro')
        except ValueError:
             metrics['ap_macro'] = 0.0
            
        return metrics

if __name__ == "__main__":
    calc = MetricsCalculator(num_classes=5)
    # Mock
    y_true = torch.tensor([[0, 1, 0, 0, 0], [1, 0, 0, 1, 0]])
    y_prob = torch.tensor([[0.1, 0.9, 0.2, 0.1, 0.0], [0.8, 0.2, 0.1, 0.85, 0.1]])
    y_pred = (y_prob > 0.5).float()
    
    calc.update(y_pred, y_true, y_prob)
    print(calc.compute())
