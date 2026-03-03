import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

class ECGGradCAM:
    """
    Computes Grad-CAM (Gradient-weighted Class Activation Mapping) for 1D ECG Signals.
    Helps identify which parts of the ECG signal the model focuses on for its predictions.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks for capturing gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)
        
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generates Grad-CAM array for a given input tensor.
        
        Args:
            input_tensor (torch.Tensor): Output from dataloader, shape (1, leads, sequence_length)
            target_class (int): Class to compute gradients for. If None, uses predicted class.
            
        Returns:
            np.ndarray: Grad-CAM heatmap normalized between 0 and 1, shape matches sequence_length.
        """
        self.model.eval()
        
        # Ensure single batch dimension
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(0)
            
        # Enable gradient tracking
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Hybrid model usually returns a dict
        if isinstance(output, dict):
            logits = output['logits']
        else:
            logits = output
            
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
            
        # Clear previous gradients
        self.model.zero_grad()
        
        # Target score for backprop
        score = logits[0, target_class]
        score.backward()
        
        # Compute Grad-CAM
        # Average pooling of gradients over the sequence dimension (1D mapping)
        weights = torch.mean(self.gradients, dim=2, keepdim=True) # (1, channels, 1)
        
        # Weighted sum of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0) # (seq_len_reduced)
        
        # ReLU to keep only features that have a positive influence
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
            
        cam = cam.cpu().numpy()
        
        # Interpolate back to original sequence length
        original_length = input_tensor.shape[-1]
        cam_resized = cv2.resize(cam.reshape(1, -1), (original_length, 1))
        
        return cam_resized.flatten()
        
    def plot_heatmap(self, signal: np.ndarray, cam: np.ndarray, save_path: str = None, title: str = "ECG Grad-CAM"):
        """Plots the original signal overlaid with the Grad-CAM heatmap."""
        plt.figure(figsize=(15, 5))
        
        # Plot signal (if multiple leads, just plot the first or average)
        if len(signal.shape) == 2:
            sig_to_plot = signal[0] # take first lead
        else:
            sig_to_plot = signal
            
        plt.plot(sig_to_plot, label='ECG Signal', color='blue', alpha=0.7)
        
        # Overlay heatmap
        # Create an extent that matches the signal x-axis, and stretches over the y-axis bounds
        y_min, y_max = np.min(sig_to_plot), np.max(sig_to_plot)
        extent = [0, len(sig_to_plot), y_min, y_max]
        
        # We need a 2D array for imshow
        cam_2d = np.expand_dims(cam, axis=0)
        
        plt.imshow(cam_2d, cmap='jet', alpha=0.4, aspect='auto', extent=extent)
        
        plt.title(title)
        plt.xlabel('Time Step')
        plt.ylabel('Amplitude')
        plt.colorbar(label='Importance')
        
        if save_path:
            # Ensure dir exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
