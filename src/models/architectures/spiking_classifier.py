import torch
import torch.nn as nn
from ..backbones.snn import SpikingEncoder

class SpikingClassifier(nn.Module):
    """
    Spiking Neural Network Classifier for ECG.
    
    Wraps the SpikingEncoder backbone and adds a classification head.
    The output is compatible with standard loss functions (logits).
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        channels: list = [32, 64, 128, 256],
        timesteps: int = 4,
        neuron_type: str = 'lif',
        **neuron_kwargs
    ):
        super().__init__()
        
        self.encoder = SpikingEncoder(
            in_channels=in_channels,
            channels=channels,
            timesteps=timesteps,
            neuron_type=neuron_type,
            **neuron_kwargs
        )
        
        # Determine the number of output features from the encoder
        # The encoder uses MaxPool2d(2) after each block.
        # Assuming input length 1000 and 4 layers:
        # 1000 -> 500 -> 250 -> 125 -> 62 (approx)
        # We will use AdaptiveAvgPool to handle variable sizes
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def forward(self, x: torch.Tensor, clinical_features=None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, L)
            clinical_features: Ignored, for API compatibility
            
        Returns:
            torch.Tensor: Logits (B, num_classes)
        """
        # SpikingEncoder expects 4D input (B, C, H, W) or can handle (B, C, L) if Conv1d?
        # The SpikingConvBlock uses Conv2d.
        # We need to reshape the 1D ECG signal to "fake" 2D or use Conv1d in the backbone.
        # Looking at snn.py again, SpikingConvBlock uses nn.Conv2d.
        # So we MUST reshape (B, C, L) -> (B, C, 1, L)
        
        if x.dim() == 3:
            x = x.unsqueeze(2) # (B, C, 1, L)
            
        x = self.encoder(x)
        
        # Encoder returns (B, C, H, W) averaged over time.
        # Since H=1, we have (B, C_out, 1, L_out).
        # We flatten to (B, C_out, L_out) for adaptive pool
        
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
