"""
Spiking Neural Network components.

Implements:
- Surrogate gradient functions for spike backpropagation
- Leaky Integrate-and-Fire (LIF) neurons
- Parametric LIF (PLIF) neurons with learnable parameters
- Spiking convolutional blocks
"""

import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple


# ============== Surrogate Gradient Functions ==============

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast sigmoid surrogate gradient for spike backpropagation.
    
    Forward: Heaviside step function
    Backward: Fast sigmoid gradient approximation
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, beta: float = 10.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.beta = beta
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, = ctx.saved_tensors
        beta = ctx.beta
        
        # Gradient: beta * 0.5 / (1 + beta * |x|)^2
        grad = beta * 0.5 / (1 + beta * torch.abs(input)) ** 2
        
        return grad_output * grad, None


class ArctanSurrogate(torch.autograd.Function):
    """
    Arctan surrogate gradient.
    Smoother gradient than fast sigmoid.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, = ctx.saved_tensors
        alpha = ctx.alpha
        
        # Gradient: alpha / (2 * (1 + (pi/2 * alpha * x)^2))
        grad = alpha / (2 * (1 + (torch.pi / 2 * alpha * input) ** 2))
        
        return grad_output * grad, None


class PiecewiseQuadraticSurrogate(torch.autograd.Function):
    """
    Piecewise quadratic surrogate gradient.
    Zero gradient outside a window, quadratic inside.
    """
    
    @staticmethod
    def forward(ctx, input: torch.Tensor, width: float = 1.0) -> torch.Tensor:
        ctx.save_for_backward(input)
        ctx.width = width
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        input, = ctx.saved_tensors
        width = ctx.width
        
        mask = torch.abs(input) < width
        grad = torch.zeros_like(input)
        grad[mask] = (1 - torch.abs(input[mask]) / width)
        
        return grad_output * grad, None


def get_surrogate_fn(name: str) -> Callable:
    """
    Get surrogate gradient function by name.
    
    Args:
        name: Name of surrogate function ('fast_sigmoid', 'arctan', 'piecewise')
        
    Returns:
        Callable: Surrogate gradient function
        
    Example:
        >>> spike_fn = get_surrogate_fn('fast_sigmoid')
        >>> spikes = spike_fn(membrane - threshold, beta=10.0)
    """
    surrogates = {
        'fast_sigmoid': FastSigmoidSurrogate.apply,
        'arctan': ArctanSurrogate.apply,
        'piecewise': PiecewiseQuadraticSurrogate.apply,
    }
    
    if name not in surrogates:
        raise ValueError(f"Unknown surrogate: {name}. Available: {list(surrogates.keys())}")
    
    return surrogates[name]


# ============== Neuron Models ==============

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron with surrogate gradients.
    
    The membrane potential evolves as:
        V[t] = decay * V[t-1] + I[t]
        S[t] = Heaviside(V[t] - threshold)
        V[t] = V[t] * (1 - S[t])  # Reset after spike
    
    Args:
        threshold: Firing threshold
        decay: Membrane potential decay factor (tau)
        surrogate: Surrogate gradient function name
        surrogate_beta: Beta parameter for surrogate gradient
        learnable: If True, threshold and decay are learnable
        
    Example:
        >>> lif = LIFNeuron(threshold=1.0, decay=0.9)
        >>> spikes, new_mem = lif(input_current, membrane)
    """
    
    def __init__(
        self,
        threshold: float = 1.0,
        decay: float = 0.9,
        surrogate: str = 'fast_sigmoid',
        surrogate_beta: float = 10.0,
        learnable: bool = False,
    ):
        super().__init__()
        
        if learnable:
            self.threshold = nn.Parameter(torch.tensor(threshold))
            self.decay = nn.Parameter(torch.tensor(decay))
        else:
            self.register_buffer('threshold', torch.tensor(threshold))
            self.register_buffer('decay', torch.tensor(decay))
        
        self.surrogate_beta = surrogate_beta
        self.spike_fn = get_surrogate_fn(surrogate)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of LIF neuron.
        
        Args:
            x: Input current (B, C, H, W) or (B, C)
            mem: Previous membrane potential (same shape as x)
            
        Returns:
            Tuple of (spikes, new_membrane)
        """
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Update membrane potential
        mem = self.decay * mem + x
        
        # Generate spikes using surrogate gradient
        spike = self.spike_fn(mem - self.threshold, self.surrogate_beta)
        
        # Reset membrane after spike (soft reset)
        mem = mem * (1.0 - spike)
        
        return spike, mem


class PLIFNeuron(LIFNeuron):
    """
    Parametric LIF neuron with learnable threshold and decay.
    
    Extends LIFNeuron with always-learnable parameters and
    optional separate thresholds per channel.
    
    Args:
        channels: Number of channels (for per-channel parameters)
        **kwargs: Additional arguments passed to LIFNeuron
        
    Example:
        >>> plif = PLIFNeuron(channels=64)
        >>> spikes, mem = plif(features, membrane)
    """
    
    def __init__(
        self,
        channels: int = 1,
        threshold: float = 1.0,
        decay: float = 0.9,
        surrogate: str = 'fast_sigmoid',
        surrogate_beta: float = 10.0,
        per_channel: bool = False,
    ):
        # Initialize parent without learnable (we'll set our own)
        nn.Module.__init__(self)
        
        if per_channel:
            self.threshold = nn.Parameter(torch.ones(channels) * threshold)
            self.decay = nn.Parameter(torch.ones(channels) * decay)
        else:
            self.threshold = nn.Parameter(torch.tensor(threshold))
            self.decay = nn.Parameter(torch.tensor(decay))
        
        self.surrogate_beta = surrogate_beta
        self.spike_fn = get_surrogate_fn(surrogate)
        self.per_channel = per_channel
    
    def forward(
        self, 
        x: torch.Tensor, 
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with per-channel support."""
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Ensure decay is in valid range [0, 1]
        decay = torch.sigmoid(self.decay)
        
        # Update membrane
        if self.per_channel and x.dim() == 4:
            # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
            decay_reshaped = decay.view(1, -1, 1, 1)
            threshold_reshaped = self.threshold.view(1, -1, 1, 1)
            mem = decay_reshaped * mem + x
            spike = self.spike_fn(mem - threshold_reshaped, self.surrogate_beta)
        else:
            mem = decay * mem + x
            spike = self.spike_fn(mem - self.threshold, self.surrogate_beta)
        
        mem = mem * (1.0 - spike)
        
        return spike, mem


# ============== Spiking Blocks ==============

class SpikingConvBlock(nn.Module):
    """
    Spiking convolutional block: Conv2d + BatchNorm + LIF.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        neuron_type: 'lif' or 'plif'
        **neuron_kwargs: Additional arguments for neuron
        
    Example:
        >>> block = SpikingConvBlock(32, 64, kernel_size=3)
        >>> spikes, mem = block(input_spikes, membrane)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        neuron_type: str = 'lif',
        **neuron_kwargs
    ):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
        if neuron_type == 'lif':
            self.neuron = LIFNeuron(**neuron_kwargs)
        elif neuron_type == 'plif':
            self.neuron = PLIFNeuron(channels=out_channels, **neuron_kwargs)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mem: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (spikes or features)
            mem: Previous membrane potential
            
        Returns:
            Tuple of (output_spikes, new_membrane)
        """
        x = self.conv(x)
        x = self.bn(x)
        spike, mem = self.neuron(x, mem)
        return spike, mem


class SpikingEncoder(nn.Module):
    """
    Multi-layer spiking encoder.
    Processes input over multiple timesteps.
    
    Args:
        in_channels: Input channels
        channels: List of channel sizes for each layer
        timesteps: Number of spike timesteps
        neuron_type: 'lif' or 'plif'
        
    Example:
        >>> encoder = SpikingEncoder(1, [32, 64, 128], timesteps=4)
        >>> output = encoder(input_image)  # Aggregated spike output
    """
    
    def __init__(
        self,
        in_channels: int,
        channels: list,
        timesteps: int = 4,
        neuron_type: str = 'lif',
        **neuron_kwargs
    ):
        super().__init__()
        
        self.timesteps = timesteps
        
        # Build layers
        layers = []
        prev_ch = in_channels
        for ch in channels:
            layers.append(SpikingConvBlock(prev_ch, ch, neuron_type=neuron_type, **neuron_kwargs))
            layers.append(nn.MaxPool2d(2))
            prev_ch = ch
        
        self.layers = nn.ModuleList(layers)
        self.out_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass over multiple timesteps.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Aggregated spike output
        """
        # Initialize membrane potentials
        mems = [None] * (len(self.layers) // 2)  # One per SpikingConvBlock
        
        # Accumulate spikes over timesteps
        spike_sum = None
        
        for t in range(self.timesteps):
            out = x
            mem_idx = 0
            
            for layer in self.layers:
                if isinstance(layer, SpikingConvBlock):
                    out, mems[mem_idx] = layer(out, mems[mem_idx])
                    mem_idx += 1
                else:
                    out = layer(out)
            
            if spike_sum is None:
                spike_sum = out
            else:
                spike_sum = spike_sum + out
        
        # Average spike rate
        return spike_sum / self.timesteps
