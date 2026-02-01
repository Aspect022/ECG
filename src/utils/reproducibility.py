"""
Utility functions for reproducibility and environment setup.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        
    Example:
        >>> set_seed(42)
        >>> torch.rand(3)  # Will be reproducible
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # For CUDA (even if not available, these are no-ops)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)


def configure_cpu_environment(num_cores: Optional[int] = None) -> int:
    """
    Configure optimal settings for AMD Ryzen 5000 CPU training.
    
    Args:
        num_cores: Number of physical CPU cores. If None, auto-detects or uses env var.
        
    Returns:
        int: Number of threads configured
        
    Example:
        >>> num_threads = configure_cpu_environment(6)
        >>> print(f"Using {num_threads} threads")
    """
    # Determine number of cores
    if num_cores is None:
        num_cores = int(os.environ.get('CPU_CORES', os.cpu_count() // 2 or 6))
    
    # Set PyTorch thread count
    torch.set_num_threads(num_cores)
    
    # Set environment variables for various backends
    os.environ['OMP_NUM_THREADS'] = str(num_cores)
    os.environ['MKL_NUM_THREADS'] = str(num_cores)
    os.environ['NUMEXPR_NUM_THREADS'] = str(num_cores)
    
    # Enable MKL-DNN optimizations (works well on AMD)
    torch.backends.mkldnn.enabled = True
    
    # Set quantization backend for x86
    torch.backends.quantized.engine = 'fbgemm'
    
    return num_cores


def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: Best available device
        
    Example:
        >>> device = get_device()
        >>> model = model.to(device)
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        int: Number of parameters
        
    Example:
        >>> model = nn.Linear(10, 5)
        >>> count_parameters(model)
        55
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_size(size_bytes: int) -> str:
    """
    Format byte size to human readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def get_model_size(model: torch.nn.Module, precision: str = 'float32') -> int:
    """
    Calculate model size in bytes.
    
    Args:
        model: PyTorch model
        precision: 'float32', 'float16', 'int8', or 'int4'
        
    Returns:
        int: Model size in bytes
    """
    num_params = count_parameters(model, trainable_only=False)
    
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'int8': 1,
        'int4': 0.5,
    }
    
    return int(num_params * bytes_per_param.get(precision, 4))
