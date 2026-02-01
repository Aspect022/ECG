"""
ECG Classification Package

A journal-level ECG classification system using:
- Convolutional Neural Networks (CNN)
- Spiking Neural Networks (SNN) 
- Vision Transformers (ViT)
- Multiple attention mechanisms
- Quantization strategies (QAT, PTQ, SNN quantization)
- Hybrid quantum-classical models (PennyLane)

Optimized for CPU training on AMD Ryzen 5000 series.
"""

__version__ = "0.1.0"
__author__ = "ECG Research Team"

from . import data
from . import models
from . import quantization
from . import training
from . import evaluation
from . import utils
