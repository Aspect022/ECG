"""
V2.0 Hybrid Quantum-Classical ECG Model.

This module provides the complete dual-path architecture for ECG classification.
"""

from .attention import (
    MultiScaleSpikingAttention,
    GlobalSpikingAttention,
    LocalWindowAttention,
    RegionalDilatedAttention,
    GlobalPooledAttention,
)
from .classical_path import (
    ClassicalPath,
    LocalPatternExtractor,
    BeatPatternExtractor,
    RhythmPatternExtractor,
    TemporalFusionBlock,
)
from .quantum_circuit import (
    VectorizedQuantumCircuit,
    QuantumMeasurement,
    QuantumGates,
)
from .quantum_path import (
    QuantumPath,
    FeatureCompressor,
    QuantumEncodingLayer,
)
from .fusion import (
    GatedFusionModule,
    ClassificationHead,
)
from .hybrid_model import (
    HybridQuantumClassicalECG,
    HybridModelConfig,
    create_model,
)

__all__ = [
    # Main model
    'HybridQuantumClassicalECG',
    'HybridModelConfig',
    'create_model',
    
    # Classical path
    'ClassicalPath',
    'LocalPatternExtractor',
    'BeatPatternExtractor',
    'RhythmPatternExtractor',
    'TemporalFusionBlock',
    
    # Attention
    'MultiScaleSpikingAttention',
    'GlobalSpikingAttention',
    
    # Quantum path
    'QuantumPath',
    'VectorizedQuantumCircuit',
    'QuantumMeasurement',
    'FeatureCompressor',
    
    # Fusion
    'GatedFusionModule',
    'ClassificationHead',
]
