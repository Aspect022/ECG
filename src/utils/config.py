"""
Configuration loading and saving utilities.
"""

import os
import yaml
from typing import Dict, Any, Optional
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        
    Example:
        >>> config = load_config('configs/default.yaml')
        >>> print(config['training']['epochs'])
        40
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path to save the configuration
        
    Example:
        >>> config = {'epochs': 40, 'lr': 0.001}
        >>> save_config(config, 'configs/experiment.yaml')
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], 
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base values
        
    Returns:
        dict: Merged configuration
        
    Example:
        >>> base = {'training': {'epochs': 40, 'lr': 0.001}}
        >>> override = {'training': {'epochs': 100}}
        >>> merged = merge_configs(base, override)
        >>> print(merged['training']['epochs'])
        100
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_experiment_name(config: Dict[str, Any], 
                        prefix: Optional[str] = None) -> str:
    """
    Generate experiment name from configuration.
    
    Args:
        config: Configuration dictionary
        prefix: Optional prefix for the experiment name
        
    Returns:
        str: Experiment name
        
    Example:
        >>> config = {'model': {'architecture': 'vit'}, 'data': {'dataset': 'ptbxl'}}
        >>> get_experiment_name(config)
        'vit_ptbxl_20241214_014500'
    """
    arch = config.get('model', {}).get('architecture', 'model')
    dataset = config.get('data', {}).get('dataset', 'data')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    parts = []
    if prefix:
        parts.append(prefix)
    parts.extend([arch, dataset, timestamp])
    
    return '_'.join(parts)
