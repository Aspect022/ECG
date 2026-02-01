"""
Logging utilities for experiment tracking.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import csv


def setup_logger(
    name: str = 'ecg_classification',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        logging.Logger: Configured logger
        
    Example:
        >>> logger = setup_logger('training', 'results/logs')
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricsLogger:
    """
    Logger for tracking metrics during training.
    Logs to both CSV and optionally TensorBoard.
    
    Example:
        >>> logger = MetricsLogger('results/logs', 'training')
        >>> logger.log({'epoch': 1, 'loss': 0.5, 'accuracy': 0.85})
        >>> logger.close()
    """
    
    def __init__(
        self, 
        log_dir: str, 
        name: str,
        use_tensorboard: bool = True
    ):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            name: Name for the log files
            use_tensorboard: Whether to use TensorBoard logging
        """
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(log_dir, f'{name}_{timestamp}.csv')
        self.csv_file = None
        self.csv_writer = None
        self.header_written = False
        
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, 'tensorboard', f'{name}_{timestamp}')
                self.writer = SummaryWriter(tb_dir)
            except ImportError:
                self.use_tensorboard = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log metrics to CSV and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Step number (defaults to epoch if present)
        """
        # CSV logging
        if self.csv_file is None:
            self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=list(metrics.keys()))
            self.csv_writer.writeheader()
            self.header_written = True
        
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        
        # TensorBoard logging
        if self.use_tensorboard and self.writer:
            if step is None:
                step = metrics.get('epoch', 0)
            
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(key, value, step)
    
    def close(self) -> None:
        """Close all log files."""
        if self.csv_file:
            self.csv_file.close()
        if self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
