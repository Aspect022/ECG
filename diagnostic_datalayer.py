
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.data.ptbxl import PTBXLDataset

FIXED_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def diagnostic_check():
    data_path = "d:/Projects/ECG/data/ptbxl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    print("--- Diagnostic: Dataset Check ---")
    ds = PTBXLDataset(data_path=data_path, folds=[1], task='superclass')
    
    # Check if classes are correctly handled
    print(f"Original Classes: {ds.mlb.classes_}")
    
    # Apply the 'fix' from train_comparison.py
    ds.classes = FIXED_CLASSES
    ds.labels, ds.mlb = ds._process_labels()
    print(f"Injected Classes: {ds.mlb.classes_}")
    
    # Check labels
    y = ds.labels
    total = len(y)
    empty = (y.sum(axis=1) == 0).sum()
    print(f"Total samples: {total}")
    print(f"Samples with NO labels: {empty} ({empty/total:.2%})")
    
    if total > 0:
        class_counts = y.sum(axis=0)
        for i, cls in enumerate(ds.mlb.classes_):
            print(f"  Class {i} ({cls}): {int(class_counts[i])} samples")

    # Check first label
    sample = ds[0]
    print(f"\nSample 0 Label: {sample['label']}")
    print(f"Sample 0 Signal Stats: mean={sample['signal'].mean():.4f}, std={sample['signal'].std():.4f}")

if __name__ == "__main__":
    diagnostic_check()
