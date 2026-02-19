
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.data.ptbxl import PTBXLDataset

# The fixed classes we expect
FIXED_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

def verify_dataset_fixes():
    data_path = "d:/Projects/ECG/data/ptbxl"
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Data path not found: {data_path}")
        print("Please run this script on the machine where the dataset is located.")
        return

    print(f"\nExample Fixed Classes: {FIXED_CLASSES}")
    
    # 1. Initialize Train Dataset with Fixed Classes
    print("\n--- Initializing Train Dataset (Folds 1-8) ---")
    train_ds = PTBXLDataset(
        data_path=data_path, 
        folds=[1, 2, 3, 4, 5, 6, 7, 8], 
        task='superclass'
    )
    # Manual injection as done in train_comparison.py
    train_ds.classes = FIXED_CLASSES
    train_ds.labels, train_ds.mlb = train_ds._process_labels()
    
    print(f"Train Classes: {train_ds.mlb.classes_}")
    
    # 2. Initialize Val Dataset with Fixed Classes
    print("\n--- Initializing Val Dataset (Fold 9) ---")
    val_ds = PTBXLDataset(
        data_path=data_path, 
        folds=[9], 
        task='superclass'
    )
    # Manual injection as done in train_comparison.py
    val_ds.classes = FIXED_CLASSES
    val_ds.labels, val_ds.mlb = val_ds._process_labels()
    
    print(f"Val Classes:   {val_ds.mlb.classes_}")
    
    # 3. Verification Assertions
    print("\n--- Verifying Consistency ---")
    
    # Check Class Consistency
    if np.array_equal(train_ds.mlb.classes_, val_ds.mlb.classes_):
        print("[PASS] Train and Val classes match exactly.")
    else:
        print("[FAIL] Train and Val classes DO NOT match!")
        print(f"Train: {train_ds.mlb.classes_}")
        print(f"Val:   {val_ds.mlb.classes_}")
        
    # Check Empty Labels in a subset
    print("\n--- Checking for Empty Labels (First 1000 samples) ---")
    empty_count = 0
    for i in range(min(1000, len(train_ds))):
        label = train_ds[i]['label']
        if label.sum() == 0:
            empty_count += 1
            
    if empty_count > 0:
        print(f"[WARNING] Found {empty_count} samples with NO labels in the first 1000 samples.")
        print("Current fix includes a warning in PTBXLDataset for this.")
    else:
        print("[PASS] No empty labels found in the checked subset.")

    print("\n--- Verification Complete ---")
    print("If [PASS] is seen above, the Data Layer fixes are working correctly.")

if __name__ == "__main__":
    verify_dataset_fixes()
