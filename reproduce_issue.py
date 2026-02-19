
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / 'src'))

from src.data.ptbxl import PTBXLDataset

def test_label_consistency():
    data_path = "d:/Projects/ECG/data/ptbxl"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"Data path not found: {data_path}")
        return

    print("Creating Train Dataset (Folds 1-8)...")
    train_ds = PTBXLDataset(data_path=data_path, folds=[1, 2, 3, 4, 5, 6, 7, 8], task='superclass')
    print(f"Train Classes: {train_ds.mlb.classes_}")
    
    print("\nCreating Val Dataset (Fold 9)...")
    val_ds = PTBXLDataset(data_path=data_path, folds=[9], task='superclass')
    print(f"Val Classes: {val_ds.mlb.classes_}")
    
    # Check for consistency
    if len(train_ds.mlb.classes_) != len(val_ds.mlb.classes_):
        print("\n[CRITICAL] Class count mismatch!")
    elif not all(train_ds.mlb.classes_ == val_ds.mlb.classes_):
        print("\n[CRITICAL] Class name mismatch!")
    else:
        print("\n[OK] Classes match.")

    # Check a sample label
    print(f"\nSample Label (Train): {train_ds[0]['label']}")

if __name__ == "__main__":
    test_label_consistency()
