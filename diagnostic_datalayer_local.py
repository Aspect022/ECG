
import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import ast
import wfdb

# Bypass package imports
# Define the class locally here to be 100% sure what we are checking
# since we can't easily import it due to torchvision error in __init__.py

class PTBXLDatasetCheck:
    def __init__(self, data_path, folds, task='superclass', classes=None):
        self.data_path = data_path
        self.folds = folds
        self.task = task
        self.df = pd.read_csv(os.path.join(data_path, 'ptbxl_database.csv'), index_col='ecg_id')
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))
        self.agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
        self.df = self.df[self.df.strat_fold.isin(self.folds)]
        self.classes = classes
        self.labels, self.mlb = self._process_labels()

    def _process_labels(self):
        if self.task == 'superclass':
            # Note: We simulate the logic in ptbxl.py
            agg_df = self.agg_df[self.agg_df.diagnostic == 1]
            def aggregate_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        tmp.append(agg_df.loc[key].diagnostic_class)
                return list(set(tmp))
            
            self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)
            from sklearn.preprocessing import MultiLabelBinarizer
            if self.classes:
                mlb = MultiLabelBinarizer(classes=self.classes)
                # CRITICAL TEST: Does fit() re-sort if classes is provided?
                mlb.fit([self.classes]) 
                y = mlb.transform(self.df['diagnostic_superclass'])
            else:
                mlb = MultiLabelBinarizer()
                y = mlb.fit_transform(self.df['diagnostic_superclass'])
            return y, mlb
        return None, None

def diagnostic_check():
    data_path = "d:/Projects/ECG/data/ptbxl"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return

    FIXED_CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
    
    print("--- Diagnostic: Dataset Check (Local Class Copy) ---")
    ds = PTBXLDatasetCheck(data_path=data_path, folds=[1,2,3,4,5,6,7,8], task='superclass', classes=FIXED_CLASSES)
    
    print(f"Fixed classes passed in: {FIXED_CLASSES}")
    print(f"MLB classes (after fit): {list(ds.mlb.classes_)}")
    
    if list(ds.mlb.classes_) != FIXED_CLASSES:
        print("[WARNING] MLB classes do not match fixed classes order!")
    
    y = ds.labels
    total = len(y)
    empty = (y.sum(axis=1) == 0).sum()
    print(f"Total samples: {total}")
    print(f"Samples with NO labels: {empty} ({empty/total:.2%})")
    
    if total > 0:
        class_counts = y.sum(axis=0)
        for i, cls in enumerate(ds.mlb.classes_):
            print(f"  Class {i} ({cls}): {int(class_counts[i])} samples")

if __name__ == "__main__":
    diagnostic_check()
