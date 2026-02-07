# Training Diagnosis Report: 100% Accuracy Anomaly

## 1. Immediate Diagnosis: Is 100% Accuracy Normal?
**NO.** Achieving **100.00% validation accuracy** on PTB-XL (a real clinical dataset) by Epoch 2 is **impossible** under normal conditions. The State-of-the-Art (SOTA) is typically around **80-90%** for super-class classification (depending on the split).

### Likely Cause: Data Leakage
The model is seeing the exact same patients in both **Train** and **Validation** sets.
- This means the model is simply "memorizing" the signals it has already trained on.
- **Evidence**:
  - `Epoch 1: Train/Val Acc: 0.57/0.99` (Validation much higher than Train immediately).
  - `Epoch 7+: 1.0000/1.0000` (Perfect score).

**Action Required**:
Check your `ptbxl_database.csv`. The `strat_fold` column must be correct (1-10). If you are using a subset or a custom CSV, ensure `strat_fold` 9 is distinct from 1-8. If using a small demo dataset with <100 samples, specific folds might be empty or overlap.

---

## 2. Model Parameter Analysis (Why it's > 50k)

You requested a list of parameters. Based on the code in `src/models/v2`, the current implementation is **significantly larger** than the "Green AI" target (< 50,000 parameters).

### Breakdown
| Module | Component | Layer | Calculated Params | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `LocalPatternExtractor` | Depthwise Conv | 36 | 12*1*3 |
| | | Pointwise Conv | 3,072 | 12*256*1 |
| **Stage 2** | `BeatPatternExtractor` | Conv1d (k=9) | **294,912** | 256*128*9 (Huge!) |
| **Stage 3** | `RhythmPatternExtractor`| Conv1d (k=27) | **221,184** | 128*64*27 (Huge!) |
| **Fusion** | `TemporalFusionBlock` | Linear Proj | 57,344 | 448*128 |
| **SNN** | `LIFNeuron` | (No weights) | 0 | Activation only |
| **Quantum** | `FeatureCompressor` | Conv + FC | ~260,000 | FC is large (64*62*64) |
| | `VQC` | 8 Qubits | < 1,000 | Very efficient |
| **Head** | `Classifier` | Linear | ~8,500 | 128->64->5 |

### **Total Estimated Parameters: ~850,000**
*(Target was < 50,000)*

### Why is it so large?
The standard `Conv1d` in Stages 2 and 3 has very dense connections (`groups=1`).
- **Fix**: Use **Depthwise Separable Convolutions** (groups=in_channels) for Stages 2 & 3, similar to Stage 1.
- This would reduce Stage 2 from ~295k to ~35k.

---

## 3. Training Configuration (From Log/Config)

Based on your logs and `configs/hybrid_v2.yaml`:

- **Batch Size**: 64
- **Accumulation Steps**: 4
- **Effective Batch Size**: 256 (64 * 4)
- **Batches per Epoch**: ~45 (from log `0/45`)
- **Dataset Size**: ~2,880 samples (45 * 64)
  - *Full PTB-XL is ~21,000 samples.*
  - *You are training on ~13% of the data.*

## Summary
1. **Fix Data**: Ensure `strat_fold` column in `ptbxl_database.csv` is correct to stop 100% leakage.
2. **Reduce Model**: Change `src/models/v2/classical_path.py` to use `groups=in_channels` for Stages 2/3 to hit the parameter target.
