# ECG Project: Graph & Asset Compilation Guide

This document lists all the outputs, visual media, and computational results currently existing across the ECG processing project, alongside standard visualizations trivially obtainable for populating figures in the upcoming journal paper. 

## 1. Currently Existing Assets & Raw Material
The project actively generates formatted artifacts resulting from rigorous CV evaluations and training iterations. These exact artifacts reside natively inside `./clean_results/` and `./docs/`.

*   **1. HTML Detailed Result Views** 
    *   **File**: `clean_results/detailed_model_analysis.html` // `clean_results/5_FOLD_CV_REPORT.html`
    *   **Content**: Clean, stylized tabular reports on model characteristics, parameters, 5-Fold evaluations, and empirical derivations matching publication-level styling metrics. 
*   **2. YAML Empirical Logs** 
    *   **File**: `clean_results/comparison_results.yaml`
    *   **Content**: Hard-coded, persistent floating point evaluations documenting strict Energy costs, Latency delays, model parameter counts, Standard Deviations across strictly 5 K-folds for every base model variant evaluated (ViT Base, ViT + SNN, ViT + QNN, Hybrid V2).
*   **3. Base Architecture Logs** 
    *   **File**: `docs/SYSTEM_ARCHITECTURE.md`
    *   **Content**: Complete textual flowchart indicating structural inputs (12x5000), internal routing of parallel stages, temporal stage bounds, block representations, and exact variable equations dictating Gated Fusion implementations.

## 2. Easily Obtainable Visualizations for Journal Inclusion
Since all deterministic raw limits (Accuracies, Latencies, Ensembles, Folds) are natively documented inside `comparison_results.yaml` globally, the following figures can be generated trivially via `matplotlib` or `seaborn` plotting arrays for rapid inclusion. 

### Figure Targets:
1.  **Model Efficiency vs Accuracy Scatter Plot**
    *   **Axes**: X = Energy Consumption (mJ), Y = Test Accuracy (%)
    *   **Purpose**: Visually identifies the pareto-front of efficiency against capability. Demonstrates ViT vs SNN tradeoffs visually, defining the "Green AI" envelope boundary explicitly.
2.  **5-Fold Validation Stability Bar Charts**
    *   **Axes**: X = Training Validation Fold (1-5), Y = Precision metric.
    *   **Purpose**: Illustrate error-bar deviations (standard deviations are < 0.005) to unequivocally prove absence of data leakage and generalized robustness vs standard prior models.
3.  **Architectural Layout & Topology Graph**
    *   **Concept**: Convert the text-flow layout diagram from `SYSTEM_ARCHITECTURE.md` into an abstract vector graphic using LaTeX/Tikz, Microsoft Visio, or Figma. Shows explicit branch isolation (SNN Temporal Pyramids Left, Variational Q-Circuit Right) converging onto explicit sigmoidal operations.
4.  **Quantum Gate Expectation Histogram**
    *   **Concept**: Since the gate operator $g \in [0,1]$ manages quantum delegation, a histogram plotting terminal decision boundaries on average PTB-XL ECGs reveals standard modality preference (e.g., how often does the model completely revert to pure SNN structures by setting $g=0$, validating baseline classical superiority in simple rhythms vs deferring computationally to quantum states).
5.  **Multi-Scale Temporal Sequence Traces**
    *   **Concept**: Print plotting of raw ECG lead voltages against the discrete, quantized binary spike outputs emitted specifically from the Stage 1 LIF layers to illustrate physical neuromorphic encoding of bio-signals.
