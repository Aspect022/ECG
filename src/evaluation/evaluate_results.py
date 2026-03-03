import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
import yaml
from scipy import stats
import glob
import matplotlib.pyplot as plt

def load_all_metrics(results_dir: str) -> pd.DataFrame:
    """Recursively loads all `kfold_summary.yaml` files inside the results directory."""
    all_data = []
    
    # We expect results to be in directories like: runs/initial_pipeline_results/ViT_Base/20231010_120000/
    search_path = os.path.join(results_dir, '**', 'kfold_summary.yaml')
    summary_files = glob.glob(search_path, recursive=True)
    
    for sum_file in summary_files:
        path_obj = Path(sum_file)
        # Variant name is typically two levels up from the timestamp folder
        variant_name = path_obj.parent.parent.name
        
        try:
            with open(sum_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Average out fold results into a single row
            row = {
                'Variant': variant_name,
                'K_Folds': data.get('k_folds', 0)
            }
            
            # Aggregate metrics across all folds
            folds = data.get('fold_results', [])
            if not folds:
                continue
                
            metrics_keys = folds[0]['final_val_metrics'].keys()
            for key in metrics_keys:
                values = [f['final_val_metrics'][key] for f in folds if key in f['final_val_metrics']]
                if values:
                    row[f'Mean_{key}'] = np.mean(values)
                    row[f'Std_{key}'] = np.std(values)
                    
            all_data.append(row)
        except Exception as e:
            print(f"Error loading {sum_file}: {e}")
            
    return pd.DataFrame(all_data)

def perform_statistical_tests(df: pd.DataFrame, baseline_name: str = 'ViT_Base'):
    """
    Performs independent t-tests comparing each variant to the baseline.
    Warning: Since we don't have the raw fold metric arrays saved cleanly per row here, 
    we approximate with descriptive stats if needed, or if we saved the folds we can run a true t-test.
    """
    if baseline_name not in df['Variant'].values:
        print(f"Baseline {baseline_name} not found in results. Skipping statistical tests.")
        return None
        
    baseline_row = df[df['Variant'] == baseline_name].iloc[0]
    results = []
    
    for _, row in df.iterrows():
        variant = row['Variant']
        if variant == baseline_name:
            continue
            
        # We simulate a 2-sample t-test from mean and std. 
        # Welch's t-test from descriptive statistics
        n1 = baseline_row.get('K_Folds', 5)
        n2 = row.get('K_Folds', 5)
        
        # We test Accuracy and AUC-ROC primarily
        test_metrics = ['Accuracy', 'AUC-ROC', 'MCC']
        sig_results = {'Variant': variant}
        
        for m in test_metrics:
            mean_col = f'Mean_{m}'
            std_col = f'Std_{m}'
            
            if mean_col in row and mean_col in baseline_row:
                mean1, std1 = baseline_row[mean_col], baseline_row[std_col]
                mean2, std2 = row[mean_col], row[std_col]
                
                # t-statistic from formulas
                # t = (X1 - X2) / sqrt(s1^2/n1 + s2^2/n2)
                var1, var2 = std1**2, std2**2
                denominator = np.sqrt(var1/n1 + var2/n2 + 1e-9)
                t_stat = (mean2 - mean1) / denominator
                
                # degrees of freedom using Welch-Satterthwaite
                df_num = (var1/n1 + var2/n2)**2
                df_den = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) + 1e-9
                dof = df_num / df_den
                
                # two-tailed p-value
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
                
                sig_results[f'{m}_p_value'] = p_val
                sig_results[f'{m}_Sig_0.05'] = p_val < 0.05
                
        results.append(sig_results)
        
    return pd.DataFrame(results)

def plot_bar_chart(df: pd.DataFrame, metric: str, output_dir: str):
    """Plots a bar chart with error bars for a given metric across variants."""
    mean_col = f'Mean_{metric}'
    std_col = f'Std_{metric}'
    
    if mean_col not in df.columns:
        return
        
    df_sorted = df.sort_values(by=mean_col, ascending=False)
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(df_sorted))
    plt.bar(x_pos, df_sorted[mean_col], yerr=df_sorted.get(std_col), 
            align='center', alpha=0.8, ecolor='black', capsize=10, color='skyblue')
    
    plt.xticks(x_pos, df_sorted['Variant'], rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f'{metric} Comparison across Pipeline Variants')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate and aggregate Initial Pipeline Results")
    parser.add_argument('--results_dir', type=str, default='runs/initial_pipeline_results', help='Path to pipeline results')
    args = parser.parse_args()
    
    out_dir = Path(args.results_dir) / 'final_report'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading metrics...")
    df = load_all_metrics(args.results_dir)
    if df.empty:
        print("No valid results found. Did the pipeline complete?")
        return
        
    df.to_csv(out_dir / 'aggregated_metrics.csv', index=False)
    print(f"Saved aggregated metrics to {out_dir / 'aggregated_metrics.csv'}")
    
    # Identify best models
    if 'Mean_Accuracy' in df.columns:
        best_acc_row = df.loc[df['Mean_Accuracy'].idxmax()]
        print(f"\nBest Model by Accuracy: {best_acc_row['Variant']} ({best_acc_row['Mean_Accuracy']:.4f} \u00B1 {best_acc_row['Std_Accuracy']:.4f})")
    if 'Mean_AUC-ROC' in df.columns:    
        best_auc_row = df.loc[df['Mean_AUC-ROC'].idxmax()]
        print(f"Best Model by AUC-ROC:  {best_auc_row['Variant']} ({best_auc_row['Mean_AUC-ROC']:.4f} \u00B1 {best_auc_row['Std_AUC-ROC']:.4f})")
        
    print("\nRunning Statistical T-Tests (Welch's vs ViT_Base)...")
    stats_df = perform_statistical_tests(df, baseline_name='ViT_Base')
    if stats_df is not None and not stats_df.empty:
        stats_df.to_csv(out_dir / 'statistical_significance.csv', index=False)
        print("Significant Improvements found (p < 0.05):")
        # List which ones actually beat the baseline significantly
        for _, row in stats_df.iterrows():
            sig_flags = [k for k, v in row.items() if 'Sig_0.05' in k and v == True]
            if sig_flags:
                print(f"  - {row['Variant']} in {', '.join([f.replace('_Sig_0.05', '') for f in sig_flags])}")

    # Plot comparisons
    plot_bar_chart(df, 'Accuracy', str(out_dir))
    plot_bar_chart(df, 'AUC-ROC', str(out_dir))
    plot_bar_chart(df, 'MCC', str(out_dir))
    plot_bar_chart(df, 'latency_ms', str(out_dir))
    plot_bar_chart(df, 'energy_mj', str(out_dir))
    
    print("\nEvaluation report completed successfully.")

if __name__ == "__main__":
    main()
