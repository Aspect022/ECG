import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# Import the necessary training functions
from train_comparison import train_single_model
from train_hybrid_v2 import train as train_hybrid

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_pipeline():
    parser = argparse.ArgumentParser(description="Unified Batch Training Pipeline")
    parser.add_argument('--epochs', type=int, default=None, help="Number of epochs to train each model")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode (fewer epochs, smaller batches)")
    parser.add_argument('--k_folds', type=int, default=5, help="Number of cross-validation folds")
    args = parser.parse_args()

    # Determine epochs
    epochs_input = args.epochs
    if epochs_input is None:
        try:
            val = input("Enter number of epochs to train each model: ")
            epochs_input = int(val.strip())
        except ValueError:
            print("Invalid input. Defaulting to 10 epochs.")
            epochs_input = 10
    
    print(f"\n{'=' * 80}")
    print(f"Starting Unified Training Pipeline for {epochs_input} Epochs")
    print(f"{'=' * 80}\n")

    from torch.utils.tensorboard import SummaryWriter
    
    # 1 & 2. Comparison models (vit_snn, vit_quantum)
    comp_config_path = 'configs/comparison.yaml'
    comp_config = load_config(comp_config_path)
    comp_config['training']['epochs'] = epochs_input

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    comp_output_root = Path(comp_config['experiment']['output_dir']) / timestamp
    comp_output_root.mkdir(parents=True, exist_ok=True)
    comp_log_dir = Path(comp_config['logging']['log_dir']) / timestamp
    comp_writer = SummaryWriter(comp_log_dir)

    print("\n--- [1/3] Training ViT + SNN ---")
    train_single_model(
        config=comp_config,
        model_type='vit_snn',
        output_root=comp_output_root,
        writer=comp_writer,
        debug=args.debug,
        k_folds=args.k_folds,
    )

    print("\n--- [2/3] Training ViT + Quantum ---")
    train_single_model(
        config=comp_config,
        model_type='vit_quantum',
        output_root=comp_output_root,
        writer=comp_writer,
        debug=args.debug,
        k_folds=args.k_folds,
    )
    comp_writer.close()

    # 3. Hybrid V2
    hybrid_config_path = 'configs/hybrid_v2.yaml'
    hybrid_config = load_config(hybrid_config_path)
    hybrid_config['training']['epochs'] = epochs_input

    print("\n--- [3/3] Training Main Hybrid V2 ---")
    train_hybrid(
        config_or_path=hybrid_config,
        debug=args.debug,
        k_folds=args.k_folds
    )

    print(f"\n{'=' * 80}")
    print("Pipeline completed successfully!")
    print(f"{'=' * 80}\n")

if __name__ == "__main__":
    run_pipeline()
