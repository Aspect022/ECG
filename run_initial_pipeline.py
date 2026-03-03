import subprocess
import yaml
from pathlib import Path
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Initial Pipeline for ViT-SNN-Quantum")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train each variant')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (faster, fewer epochs, small batch size)')
    
    args = parser.parse_args()

    print("="*60)
    print("Starting Initial SNN & Quantum Variant Evaluation Pipeline")
    print("="*60)

    base_config_path = Path("configs/hybrid_v2/default.yaml")
    
    # Check if base config exists
    if not base_config_path.exists():
        print(f"Error: Base config not found at {base_config_path}")
        return

    # Load base config
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    epochs = args.epochs
    if args.debug:
        print("DEBUG MODE ENABLED")
        epochs = 2
        config['data']['batch_size'] = 16
        config['experiment']['num_workers'] = 0

    config['training']['epochs'] = epochs

    # Create temporary config directory
    temp_conf_dir = Path("configs/temp_pipeline")
    temp_conf_dir.mkdir(parents=True, exist_ok=True)

    variant_configs = []

    # 1. Base ViT Model
    vit_base = dict(config)
    vit_base['model']['components'] = ['classical']
    variant_configs.append(('ViT_Base', vit_base))

    # 2. SNN Variants
    for neuron_type in ['LIF', 'QLIF', 'ExpLIF']:
        snn_conf = dict(config)
        snn_conf['model']['components'] = ['classical', 'snn']
        snn_conf['model']['snn']['neuron_type'] = neuron_type
        variant_configs.append((f'ViT_SNN_{neuron_type}', snn_conf))

    # 3. Quantum Variants (with combinations of Rotations and Backends)
    backends = ['local', 'pennylane', 'qiskit']
    rotations = ['rx', 'ry', 'rz', 'rxy', 'rxz', 'ryz', 'rxyz'] # Simplified from list: 'xy', 'xyz'

    # Limit search space if debug
    if args.debug:
        backends = ['local']
        rotations = ['rx', 'rxy']

    for backend in backends:
        for rot in rotations:
            q_conf = dict(config)
            q_conf['model']['components'] = ['classical', 'quantum']
            q_conf['model']['quantum']['backend'] = backend
            q_conf['model']['quantum']['rotation_axes'] = rot
            q_conf['model']['quantum']['entanglement'] = 'none' # No Entanglement for initial search
            
            variant_configs.append((f'ViT_QC_{backend}_Rot_{rot}_NoEnt', q_conf))
            
            # Add Circular Entanglement for a subset
            if rot in ['ry', 'rxy', 'rxyz']:
                q_conf_ent = dict(q_conf)
                q_conf_ent['model']['quantum']['entanglement'] = 'circular'
                variant_configs.append((f'ViT_QC_{backend}_Rot_{rot}_CircEnt', q_conf_ent))


    print(f"\nGenerated {len(variant_configs)} variant configurations.")
    
    # Run the pipeline
    for name, v_config in variant_configs:
        print(f"\n[{name}] {'-'*40}")
        
        # Save temp config
        conf_path = temp_conf_dir / f"{name}.yaml"
        # Update output directory for the variant
        v_config['experiment']['output_dir'] = f"runs/initial_pipeline_results/{name}"
        
        with open(conf_path, 'w') as f:
            yaml.dump(v_config, f)
            
        print(f"Running training for {name} with config {conf_path}...")
        
        # We invoke train_hybrid_v2.py
        # Here we map arguments based on script needs
        cmd = ["python", "train_hybrid_v2.py", "--config", str(conf_path)]
        if args.debug:
            cmd.append("--debug")
            
        try:
            # We use check_call to halt if any inner script throws exception
            subprocess.check_call(cmd)
            print(f"Successfully completed {name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Training failed for {name} with error code {e.returncode}")
            # Optionally continue or halt. Halting is safer.
            print("Halting pipeline.")
            break
            
    print("\nPipeline execution complete.")

if __name__ == "__main__":
    main()
