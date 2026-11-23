import sys
import os
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.v2.hybrid_model import create_model

def verify_gradient_flow():
    print("=" * 80)
    print("ECG HYBRID MODEL GRADIENT FLOW AND INTEGRITY VERIFIER")
    print("=" * 80)
    
    # 1. Create Model
    print("\n[Step 1] Initializing Hybrid Model...")
    try:
        model = create_model(num_classes=5, n_qubits=8)
        print("  Model created successfully!")
        print(f"  Total trainable parameters: {model.num_parameters:,}")
    except Exception as e:
        print(f"  [FAIL] Failed to create model: {e}")
        return False
        
    # 2. Run Forward Pass
    print("\n[Step 2] Testing Forward Pass...")
    try:
        # Batch of 4, 12 ECG channels, 5000 sequence length
        dummy_input = torch.randn(4, 12, 5000)
        output = model(dummy_input, return_gate=True)
        logits = output['logits']
        reg_loss = output['reg_loss']
        gate_value = output['gate']
        
        print("  Forward pass completed successfully!")
        print(f"  Logits shape:      {logits.shape} (Expected: [4, 5])")
        print(f"  Reg loss value:    {reg_loss.item():.6f}")
        print(f"  Mean Gate value:   {gate_value.mean().item():.4f}")
    except Exception as e:
        print(f"  [FAIL] Forward pass crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 3. Run Backward Pass
    print("\n[Step 3] Testing Backward Pass...")
    try:
        # Zero gradients
        model.zero_grad()
        
        # Loss: sum of logits + regularization loss
        loss = logits.sum() + reg_loss
        print(f"  Computed loss: {loss.item():.4f}. Running backward...")
        loss.backward()
        print("  Backward pass completed successfully!")
    except Exception as e:
        print(f"  [FAIL] Backward pass crashed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # 4. Check Gradient Flow
    print("\n[Step 4] Checking Gradient Flow to Key Modules...")
    
    # Check Stage 1 Convolution (before the first LIF neuron)
    stage1_conv_weight = model.classical_path.stage1.depthwise.weight
    stage1_grad = stage1_conv_weight.grad
    
    # Check Quantum Path Compressor FC (before VQC)
    compressor_fc_weight = model.quantum_path.compressor.fc.weight
    compressor_grad = compressor_fc_weight.grad
    
    # Check VQC parameters
    vqc_param = list(model.quantum_path.circuit.params.values())[0]
    vqc_grad = vqc_param.grad
    
    # Check Fusion gate parameters
    gate_weight = model.fusion.gate_network[0].weight
    gate_grad = gate_weight.grad
    
    success = True
    
    # Verify Stage 1 gradients
    if stage1_grad is None:
        print("  [FAIL] Stage 1 depthwise conv weight has NO gradient!")
        success = False
    elif torch.all(stage1_grad == 0):
        print("  [FAIL] Stage 1 depthwise conv gradient is ALL ZEROS (surrogate gradient blocked)!")
        success = False
    else:
        print(f"  [PASS] Stage 1 conv gradient flowing. Mean abs grad: {stage1_grad.abs().mean().item():.8f}")
        
    # Verify Quantum Compressor gradients
    if compressor_grad is None:
        print("  [FAIL] Quantum Feature Compressor FC weight has NO gradient!")
        success = False
    elif torch.all(compressor_grad == 0):
        print("  [FAIL] Quantum Feature Compressor FC gradient is ALL ZEROS!")
        success = False
    else:
        print(f"  [PASS] Feature Compressor FC gradient flowing. Mean abs grad: {compressor_grad.abs().mean().item():.8f}")
        
    # Verify VQC gradients
    if vqc_grad is None:
        print("  [FAIL] Vectorized VQC parameters have NO gradient!")
        success = False
    elif torch.all(vqc_grad == 0):
        print("  [FAIL] Vectorized VQC gradient is ALL ZEROS!")
        success = False
    else:
        print(f"  [PASS] VQC gradient flowing. Mean abs grad: {vqc_grad.abs().mean().item():.8f}")
        
    # Verify Gated Fusion gradients
    if gate_grad is None:
        print("  [FAIL] Gated Fusion gate weight has NO gradient!")
        success = False
    elif torch.all(gate_grad == 0):
        print("  [FAIL] Gated Fusion gate gradient is ALL ZEROS!")
        success = False
    else:
        print(f"  [PASS] Gated Fusion gate gradient flowing. Mean abs grad: {gate_grad.abs().mean().item():.8f}")
        
    print("\n" + "=" * 80)
    if success:
        print("VERIFICATION SUCCESSFUL: ALL CRASHES FIXED & GRADIENT FLOW FULLY RESTORED! SUCCESS!")
    else:
        print("VERIFICATION FAILED: Gradient flow is blocked in one or more layers.")
    print("=" * 80)
    return success

if __name__ == "__main__":
    verify_gradient_flow()
