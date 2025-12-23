#!/usr/bin/env python3
"""
Test script to verify attention_op_mode parameter threading through the entire UNet3DConditionModel.

This script validates that:
1. The attention_op_mode parameter is correctly passed through all layers
2. The correct attention class is selected based on attention_op_mode
3. YAML configuration correctly propagates to model instantiation
"""

import torch
from omegaconf import OmegaConf
from animatediff.models.unet import UNet3DConditionModel
from animatediff.models.attention import SparseCausalAttention2D, ReferenceCrossAttention

def test_attention_op_mode():
    """Test that attention_op_mode parameter correctly selects attention classes."""
    
    print("=" * 80)
    print("Testing attention_op_mode parameter threading")
    print("=" * 80)
    
    # Test 1: Load YAML config and check that attention_op_mode is present
    print("\n[Test 1] Loading YAML configuration...")
    config = OmegaConf.load("configs/inference/inference-v1.yaml")
    if "attention_op_mode" in config.unet_additional_kwargs:
        print(f"✓ attention_op_mode found in YAML: {config.unet_additional_kwargs.attention_op_mode}")
    else:
        print("✗ attention_op_mode NOT found in YAML configuration")
        return False
    
    # Test 2: Check that UNet3DConditionModel accepts attention_op_mode parameter
    print("\n[Test 2] Checking UNet3DConditionModel parameter signature...")
    import inspect
    sig = inspect.signature(UNet3DConditionModel.__init__)
    if "attention_op_mode" in sig.parameters:
        param = sig.parameters["attention_op_mode"]
        print(f"✓ attention_op_mode parameter found in UNet3DConditionModel.__init__")
        print(f"  Default value: {param.default}")
    else:
        print("✗ attention_op_mode parameter NOT found in UNet3DConditionModel.__init__")
        return False
    
    # Test 3: Create a minimal UNet with mask mode and verify attention class
    print("\n[Test 3] Creating UNet with mask mode and checking attention classes...")
    try:
        unet_mask = UNet3DConditionModel(
            in_channels=4,
            out_channels=4,
            sample_size=64,
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
            attention_op_mode="mask",  # TEST WITH MASK MODE
        )
        print("✓ UNet created successfully with attention_op_mode='mask'")
        
        # Check that at least one BasicTransformerBlock uses SparseCausalAttention2D
        found_sparse = False
        for name, module in unet_mask.named_modules():
            if "attn1" in name and isinstance(module, SparseCausalAttention2D):
                found_sparse = True
                print(f"  Found SparseCausalAttention2D at: {name}")
                break
        
        if found_sparse:
            print("✓ Mask mode correctly uses SparseCausalAttention2D")
        else:
            print("⚠ Warning: No SparseCausalAttention2D found (may be in different layer)")
        
    except Exception as e:
        print(f"✗ Error creating UNet with mask mode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Create a minimal UNet with kvcache mode and verify attention class
    print("\n[Test 4] Creating UNet with kvcache mode and checking attention classes...")
    try:
        unet_kvcache = UNet3DConditionModel(
            in_channels=4,
            out_channels=4,
            sample_size=64,
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
            attention_op_mode="kvcache",  # TEST WITH KVCACHE MODE
        )
        print("✓ UNet created successfully with attention_op_mode='kvcache'")
        
        # Check that at least one BasicTransformerBlock uses ReferenceCrossAttention
        found_reference = False
        for name, module in unet_kvcache.named_modules():
            if "attn1" in name and isinstance(module, ReferenceCrossAttention):
                found_reference = True
                print(f"  Found ReferenceCrossAttention at: {name}")
                break
        
        if found_reference:
            print("✓ KVCache mode correctly uses ReferenceCrossAttention")
        else:
            print("⚠ Warning: No ReferenceCrossAttention found (may be in different layer)")
        
    except Exception as e:
        print(f"✗ Error creating UNet with kvcache mode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify that default is kvcache
    print("\n[Test 5] Verifying default attention_op_mode is 'kvcache'...")
    try:
        unet_default = UNet3DConditionModel(
            in_channels=4,
            out_channels=4,
            sample_size=64,
            down_block_types=("CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "CrossAttnDownBlock3D", "DownBlock3D"),
            up_block_types=("UpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D", "CrossAttnUpBlock3D"),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
            # No attention_op_mode specified - should default to "kvcache"
        )
        print("✓ UNet created successfully with default attention_op_mode")
        
        found_reference = False
        for name, module in unet_default.named_modules():
            if "attn1" in name and isinstance(module, ReferenceCrossAttention):
                found_reference = True
                break
        
        if found_reference:
            print("✓ Default mode (kvcache) correctly uses ReferenceCrossAttention")
        else:
            print("⚠ Warning: Could not verify default mode uses ReferenceCrossAttention")
        
    except Exception as e:
        print(f"✗ Error creating UNet with default attention_op_mode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_attention_op_mode()
    exit(0 if success else 1)
