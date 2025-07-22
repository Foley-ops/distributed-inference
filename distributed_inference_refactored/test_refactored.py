#!/usr/bin/env python3
"""Test script for refactored distributed inference."""

import sys
import subprocess
import time


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from config import DistributedConfig, ModelConfig
        from shard_wrapper import ShardWrapper
        from model_splitting import ModelSplitter
        from shard_deployment import ShardDeployer
        from distributed_model import DistributedModel
        from inference_runner import InferenceRunner
        import rpc_utils
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration creation."""
    print("\nTesting configuration...")
    try:
        from config import DistributedConfig, ModelConfig, NetworkConfig, InferenceConfig
        
        config = DistributedConfig(
            rank=0,
            world_size=3,
            num_splits=1,
            workers=["worker1", "worker2"],
            model=ModelConfig(
                model_type="mobilenetv2",
                num_classes=10
            ),
            network=NetworkConfig(),
            inference=InferenceConfig(
                batch_size=8,
                num_test_samples=16
            )
        )
        
        config.validate()
        print("✓ Configuration created and validated")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def compare_with_original():
    """Compare line counts with original."""
    print("\nComparing code size...")
    
    # Count lines in original
    original_lines = 0
    with open('../distributed_runner.py', 'r') as f:
        original_lines = len(f.readlines())
    
    # Count lines in refactored version
    refactored_files = [
        'config.py',
        'shard_wrapper.py', 
        'model_splitting.py',
        'shard_deployment.py',
        'distributed_model.py',
        'inference_runner.py',
        'rpc_utils.py',
        'main.py'
    ]
    
    refactored_lines = 0
    for file in refactored_files:
        try:
            with open(file, 'r') as f:
                lines = len(f.readlines())
                refactored_lines += lines
                print(f"  {file}: {lines} lines")
        except:
            pass
    
    print(f"\nOriginal: {original_lines} lines")
    print(f"Refactored: {refactored_lines} lines")
    print(f"Reduction: {original_lines - refactored_lines} lines ({(1 - refactored_lines/original_lines)*100:.1f}%)")


def main():
    """Run all tests."""
    print("Testing refactored distributed inference...")
    print("=" * 50)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_config():
        all_passed = False
    
    compare_with_original()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()