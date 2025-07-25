#!/usr/bin/env python3
"""
Test script to verify model splits locally before distributed inference.
This helps catch dimension mismatches and architectural issues early.
"""

import torch
import torch.nn as nn
import json
import os
import sys
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_shard(shard_path: str) -> nn.Module:
    """Load a shard from file."""
    checkpoint = torch.load(shard_path, map_location='cpu')
    return checkpoint['model']


def test_shard_connection(shard1: nn.Module, shard2: nn.Module, 
                         input_shape: Tuple[int, ...], model_name: str) -> bool:
    """Test if two shards connect properly."""
    try:
        # Create random input
        x = torch.randn(*input_shape)
        logger.info(f"Testing {model_name} with input shape: {x.shape}")
        
        # Forward through first shard
        with torch.no_grad():
            output1 = shard1(x)
        logger.info(f"Shard 1 output shape: {output1.shape}")
        
        # Forward through second shard
        with torch.no_grad():
            output2 = shard2(output1)
        logger.info(f"Shard 2 output shape: {output2.shape}")
        
        logger.info(f"✓ {model_name} shards connect successfully!")
        return True
        
    except Exception as e:
        logger.error(f"✗ {model_name} shard connection failed: {e}")
        return False


def analyze_split_point(shard_path: str) -> None:
    """Analyze what layers are at the split boundary."""
    checkpoint = torch.load(shard_path, map_location='cpu')
    shard = checkpoint['model']
    
    # Get all modules
    modules = list(shard.named_modules())
    
    # Find last few meaningful layers
    meaningful_layers = []
    for name, module in modules:
        if len(list(module.children())) == 0 and name:  # Leaf modules
            meaningful_layers.append((name, module.__class__.__name__))
    
    if meaningful_layers:
        logger.info("Last 5 layers in shard:")
        for name, class_name in meaningful_layers[-5:]:
            logger.info(f"  {name}: {class_name}")


def test_model_splits(model_type: str, split_dir: str) -> bool:
    """Test splits for a specific model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_type.upper()} splits")
    logger.info(f"{'='*60}")
    
    # Check if metadata exists
    metadata_path = os.path.join(split_dir, f"{model_type}_shards_metadata.json")
    if not os.path.exists(metadata_path):
        logger.error(f"Metadata not found: {metadata_path}")
        return False
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    logger.info(f"Model: {metadata['model_type']}")
    logger.info(f"Shards: {metadata['num_shards']}")
    logger.info(f"Split type: {metadata.get('split_type', 'unknown')}")
    
    # Load shards
    shards = []
    for shard_info in metadata['shards']:
        shard_path = os.path.join(split_dir, shard_info['filename'])
        if not os.path.exists(shard_path):
            logger.error(f"Shard file not found: {shard_path}")
            return False
        
        logger.info(f"\nLoading shard {shard_info['shard_id']}...")
        shard = load_shard(shard_path)
        shards.append(shard)
        
        # Analyze split point for first shard
        if shard_info['shard_id'] == 0:
            analyze_split_point(shard_path)
    
    # Define input shapes for each model
    input_shapes = {
        'mobilenetv2': (1, 3, 224, 224),
        'resnet18': (1, 3, 224, 224),
        'resnet50': (1, 3, 224, 224),
        'vgg16': (1, 3, 224, 224),
        'alexnet': (1, 3, 224, 224),
        'inceptionv3': (1, 3, 299, 299),  # InceptionV3 needs 299x299
    }
    
    input_shape = input_shapes.get(model_type.lower(), (1, 3, 224, 224))
    
    # Test shard connections
    logger.info(f"\nTesting shard connections...")
    if len(shards) == 2:
        return test_shard_connection(shards[0], shards[1], input_shape, model_type)
    else:
        logger.warning(f"Expected 2 shards, found {len(shards)}")
        return False


def main():
    if len(sys.argv) > 1:
        models = sys.argv[1:]
    else:
        models = ['resnet18', 'resnet50', 'vgg16', 'alexnet', 'inceptionv3']
    
    base_dir = os.path.expanduser("~/datasets/model_shards/split_0")
    
    results = {}
    for model in models:
        success = test_model_splits(model, base_dir)
        results[model] = success
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    for model, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{model.upper()}: {status}")


if __name__ == "__main__":
    main()