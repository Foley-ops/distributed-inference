#!/usr/bin/env python3
"""
Analyze actual split points for each model architecture.
Split points are BETWEEN components where we can divide the model.
"""

import torch
import torch.nn as nn
from core import ModelLoader

def analyze_model_splits(model_type: str):
    """Analyze and count actual split points for a model."""
    # Load model
    model_loader = ModelLoader("./models")
    model = model_loader.load_model(model_type, num_classes=10)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_type.upper()}")
    print(f"{'='*60}")
    
    split_points = 0
    
    if model_type.lower() == 'mobilenetv2':
        # MobileNetV2: count blocks in features
        blocks = list(model.features.children())
        split_points = len(blocks) - 1  # splits between blocks
        print(f"MobileNetV2 has {len(blocks)} blocks")
        print(f"Split points: {split_points} (between blocks)")
        
    elif model_type.lower() == 'resnet18':
        # ResNet18: count BasicBlocks
        total_blocks = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            blocks = list(layer.children())
            total_blocks += len(blocks)
        split_points = total_blocks  # Can split after each block
        print(f"ResNet18 has {total_blocks} BasicBlocks")
        print(f"Split points: {split_points} (after each block)")
        
    elif model_type.lower() == 'vgg16':
        # VGG16: count layers in features
        feature_layers = list(model.features.children())
        print(f"VGG16 has {len(feature_layers)} feature layers")
        
        # Show all layer types
        print("Layer breakdown:")
        conv_count = 0
        relu_count = 0
        pool_count = 0
        for i, layer in enumerate(feature_layers):
            layer_type = type(layer).__name__
            if i < 10:  # Show first 10
                print(f"  Layer {i}: {layer_type}")
            if isinstance(layer, nn.Conv2d):
                conv_count += 1
            elif isinstance(layer, nn.ReLU):
                relu_count += 1
            elif isinstance(layer, nn.MaxPool2d):
                pool_count += 1
        
        print(f"  ... ({len(feature_layers)} total layers)")
        print(f"Layer counts: Conv2d={conv_count}, ReLU={relu_count}, MaxPool2d={pool_count}")
        
        # VGG can be split between ANY layers (except after the last one)
        split_points = len(feature_layers) - 1
        print(f"Split points: {split_points} (between any adjacent layers)")
        
    elif model_type.lower() == 'alexnet':
        # AlexNet: count feature layers
        feature_layers = list(model.features.children())
        split_points = len(feature_layers) - 1  # splits within features
        print(f"AlexNet has {len(feature_layers)} feature layers")
        print(f"Split points: {split_points} (between feature layers)")
        
    elif model_type.lower() == 'inceptionv3':
        # InceptionV3: count all major blocks
        all_blocks = []
        for name, module in model.named_children():
            all_blocks.append(name)
            if len(all_blocks) <= 20:  # Show first 20
                print(f"  Block {len(all_blocks)-1}: {name}")
        
        print(f"Total blocks: {len(all_blocks)}")
        
        # InceptionV3 typically has these major components
        major_components = [name for name, _ in model.named_children() 
                          if not name.startswith('_')]  # Skip private attributes
        split_points = len(major_components) - 1
        print(f"Major components: {len(major_components)}")
        print(f"Split points: {split_points} (between major components)")
        
    elif model_type.lower() == 'squeezenet':
        # SqueezeNet: count Fire modules
        feature_layers = list(model.features.children())
        fire_modules = [l for l in feature_layers if 'Fire' in type(l).__name__]
        split_points = len(fire_modules) - 1
        print(f"SqueezeNet has {len(fire_modules)} Fire modules")
        print(f"Split points: {split_points} (between Fire modules)")
    
    return split_points

def main():
    print("\nModel Split Point Analysis")
    print("Comparing with GAPP results\n")
    
    models = {
        'alexnet': 12,
        'inceptionv3': 19,
        'resnet18': 8,
        'vgg16': 30,
        'squeezenet': None,  # Not mentioned in GAPP
        'mobilenetv2': 19  # Actually 19 blocks, 18 split points
    }
    
    results = {}
    for model_type, gapp_splits in models.items():
        try:
            actual_splits = analyze_model_splits(model_type)
            results[model_type] = actual_splits
        except Exception as e:
            print(f"Error analyzing {model_type}: {e}")
            results[model_type] = None
    
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'GAPP Splits':<15} {'Our Analysis':<15} {'Match?':<10}")
    print("-" * 60)
    
    for model_type, gapp_splits in models.items():
        our_splits = results.get(model_type)
        if gapp_splits and our_splits:
            match = "✓" if abs(gapp_splits - our_splits) <= 1 else "✗"
        else:
            match = "?"
        
        gapp_str = str(gapp_splits) if gapp_splits else "N/A"
        our_str = str(our_splits) if our_splits is not None else "Error"
        
        print(f"{model_type:<15} {gapp_str:<15} {our_str:<15} {match:<10}")

if __name__ == "__main__":
    main()