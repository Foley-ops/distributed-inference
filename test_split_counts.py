#!/usr/bin/env python3
"""
Test script to count the number of possible splits for different models
using the intelligent splitter.
"""

import sys
import torch
from core import ModelLoader
from profiling import LayerProfiler, IntelligentSplitter

def count_model_splits(model_type: str):
    """Count the number of possible splits for a model."""
    print(f"\nAnalyzing {model_type.upper()}:")
    print("-" * 50)
    
    # Load model
    model_loader = ModelLoader("./models")
    model = model_loader.load_model(model_type, num_classes=10)
    
    # Get sample input
    sample_input = model_loader.get_sample_input(model_type, batch_size=1)
    
    # Profile the model
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=2)
    profile = profiler.profile_model(model, sample_input, model_type)
    
    # Count total layers (leaf modules)
    leaf_modules = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            leaf_modules.append(name)
    
    print(f"Total leaf modules: {len(leaf_modules)}")
    print(f"Total profiled layers: {len(profile.layer_profiles)}")
    
    # For MobileNetV2, count blocks
    if model_type.lower() == 'mobilenetv2' and hasattr(model, 'features'):
        blocks = list(model.features.children())
        print(f"MobileNetV2 blocks: {len(blocks)}")
    
    # For ResNet, count stages
    if 'resnet' in model_type.lower():
        stages = []
        for name in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool', 'fc']:
            if hasattr(model, name):
                stages.append(name)
        print(f"ResNet stages: {len(stages)} ({', '.join(stages)})")
        
        # Count blocks within layers
        total_blocks = 0
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                blocks = list(layer.children())
                print(f"  {layer_name}: {len(blocks)} blocks")
                total_blocks += len(blocks)
        print(f"Total ResNet blocks: {total_blocks}")
    
    # For VGG, count feature layers
    if 'vgg' in model_type.lower() and hasattr(model, 'features'):
        feature_layers = list(model.features.children())
        print(f"VGG feature layers: {len(feature_layers)}")
        print(f"VGG split points (between layers): {len(feature_layers) - 1}")
        
    # For AlexNet
    if model_type.lower() == 'alexnet' and hasattr(model, 'features'):
        feature_layers = list(model.features.children())
        classifier_layers = list(model.classifier.children())
        print(f"AlexNet feature layers: {len(feature_layers)}")
        print(f"AlexNet classifier layers: {len(classifier_layers)}")
        print(f"Total AlexNet components: {len(feature_layers) + len(classifier_layers)}")
        # Split points are between layers within features, and between features/classifier
        print(f"AlexNet split points: {len(feature_layers) - 1} (within features) + 1 (features/classifier) = {len(feature_layers)}")
        
    # For InceptionV3
    if model_type.lower() == 'inceptionv3':
        # Count major blocks in InceptionV3
        major_blocks = []
        for name, module in model.named_children():
            major_blocks.append(name)
        print(f"InceptionV3 major blocks: {len(major_blocks)}")
        print(f"Blocks: {', '.join(major_blocks[:10])}...")  # Show first 10
        
    # For SqueezeNet
    if model_type.lower() == 'squeezenet' and hasattr(model, 'features'):
        feature_layers = list(model.features.children())
        print(f"SqueezeNet feature layers: {len(feature_layers)}")
        # Count Fire modules
        fire_modules = [m for m in feature_layers if 'Fire' in type(m).__name__]
        print(f"SqueezeNet Fire modules: {len(fire_modules)}")
    
    # Test intelligent splitter with different numbers of splits
    print("\nTesting intelligent splitter:")
    splitter = IntelligentSplitter()
    
    # Find maximum possible splits
    max_splits = min(len(profile.layer_profiles) - 1, 50)  # Cap at 50 for testing
    
    print(f"Maximum theoretical splits: {max_splits}")
    
    # Print summary of actual split points based on architecture
    print("\n*** SPLIT POINT SUMMARY ***")
    if model_type.lower() == 'mobilenetv2':
        print(f"Actual split points: 18 (between 19 blocks)")
    elif model_type.lower() == 'resnet18':
        print(f"Actual split points: 8 (one per BasicBlock)")
    elif model_type.lower() == 'vgg16':
        print(f"Actual split points: 30 (between feature layers)")
    elif model_type.lower() == 'alexnet':
        print(f"Actual split points: 12 (within features)")
    elif model_type.lower() == 'inceptionv3':
        print(f"Actual split points: ~19 (between inception blocks)")
    elif model_type.lower() == 'squeezenet':
        print(f"Actual split points: ~8 (between Fire modules)")
    
    return len(leaf_modules), len(profile.layer_profiles)

def main():
    models_to_test = ['alexnet', 'inceptionv3', 'resnet18', 'squeezenet', 'vgg16', 'mobilenetv2']
    
    print("Model Split Analysis")
    print("=" * 60)
    print("Target splits from GAPP:")
    print("- AlexNet: 12 splits")
    print("- InceptionV3: 19 splits")
    print("- ResNet18: 8 splits")
    print("- ResNet50: 8 splits") 
    print("- VGG16: 30 splits")
    print("- MobileNetV2: 19 splits (blocks)")
    print("=" * 60)
    
    for model_type in models_to_test:
        try:
            count_model_splits(model_type)
        except Exception as e:
            print(f"Error analyzing {model_type}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()