#!/usr/bin/env python3
"""
Analyze different PyTorch model architectures to understand their block structures.
This script examines how models like ResNet, VGG, AlexNet are organized into blocks/stages.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict, Any, Tuple
import json
from collections import OrderedDict


class ModelArchitectureAnalyzer:
    """Analyze the architecture of different PyTorch models."""
    
    def __init__(self):
        self.models_to_analyze = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16': models.vgg16,
            'vgg19': models.vgg19,
            'alexnet': models.alexnet,
            'mobilenet_v2': models.mobilenet_v2,
            'squeezenet1_0': models.squeezenet1_0,
            'densenet121': models.densenet121,
            'inception_v3': models.inception_v3,
            'googlenet': models.googlenet,
            'efficientnet_b0': models.efficientnet_b0,
        }
    
    def analyze_model(self, model_name: str) -> Dict[str, Any]:
        """Analyze a specific model's architecture."""
        if model_name not in self.models_to_analyze:
            raise ValueError(f"Model {model_name} not supported")
        
        # Create model instance
        model = self.models_to_analyze[model_name](weights=None)
        
        analysis = {
            'model_name': model_name,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'top_level_modules': {},
            'has_sequential_structure': False,
            'potential_split_points': [],
            'architecture_pattern': 'unknown'
        }
        
        # Analyze top-level structure
        for name, module in model.named_children():
            module_info = self._analyze_module(module, name)
            analysis['top_level_modules'][name] = module_info
        
        # Detect architecture patterns
        analysis['architecture_pattern'] = self._detect_pattern(model, model_name)
        
        # Find natural split points
        analysis['potential_split_points'] = self._find_split_points(model, model_name)
        
        return analysis
    
    def _analyze_module(self, module: nn.Module, module_name: str) -> Dict[str, Any]:
        """Analyze a specific module."""
        info = {
            'type': module.__class__.__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
            'is_sequential': isinstance(module, nn.Sequential),
            'num_children': len(list(module.children())),
            'children_types': []
        }
        
        # Get types of immediate children
        for child in module.children():
            child_type = child.__class__.__name__
            if child_type not in info['children_types']:
                info['children_types'].append(child_type)
        
        # Special handling for Sequential modules
        if isinstance(module, nn.Sequential):
            info['num_blocks'] = len(module)
            info['block_types'] = [child.__class__.__name__ for child in module.children()]
        
        return info
    
    def _detect_pattern(self, model: nn.Module, model_name: str) -> str:
        """Detect the architectural pattern of the model."""
        
        # ResNet pattern: conv1, bn1, relu, maxpool, layer1-4, avgpool, fc
        if 'resnet' in model_name:
            if all(hasattr(model, attr) for attr in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']):
                return 'resnet_style'
        
        # VGG pattern: features (Sequential), avgpool, classifier (Sequential)
        elif 'vgg' in model_name:
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                if isinstance(model.features, nn.Sequential) and isinstance(model.classifier, nn.Sequential):
                    return 'vgg_style'
        
        # AlexNet pattern: similar to VGG
        elif 'alexnet' in model_name:
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                return 'alexnet_style'
        
        # MobileNet pattern: features (Sequential), classifier
        elif 'mobilenet' in model_name:
            if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                return 'mobilenet_style'
        
        # DenseNet pattern: features (Sequential), classifier
        elif 'densenet' in model_name:
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                return 'densenet_style'
        
        # EfficientNet pattern: features (Sequential), avgpool, classifier
        elif 'efficientnet' in model_name:
            if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                return 'efficientnet_style'
        
        # Inception pattern: complex nested structure
        elif 'inception' in model_name or 'googlenet' in model_name:
            return 'inception_style'
        
        return 'unknown'
    
    def _find_split_points(self, model: nn.Module, model_name: str) -> List[Dict[str, Any]]:
        """Find natural split points in the model."""
        split_points = []
        
        # ResNet-style models
        if 'resnet' in model_name:
            # Natural boundaries are between layer1, layer2, layer3, layer4
            layers = ['layer1', 'layer2', 'layer3', 'layer4']
            for i, layer_name in enumerate(layers[:-1]):
                if hasattr(model, layer_name):
                    split_points.append({
                        'after': layer_name,
                        'before': layers[i+1],
                        'type': 'between_residual_stages'
                    })
        
        # VGG/AlexNet-style models
        elif any(name in model_name for name in ['vgg', 'alexnet']):
            if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                features = model.features
                
                # Find transitions between conv blocks and pooling layers
                for i, module in enumerate(features):
                    if isinstance(module, nn.MaxPool2d) and i < len(features) - 1:
                        split_points.append({
                            'after_index': i,
                            'after_module': module.__class__.__name__,
                            'in_sequential': 'features',
                            'type': 'after_pooling'
                        })
                
                # Split between features and classifier
                split_points.append({
                    'after': 'features',
                    'before': 'classifier',
                    'type': 'features_to_classifier'
                })
        
        # MobileNet-style models
        elif 'mobilenet' in model_name:
            if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                features = model.features
                
                # Find InvertedResidual blocks
                for i, module in enumerate(features):
                    module_name = module.__class__.__name__
                    if 'InvertedResidual' in module_name or 'ConvBNActivation' in module_name:
                        # Split after every few blocks
                        if i > 0 and i % 4 == 0 and i < len(features) - 1:
                            split_points.append({
                                'after_index': i,
                                'in_sequential': 'features',
                                'type': 'between_blocks'
                            })
        
        # DenseNet-style models
        elif 'densenet' in model_name:
            if hasattr(model, 'features'):
                # DenseNet has denseblock and transition modules
                for name, module in model.features.named_children():
                    if 'transition' in name:
                        split_points.append({
                            'after': name,
                            'in_features': True,
                            'type': 'after_transition'
                        })
        
        # EfficientNet-style models
        elif 'efficientnet' in model_name:
            if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
                # Similar to MobileNet, split between blocks
                features = model.features
                for i in range(0, len(features), 3):
                    if i > 0 and i < len(features) - 1:
                        split_points.append({
                            'after_index': i,
                            'in_sequential': 'features',
                            'type': 'between_blocks'
                        })
        
        return split_points
    
    def analyze_all_models(self) -> Dict[str, Any]:
        """Analyze all supported models."""
        results = {}
        
        for model_name in self.models_to_analyze.keys():
            try:
                print(f"\nAnalyzing {model_name}...")
                analysis = self.analyze_model(model_name)
                results[model_name] = analysis
                
                # Print summary
                print(f"  Architecture pattern: {analysis['architecture_pattern']}")
                print(f"  Top-level modules: {list(analysis['top_level_modules'].keys())}")
                print(f"  Number of potential split points: {len(analysis['potential_split_points'])}")
                
            except Exception as e:
                print(f"  Error analyzing {model_name}: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def print_detailed_structure(self, model_name: str):
        """Print detailed structure of a specific model."""
        if model_name not in self.models_to_analyze:
            raise ValueError(f"Model {model_name} not supported")
        
        model = self.models_to_analyze[model_name](weights=None)
        
        print(f"\n{'='*60}")
        print(f"Detailed Structure of {model_name}")
        print(f"{'='*60}")
        
        # Print top-level modules
        print("\nTop-level modules:")
        for name, module in model.named_children():
            print(f"\n{name}: {module.__class__.__name__}")
            
            # If it's a Sequential, print its contents
            if isinstance(module, nn.Sequential):
                print(f"  Sequential with {len(module)} sub-modules:")
                for i, submodule in enumerate(module):
                    print(f"    [{i}] {submodule.__class__.__name__}", end="")
                    
                    # Print some details for specific layer types
                    if isinstance(submodule, nn.Conv2d):
                        print(f" (in={submodule.in_channels}, out={submodule.out_channels}, kernel={submodule.kernel_size})")
                    elif isinstance(submodule, nn.Linear):
                        print(f" (in={submodule.in_features}, out={submodule.out_features})")
                    else:
                        print()
            
            # For ResNet layers
            elif hasattr(module, '__len__') and not isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                try:
                    print(f"  Contains {len(module)} blocks")
                except:
                    pass


def main():
    analyzer = ModelArchitectureAnalyzer()
    
    # Analyze all models
    print("Analyzing all supported models...")
    results = analyzer.analyze_all_models()
    
    # Save results to JSON
    with open('model_architecture_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to model_architecture_analysis.json")
    
    # Print detailed structure for specific models
    print("\n" + "="*80)
    print("DETAILED STRUCTURES")
    print("="*80)
    
    for model_name in ['resnet18', 'vgg16', 'alexnet', 'mobilenet_v2']:
        try:
            analyzer.print_detailed_structure(model_name)
        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    main()