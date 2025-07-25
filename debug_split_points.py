#!/usr/bin/env python3
"""Debug script to see where the splitter is choosing split points."""

import sys
sys.path.append('.')

from core import ModelLoader
from profiling import LayerProfiler, IntelligentSplitter

def debug_split_points(model_type: str):
    print(f"\nDEBUGGING {model_type.upper()} SPLIT POINTS")
    print("="*60)
    
    # Load model
    model_loader = ModelLoader("./models")
    model = model_loader.load_model(model_type, num_classes=10)
    
    # Profile
    sample_input = model_loader.get_sample_input(model_type, batch_size=1)
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=1)
    profile = profiler.profile_model(model, sample_input, model_type)
    
    # Get split points
    splitter = IntelligentSplitter()
    split_points = splitter._generate_split_points(profile)
    
    print(f"Total layers: {len(profile.layer_profiles)}")
    print(f"Valid split points: {len(split_points)}")
    print("\nSplit points:")
    
    for i, sp in enumerate(split_points[:20]):  # Show first 20
        layer = profile.layer_profiles[sp.layer_index]
        next_layer = profile.layer_profiles[sp.layer_index + 1] if sp.layer_index + 1 < len(profile.layer_profiles) else None
        
        print(f"\n{i}: After layer {sp.layer_index} ({layer.layer_name})")
        if next_layer:
            print(f"   Before layer {sp.layer_index + 1} ({next_layer.layer_name})")
        print(f"   Split cost: {sp.split_cost:.2f}ms")
        
    # Try to find the split
    result = splitter.find_optimal_splits(profile, num_splits=1, method="greedy")
    print(f"\nSelected split indices: {result.split_points[0].layer_index if result.split_points else 'None'}")
    if result.split_points:
        sp = result.split_points[0]
        layer = profile.layer_profiles[sp.layer_index]
        print(f"Split after: {layer.layer_name}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "resnet18"
    debug_split_points(model)