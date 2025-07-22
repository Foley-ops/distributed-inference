"""Model splitting strategies for distributed inference."""

import torch
import torch.nn as nn
import logging
import os
import json
from typing import List, Tuple, Dict, Any, Optional

from profiling import LayerProfiler, split_model_intelligently
from core import ModelLoader
from config import ModelConfig, NetworkConfig


class ModelSplitter:
    """Handles various model splitting strategies."""
    
    def __init__(self, model_config: ModelConfig, network_config: NetworkConfig):
        self.model_config = model_config
        self.network_config = network_config
        self.logger = logging.getLogger(__name__)
        self.model_loader = ModelLoader(model_config.models_dir)
    
    def split_model(
        self, 
        model: nn.Module, 
        num_splits: int,
        use_intelligent_splitting: bool = True
    ) -> Tuple[List[nn.Module], Optional[Dict[str, Any]]]:
        """
        Split model using the appropriate strategy.
        
        Returns:
            Tuple of (shards, split_config)
        """
        split_config = None
        
        if use_intelligent_splitting:
            # Try block-level splitting first
            if hasattr(model, 'features') and hasattr(model, 'classifier'):
                self.logger.info("Using block-level splitting")
                shards = self._split_block_level(model, num_splits)
            else:
                self.logger.info("Using intelligent profiling-based splitting")
                shards, split_config = self._split_intelligent(model, num_splits)
        else:
            self.logger.info("Using simple sequential splitting")
            shards = self._split_sequential(model, num_splits)
        
        self._log_split_info(shards)
        return shards, split_config
    
    def _split_intelligent(
        self, 
        model: nn.Module, 
        num_splits: int
    ) -> Tuple[List[nn.Module], Dict[str, Any]]:
        """Split using profiling data."""
        # Profile the model
        self.logger.info(f"Profiling model: {self.model_config.model_type}")
        sample_input = self.model_loader.get_sample_input(
            self.model_config.model_type, 
            batch_size=1
        )
        
        profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
        model_profile = profiler.profile_model(
            model, 
            sample_input, 
            self.model_config.model_type
        )
        
        # Save profile
        os.makedirs("./profiles", exist_ok=True)
        profile_path = f"./profiles/{self.model_config.model_type}_profile.json"
        model_profile.save_to_file(profile_path)
        
        # Split intelligently
        network_params = {
            'communication_latency_ms': self.network_config.communication_latency_ms,
            'network_bandwidth_mbps': self.network_config.network_bandwidth_mbps
        }
        
        shards, split_config = split_model_intelligently(
            model, 
            model_profile, 
            num_splits,
            network_config=network_params
        )
        
        return shards, split_config
    
    def _split_block_level(self, model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split at the block level (for models with features/classifier structure)."""
        feature_blocks = list(model.features.children())
        total_blocks = len(feature_blocks)
        
        # Determine split point
        if self.model_config.split_block is not None:
            split_point = self.model_config.split_block
        elif num_splits == 1:
            # Default split point for 2-way split
            split_point = 8 if self.model_config.model_type.lower() == 'mobilenetv2' else total_blocks // 2
        else:
            split_point = total_blocks // (num_splits + 1)
        
        self.logger.info(f"Splitting at block {split_point}/{total_blocks}")
        
        # Create shards
        shards = []
        
        # Shard 1: First part of features
        shard1 = nn.Sequential(*feature_blocks[:split_point])
        shards.append(shard1)
        
        # Shard 2: Remaining features + pooling + classifier
        shard2_modules = feature_blocks[split_point:]
        shard2_modules.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            model.classifier
        ])
        shard2 = nn.Sequential(*shard2_modules)
        shards.append(shard2)
        
        return shards
    
    def _split_sequential(self, model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Simple sequential splitting."""
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # Handle models with features/classifier structure
            all_modules = list(model.features.children())
            all_modules.extend([
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                model.classifier
            ])
        else:
            # Generic sequential model
            all_modules = list(model.children())
        
        # Calculate split points
        total_modules = len(all_modules)
        modules_per_shard = total_modules // (num_splits + 1)
        
        shards = []
        for i in range(num_splits + 1):
            start_idx = i * modules_per_shard
            end_idx = (i + 1) * modules_per_shard if i < num_splits else total_modules
            
            shard_modules = all_modules[start_idx:end_idx]
            if shard_modules:
                shards.append(nn.Sequential(*shard_modules))
        
        return shards
    
    def _log_split_info(self, shards: List[nn.Module]):
        """Log information about the split."""
        total_params = sum(sum(p.numel() for p in shard.parameters()) for shard in shards)
        
        self.logger.info(f"Created {len(shards)} shards")
        for i, shard in enumerate(shards):
            shard_params = sum(p.numel() for p in shard.parameters())
            percentage = (shard_params / total_params * 100) if total_params > 0 else 0
            self.logger.info(
                f"Shard {i}: {shard_params:,} parameters ({percentage:.1f}%)"
            )
    
    def create_shard_configs(
        self, 
        shards: List[nn.Module], 
        split_block: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Create configuration for each shard for local loading."""
        shard_configs = []
        
        # Check for pre-split metadata
        metadata_path = self._get_metadata_path(split_block)
        
        if os.path.exists(metadata_path):
            # Load from metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            for shard_info in metadata['shards']:
                config = {
                    'model_type': self.model_config.model_type,
                    'models_dir': self.model_config.models_dir,
                    'shards_dir': self.model_config.shards_dir,
                    'num_classes': self.model_config.num_classes,
                    'shard_id': shard_info['shard_id'],
                    'total_shards': metadata['num_shards'],
                    'split_block': split_block
                }
                shard_configs.append(config)
        else:
            # Create generic configs
            for i in range(len(shards)):
                config = {
                    'model_type': self.model_config.model_type,
                    'models_dir': self.model_config.models_dir,
                    'shards_dir': self.model_config.shards_dir,
                    'num_classes': self.model_config.num_classes,
                    'shard_id': i,
                    'total_shards': len(shards),
                    'split_block': split_block
                }
                shard_configs.append(config)
        
        return shard_configs
    
    def _get_metadata_path(self, split_block: Optional[int]) -> str:
        """Get path to shard metadata file."""
        if split_block is not None:
            split_dir = os.path.join(self.model_config.shards_dir, f"split_{split_block}")
            return os.path.join(split_dir, f"{self.model_config.model_type}_shards_metadata.json")
        else:
            return os.path.join(
                self.model_config.shards_dir, 
                f"{self.model_config.model_type}_shards_metadata.json"
            )