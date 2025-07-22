"""Unified shard wrapper for distributed model execution."""

import time
import torch
import torch.nn as nn
import socket
import logging
from typing import Optional, Dict, Any, List
import os
import json
from torch.distributed.rpc import RRef

from core import ModelLoader
from metrics import EnhancedMetricsCollector


class ShardWrapper(nn.Module):
    """Unified shard wrapper supporting both direct loading and local file loading."""
    
    def __init__(
        self, 
        shard_or_config: Any, 
        shard_id: int, 
        metrics_collector: Optional[EnhancedMetricsCollector] = None,
        loading_strategy: str = 'direct'
    ):
        """
        Initialize shard wrapper.
        
        Args:
            shard_or_config: Either a nn.Module (direct) or config dict (local loading)
            shard_id: ID of this shard
            metrics_collector: Optional metrics collector
            loading_strategy: 'direct' or 'local'
        """
        super().__init__()
        self.shard_id = shard_id
        self.loading_strategy = loading_strategy
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics collector if not provided
        if metrics_collector is None:
            try:
                import torch.distributed.rpc as rpc
                rank = rpc.get_worker_info().id
            except:
                rank = 0
            self.metrics_collector = EnhancedMetricsCollector(rank)
        else:
            self.metrics_collector = metrics_collector
        
        # Load the module based on strategy
        if loading_strategy == 'direct':
            if not isinstance(shard_or_config, nn.Module):
                raise TypeError(f"Direct loading expects nn.Module, got {type(shard_or_config)}")
            self.module = shard_or_config.to("cpu")
            self.logger.info(f"Shard {shard_id} loaded directly")
        else:  # local loading
            if not isinstance(shard_or_config, dict):
                raise TypeError(f"Local loading expects config dict, got {type(shard_or_config)}")
            self.module = self._load_local_shard(shard_or_config)
            self.logger.info(f"Shard {shard_id} loaded from local files")
    
    def _load_local_shard(self, config: Dict[str, Any]) -> nn.Module:
        """Load shard from pre-split weight files."""
        shards_dir = config.get('shards_dir', './model_shards')
        model_type = config['model_type']
        shard_id = config['shard_id']
        
        # Handle split_block if specified
        split_block = config.get('split_block')
        if split_block is not None:
            shards_dir = os.path.join(shards_dir, f"split_{split_block}")
        
        # Look for pre-split shard file
        shard_filename = f"{model_type}_shard_{shard_id}_of_{config['total_shards']}.pth"
        shard_path = os.path.join(shards_dir, shard_filename)
        
        if os.path.exists(shard_path):
            self.logger.info(f"Loading pre-split shard from {shard_path}")
            checkpoint = torch.load(shard_path, map_location='cpu')
            
            if 'model' in checkpoint:
                return checkpoint['model'].to("cpu")
            else:
                self.logger.error(f"Invalid checkpoint format at {shard_path}")
                raise ValueError("Checkpoint missing 'model' key")
        else:
            self.logger.error(f"Pre-split shard not found at {shard_path}")
            raise FileNotFoundError(f"Shard file not found: {shard_path}")
    
    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with metrics collection."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")
        
        x = x.to("cpu")
        start_time = time.time()
        
        # Forward pass
        with torch.no_grad():
            output = self.module(x).cpu()
        
        end_time = time.time()
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_pipeline_stage(
                batch_id=batch_id if batch_id is not None else 0,
                stage_id=self.shard_id,
                stage_name=f"shard_{self.shard_id}",
                start_time=start_time,
                end_time=end_time,
                input_size_bytes=x.numel() * x.element_size(),
                output_size_bytes=output.numel() * output.element_size()
            )
        
        return output
    
    def is_shard_loaded(self) -> bool:
        """Check if shard is loaded and ready."""
        return self.module is not None
    
    def parameter_rrefs(self):
        """Get parameter RRefs for distributed training (if needed)."""
        return [RRef(p) for p in self.parameters()]