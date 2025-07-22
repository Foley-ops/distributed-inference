"""Simplified distributed model implementation."""

import time
import torch
import torch.nn as nn
import logging
from typing import List, Optional
from torch.distributed.rpc import RRef

from config import DistributedConfig
from model_splitting import ModelSplitter
from shard_deployment import ShardDeployer
from metrics import EnhancedMetricsCollector
from pipelining import PipelineManager
from core import ModelLoader


class DistributedModel(nn.Module):
    """Simplified distributed model with clean separation of concerns."""
    
    def __init__(
        self, 
        config: DistributedConfig,
        metrics_collector: Optional[EnhancedMetricsCollector] = None
    ):
        super().__init__()
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_loader = ModelLoader(config.model.models_dir)
        self.splitter = ModelSplitter(config.model, config.network)
        self.deployer = ShardDeployer(config)
        
        # Load and split model
        self.original_model = self._load_model()
        self.shards, self.split_config = self._split_model()
        
        # Deploy shards
        self.worker_rrefs = self._deploy_shards()
        
        # Setup pipeline if enabled
        self.pipeline_manager = self._setup_pipeline() if config.use_pipelining else None
    
    def _load_model(self) -> nn.Module:
        """Load the model."""
        self.logger.info(f"Loading model: {self.config.model.model_type}")
        return self.model_loader.load_model(
            self.config.model.model_type,
            self.config.model.num_classes
        )
    
    def _split_model(self):
        """Split the model into shards."""
        self.logger.info("Splitting model into shards")
        return self.splitter.split_model(
            self.original_model,
            self.config.num_splits,
            self.config.use_intelligent_splitting
        )
    
    def _deploy_shards(self) -> List[RRef]:
        """Deploy shards to workers."""
        if not self.config.workers:
            self.logger.warning("No workers available - model will run locally")
            return []
        
        # Create shard configs if using local loading
        shard_configs = None
        if self.config.use_local_loading:
            shard_configs = self.splitter.create_shard_configs(
                self.shards,
                self.config.model.split_block
            )
        
        return self.deployer.deploy_shards(self.shards, shard_configs)
    
    def _setup_pipeline(self) -> Optional[PipelineManager]:
        """Setup pipeline manager if pipelining is enabled."""
        if not self.config.use_pipelining:
            return None
        
        self.logger.info("Setting up pipeline manager")
        
        metrics_callback = None
        if self.metrics_collector:
            metrics_callback = self.metrics_collector.record_pipeline_stage
        
        return PipelineManager(
            shards=self.shards,
            workers=self.config.workers,
            metrics_callback=metrics_callback,
            use_local_pipeline=False,
            max_concurrent_batches=4
        )
    
    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the distributed model."""
        if self.config.use_pipelining and self.pipeline_manager:
            return self.pipeline_manager.process_batch_rpc_pipelined(x)
        else:
            return self._forward_sequential(x, batch_id)
    
    def _forward_sequential(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Sequential forward pass (non-pipelined)."""
        current_tensor = x
        
        for i, shard_rref in enumerate(self.worker_rrefs):
            start_time = time.time()
            current_tensor = shard_rref.rpc_sync().forward(current_tensor, batch_id=batch_id)
            rpc_time = (time.time() - start_time) * 1000
            
            # Record metrics
            if self.metrics_collector:
                # Estimate network overhead
                tensor_size_mb = (current_tensor.numel() * current_tensor.element_size()) / (1024 * 1024)
                network_ms = 0.5 + (tensor_size_mb * 0.3) + (tensor_size_mb * 8 / 940) * 1000 * 2
                
                self.metrics_collector.record_network_metrics(rpc_time, network_ms)
        
        return current_tensor
    
    def get_pipeline_stats(self) -> dict:
        """Get pipeline statistics if available."""
        if self.pipeline_manager:
            return self.pipeline_manager.get_pipeline_stats()
        return {}
    
    def parameter_rrefs(self):
        """Get parameter RRefs from all shards."""
        remote_params = []
        for rref in self.worker_rrefs:
            remote_params.extend(rref.remote().parameter_rrefs().to_here())
        return remote_params