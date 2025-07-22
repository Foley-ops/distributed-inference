"""Shard deployment logic for distributed inference."""

import time
import logging
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
from typing import List, Dict, Any

from shard_wrapper import ShardWrapper
from config import DistributedConfig


class ShardDeployer:
    """Handles deployment of model shards to worker nodes."""
    
    def __init__(self, config: DistributedConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def deploy_shards(
        self, 
        shards: List[Any], 
        shard_configs: List[Dict[str, Any]] = None
    ) -> List[RRef]:
        """
        Deploy shards to worker nodes.
        
        Args:
            shards: List of shard modules (for direct deployment)
            shard_configs: List of shard configurations (for local loading)
            
        Returns:
            List of RRefs to deployed shards
        """
        worker_rrefs = []
        
        if self.config.use_local_loading and shard_configs:
            self.logger.info("Deploying shards with local loading")
            worker_rrefs = self._deploy_with_local_loading(shard_configs)
        else:
            self.logger.info("Deploying shards directly")
            worker_rrefs = self._deploy_direct(shards)
        
        # Verify all shards are loaded
        self._verify_deployment(worker_rrefs)
        
        return worker_rrefs
    
    def _deploy_with_local_loading(self, shard_configs: List[Dict[str, Any]]) -> List[RRef]:
        """Deploy shards using local loading strategy."""
        worker_rrefs = []
        
        for i, config in enumerate(shard_configs):
            worker_name = self.config.workers[i % len(self.config.workers)]
            
            self.logger.info(
                f"Deploying shard {i} to {worker_name} "
                f"(model_type={config.get('model_type')}, "
                f"split_block={config.get('split_block')})"
            )
            
            start_time = time.time()
            
            # Create remote shard with local loading
            rref = rpc.remote(
                worker_name,
                ShardWrapper,
                args=(config, i, None, 'local')
            )
            
            deploy_time = time.time() - start_time
            worker_rrefs.append(rref)
            
            self.logger.info(
                f"Deployed shard {i} to {worker_name} in {deploy_time:.3f}s"
            )
        
        return worker_rrefs
    
    def _deploy_direct(self, shards: List[Any]) -> List[RRef]:
        """Deploy shards directly."""
        worker_rrefs = []
        
        for i, shard in enumerate(shards):
            worker_name = self.config.workers[i % len(self.config.workers)]
            
            self.logger.info(f"Deploying shard {i} object to {worker_name}")
            
            start_time = time.time()
            
            # Create remote shard with direct loading
            rref = rpc.remote(
                worker_name,
                ShardWrapper,
                args=(shard, i, None, 'direct')
            )
            
            deploy_time = time.time() - start_time
            worker_rrefs.append(rref)
            
            self.logger.info(
                f"Deployed shard {i} to {worker_name} in {deploy_time:.3f}s"
            )
        
        return worker_rrefs
    
    def _verify_deployment(self, worker_rrefs: List[RRef]):
        """Verify all shards are properly deployed and loaded."""
        self.logger.info("Verifying shard deployment...")
        
        for i, rref in enumerate(worker_rrefs):
            worker_name = self.config.workers[i % len(self.config.workers)]
            
            try:
                # Check if shard is loaded
                is_ready = rref.rpc_sync().is_shard_loaded()
                
                if is_ready:
                    self.logger.info(f"Worker {worker_name} confirmed shard {i} is loaded")
                else:
                    raise RuntimeError(f"Worker {worker_name} failed to load shard {i}")
                    
            except Exception as e:
                self.logger.error(f"Failed to verify shard {i} on {worker_name}: {e}")
                raise
        
        self.logger.info("All shards verified successfully")