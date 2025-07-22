"""Configuration classes for distributed inference."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class NetworkConfig:
    """Network configuration parameters."""
    communication_latency_ms: float = 200.0
    network_bandwidth_mbps: float = 3.5
    rpc_timeout: int = 3600
    num_worker_threads: int = 4


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_type: str
    num_classes: int = 10
    models_dir: str = "./models"
    shards_dir: str = "./model_shards"
    split_block: Optional[int] = None


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    batch_size: int = 8
    num_test_samples: int = 64
    dataset: str = "cifar10"
    enable_prefetch: bool = False
    prefetch_batches: int = 2


@dataclass
class DistributedConfig:
    """Complete configuration for distributed inference."""
    # Core settings
    rank: int
    world_size: int
    num_splits: int
    workers: List[str] = field(default_factory=list)
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # Feature flags
    use_intelligent_splitting: bool = True
    use_pipelining: bool = False
    use_local_loading: bool = True
    
    # Paths
    metrics_dir: str = "./enhanced_metrics"
    
    @property
    def master_addr(self) -> str:
        """Get master address from environment or default."""
        import os
        return os.getenv('MASTER_ADDR', 'localhost')
    
    @property
    def master_port(self) -> str:
        """Get master port from environment or default."""
        import os
        return os.getenv('MASTER_PORT', '29555')
    
    @property
    def is_master(self) -> bool:
        """Check if this is the master node."""
        return self.rank == 0
    
    def validate(self):
        """Validate configuration."""
        if self.world_size > 1 and self.num_splits > self.world_size - 1:
            raise ValueError(
                f"Number of splits ({self.num_splits}) cannot exceed "
                f"number of workers ({self.world_size - 1})"
            )