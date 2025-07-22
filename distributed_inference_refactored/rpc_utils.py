"""RPC utilities and setup functions."""

import os
import socket
import logging
import time
import torch.distributed.rpc as rpc
from typing import Dict, Any

from config import DistributedConfig
from metrics import EnhancedMetricsCollector


# Global metrics collector for RPC access
global_metrics_collector = None


def setup_rpc_environment(config: DistributedConfig):
    """Setup RPC environment variables."""
    os.environ['MASTER_ADDR'] = config.master_addr
    os.environ['MASTER_PORT'] = config.master_port
    
    # Use appropriate network interface
    if config.is_master:
        gloo_ifname = os.getenv('GLOO_SOCKET_IFNAME', 'eth0')
        os.environ['TENSORPIPE_SOCKET_IFADDR'] = '0.0.0.0'
    else:
        gloo_ifname = os.getenv('GLOO_SOCKET_IFNAME', 'eth0')
    
    os.environ['GLOO_SOCKET_IFNAME'] = gloo_ifname


def init_rpc(config: DistributedConfig) -> bool:
    """Initialize RPC framework."""
    logger = logging.getLogger(__name__)
    
    node_name = "master" if config.is_master else f"worker{config.rank}"
    
    try:
        logger.info(f"Initializing RPC for {node_name}")
        
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=config.network.num_worker_threads,
            rpc_timeout=config.network.rpc_timeout,
            init_method=f"tcp://{config.master_addr}:{config.master_port}"
        )
        
        rpc.init_rpc(
            node_name,
            rank=config.rank,
            world_size=config.world_size,
            rpc_backend_options=rpc_backend_options
        )
        
        logger.info(f"RPC initialized successfully for {node_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RPC: {e}")
        return False


def shutdown_rpc():
    """Shutdown RPC framework."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Shutting down RPC")
        start_time = time.time()
        rpc.shutdown()
        shutdown_time = time.time() - start_time
        logger.info(f"RPC shutdown completed in {shutdown_time:.2f}s")
    except Exception as e:
        logger.error(f"Error during RPC shutdown: {e}")


def collect_worker_summary(model_name: str, batch_size: int, num_parameters: int = 0) -> Dict[str, Any]:
    """RPC function to collect summary from workers."""
    global global_metrics_collector
    if global_metrics_collector:
        return global_metrics_collector.get_device_summary()
    return {}


def register_metrics_collector(metrics_collector: EnhancedMetricsCollector):
    """Register metrics collector for RPC access."""
    global global_metrics_collector
    global_metrics_collector = metrics_collector


def wait_for_workers(config: DistributedConfig, max_retries: int = 30) -> bool:
    """Worker nodes wait for master connection."""
    logger = logging.getLogger(__name__)
    
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            logger.info(f"Connection attempt {retry_count + 1}/{max_retries}")
            
            if init_rpc(config):
                logger.info("Successfully connected to master")
                return True
                
        except Exception as e:
            retry_count += 1
            logger.warning(f"Connection attempt {retry_count} failed: {e}")
            
            if retry_count >= max_retries:
                logger.error(f"Failed to connect after {max_retries} attempts")
                return False
            
            wait_time = 10 + (retry_count % 5)
            logger.info(f"Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return False


def setup_logging(config: DistributedConfig):
    """Setup logging with hostname and rank."""
    hostname = socket.gethostname()
    
    # Create custom log record factory
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.rank = config.rank
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    # Update formatter for all handlers
    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(hostname)s:rank%(rank)s] - %(message)s'
        ))