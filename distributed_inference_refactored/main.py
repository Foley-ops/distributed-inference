#!/usr/bin/env python3
"""Main entry point for refactored distributed inference."""

import os
os.environ['ATEN_CPU_CAPABILITY'] = ''  # Disable PyTorch CPU optimizations for RPi

import argparse
import logging
import time
import sys
from dotenv import load_dotenv
import torch.distributed.rpc as rpc

from config import DistributedConfig, ModelConfig, NetworkConfig, InferenceConfig
from metrics import EnhancedMetricsCollector
from inference_runner import InferenceRunner
from core import ModelLoader
import rpc_utils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def run_master(config: DistributedConfig):
    """Run master node logic."""
    logger = logging.getLogger(__name__)
    logger.info("Starting master node")
    
    # Initialize metrics collector
    metrics_collector = EnhancedMetricsCollector(
        config.rank, 
        config.metrics_dir, 
        enable_realtime=True
    )
    
    # Register for RPC access
    rpc_utils.register_metrics_collector(metrics_collector)
    
    # Initialize RPC
    if not rpc_utils.init_rpc(config):
        logger.error("Failed to initialize RPC")
        return
    
    try:
        # Create and run inference
        runner = InferenceRunner(config, metrics_collector)
        runner.setup()
        
        logger.info("Starting inference")
        results = runner.run_inference()
        
        # Log results
        logger.info("=" * 50)
        logger.info("INFERENCE RESULTS")
        logger.info("=" * 50)
        logger.info(f"Total images: {results['total_images']}")
        logger.info(f"Total time: {results['total_time']:.2f}s")
        logger.info(f"Accuracy: {results['accuracy']:.2f}%")
        logger.info(f"Throughput: {results['throughput']:.2f} images/sec")
        
        # Collect worker metrics
        logger.info("Collecting worker metrics...")
        worker_summaries = []
        
        for i in range(1, config.world_size):
            worker_name = f"worker{i}"
            try:
                summary = rpc.rpc_sync(
                    worker_name, 
                    rpc_utils.collect_worker_summary,
                    args=(config.model.model_type, config.inference.batch_size, 0)
                )
                if summary:
                    worker_summaries.append(summary)
                    logger.info(f"Collected summary from {worker_name}")
            except Exception as e:
                logger.warning(f"Failed to collect summary from {worker_name}: {e}")
        
        # Merge worker summaries
        for summary in worker_summaries:
            if hasattr(metrics_collector, 'merge_summary'):
                metrics_collector.merge_summary(summary)
        
        # Finalize metrics
        final_metrics = metrics_collector.finalize(config.model.model_type)
        
        logger.info("=" * 50)
        logger.info("FINAL METRICS SUMMARY")
        logger.info("=" * 50)
        
        device_summary = final_metrics['device_summary']
        efficiency_stats = final_metrics['efficiency_stats']
        
        logger.info(f"Images per second: {device_summary.get('images_per_second', 0):.2f}")
        logger.info(f"Average processing time: {device_summary.get('average_processing_time_ms', 0):.2f}ms")
        logger.info(f"Pipeline utilization: {efficiency_stats.get('average_pipeline_utilization', 0):.2f}")
        
    except Exception as e:
        logger.error(f"Error in master node: {e}", exc_info=True)
    finally:
        logger.info("Master work complete")


def run_worker(config: DistributedConfig):
    """Run worker node logic."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting worker node {config.rank}")
    
    # Initialize metrics collector
    metrics_collector = EnhancedMetricsCollector(
        config.rank, 
        config.metrics_dir, 
        enable_realtime=True
    )
    
    # Register for RPC access
    rpc_utils.register_metrics_collector(metrics_collector)
    
    # Wait for master connection
    if not rpc_utils.wait_for_workers(config):
        logger.error("Failed to connect to master")
        sys.exit(1)
    
    logger.info("Worker ready and waiting for tasks...")
    
    # Keep worker alive until RPC shutdown
    try:
        while rpc._is_current_rpc_agent_set():
            time.sleep(1)
    except Exception as e:
        logger.info(f"Worker loop exited: {e}")
    
    logger.info("Worker shutting down")
    
    # Finalize metrics
    metrics_collector.finalize(config.model.model_type)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Refactored Distributed DNN Inference"
    )
    
    # Core arguments
    parser.add_argument("--rank", type=int, default=0, 
                       help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, 
                       help="World size (1 master + N workers)")
    
    # Model arguments
    parser.add_argument("--model", type=str, default="mobilenetv2",
                       choices=ModelLoader.list_supported_models(),
                       help="Model architecture")
    parser.add_argument("--num-classes", type=int, default=10,
                       help="Number of output classes")
    parser.add_argument("--models-dir", type=str, default="./models",
                       help="Directory containing model weight files")
    parser.add_argument("--shards-dir", type=str, default="./model_shards",
                       help="Directory containing pre-split model shards")
    parser.add_argument("--split-block", type=int, default=None,
                       help="Specific block number to split at")
    
    # Inference arguments
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "dummy"],
                       help="Dataset to use")
    parser.add_argument("--num-test-samples", type=int, default=64,
                       help="Number of images to test")
    
    # Distribution arguments
    parser.add_argument("--num-partitions", type=int, default=2,
                       help="Number of model partitions")
    parser.add_argument("--num-threads", type=int, default=4,
                       help="Number of RPC threads")
    
    # Feature flags
    parser.add_argument("--disable-intelligent-splitting", action="store_true",
                       help="Disable intelligent splitting")
    parser.add_argument("--use-pipelining", action="store_true",
                       help="Enable pipelined execution")
    parser.add_argument("--disable-local-loading", action="store_true",
                       help="Disable local loading of shards")
    parser.add_argument("--enable-prefetch", action="store_true",
                       help="Enable data prefetching")
    parser.add_argument("--prefetch-batches", type=int, default=2,
                       help="Number of batches to prefetch")
    
    # Other arguments
    parser.add_argument("--metrics-dir", type=str, default="./enhanced_metrics",
                       help="Directory for metrics")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Create configuration
    config = DistributedConfig(
        rank=args.rank,
        world_size=args.world_size,
        num_splits=args.num_partitions - 1,
        workers=[f"worker{i}" for i in range(1, args.world_size)],
        model=ModelConfig(
            model_type=args.model,
            num_classes=args.num_classes,
            models_dir=args.models_dir,
            shards_dir=args.shards_dir,
            split_block=args.split_block
        ),
        network=NetworkConfig(
            num_worker_threads=args.num_threads
        ),
        inference=InferenceConfig(
            batch_size=args.batch_size,
            num_test_samples=args.num_test_samples,
            dataset=args.dataset,
            enable_prefetch=args.enable_prefetch,
            prefetch_batches=args.prefetch_batches
        ),
        use_intelligent_splitting=not args.disable_intelligent_splitting,
        use_pipelining=args.use_pipelining,
        use_local_loading=not args.disable_local_loading,
        metrics_dir=args.metrics_dir
    )
    
    # Validate configuration
    config.validate()
    
    # Setup environment and logging
    rpc_utils.setup_rpc_environment(config)
    rpc_utils.setup_logging(config)
    
    # Run appropriate node type
    if config.is_master:
        run_master(config)
    else:
        run_worker(config)
    
    # Cleanup
    rpc_utils.shutdown_rpc()


if __name__ == "__main__":
    main()