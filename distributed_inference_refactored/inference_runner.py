"""Clean inference runner for distributed model execution."""

import time
import torch
import logging
from typing import Optional, Dict, Any

from config import DistributedConfig
from distributed_model import DistributedModel
from metrics import EnhancedMetricsCollector
from core import ModelLoader


class InferenceRunner:
    """Handles the inference execution logic."""
    
    def __init__(
        self, 
        config: DistributedConfig,
        metrics_collector: EnhancedMetricsCollector
    ):
        self.config = config
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Initialize model and data loader
        self.model = None
        self.data_loader = None
    
    def setup(self):
        """Setup model and data loader."""
        self.logger.info("Setting up inference runner")
        
        # Create distributed model
        self.model = DistributedModel(self.config, self.metrics_collector)
        
        # Load dataset
        model_loader = ModelLoader(self.config.model.models_dir)
        self.data_loader = model_loader.load_dataset(
            self.config.inference.dataset,
            self.config.model.model_type,
            self.config.inference.batch_size
        )
        
        self.logger.info("Inference runner setup complete")
    
    def run_inference(self) -> Dict[str, Any]:
        """Run inference and return results."""
        if self.config.use_pipelining:
            return self._run_pipelined_inference()
        else:
            return self._run_sequential_inference()
    
    def _run_sequential_inference(self) -> Dict[str, Any]:
        """Run sequential (non-pipelined) inference."""
        self.logger.info("Starting sequential inference")
        
        start_time = time.time()
        total_images = 0
        num_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for images, labels in self.data_loader:
                if total_images >= self.config.inference.num_test_samples:
                    break
                
                # Trim batch if necessary
                remaining = self.config.inference.num_test_samples - total_images
                if images.size(0) > remaining:
                    images = images[:remaining]
                    labels = labels[:remaining]
                
                # Move to CPU for consistency
                images = images.to("cpu")
                labels = labels.to("cpu")
                
                # Start batch tracking
                self.metrics_collector.start_batch(batch_count, len(images))
                
                # Run inference
                batch_start = time.time()
                output = self.model(images, batch_id=batch_count)
                batch_time = time.time() - batch_start
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                batch_correct = (predicted == labels).sum().item()
                num_correct += batch_correct
                
                batch_accuracy = (batch_correct / len(labels)) * 100.0
                
                # End batch tracking
                self.metrics_collector.end_batch(batch_count, accuracy=batch_accuracy)
                
                self.logger.info(
                    f"Batch {batch_count + 1}: accuracy={batch_accuracy:.2f}%, "
                    f"time={batch_time:.3f}s, throughput={len(images)/batch_time:.2f} img/s"
                )
                
                total_images += len(images)
                batch_count += 1
        
        elapsed_time = time.time() - start_time
        
        return {
            'total_images': total_images,
            'total_time': elapsed_time,
            'accuracy': (num_correct / total_images * 100.0) if total_images > 0 else 0.0,
            'throughput': total_images / elapsed_time if elapsed_time > 0 else 0.0,
            'batch_count': batch_count
        }
    
    def _run_pipelined_inference(self) -> Dict[str, Any]:
        """Run pipelined inference with multiple batches in flight."""
        self.logger.info("Starting pipelined inference")
        
        if not self.model.pipeline_manager:
            raise RuntimeError("Pipeline manager not initialized")
        
        start_time = time.time()
        total_images = 0
        num_correct = 0
        batch_count = 0
        
        # Configuration
        max_batches_in_flight = 3
        active_batches = {}
        
        with torch.no_grad():
            data_iter = iter(self.data_loader)
            batch_id = 0
            
            while total_images < self.config.inference.num_test_samples:
                # Start new batches if we have capacity
                while (len(active_batches) < max_batches_in_flight and 
                       total_images < self.config.inference.num_test_samples):
                    try:
                        images, labels = next(data_iter)
                    except StopIteration:
                        break
                    
                    # Trim if necessary
                    remaining = self.config.inference.num_test_samples - total_images
                    if images.size(0) > remaining:
                        images = images[:remaining]
                        labels = labels[:remaining]
                    
                    images = images.to("cpu")
                    labels = labels.to("cpu")
                    
                    # Start batch
                    batch_start_time = self.metrics_collector.start_batch(batch_id, len(images))
                    
                    # Start pipeline processing
                    pipeline_batch_id = self.model.pipeline_manager.start_batch_rpc_pipelined(
                        images, labels
                    )
                    active_batches[pipeline_batch_id] = (images, labels, batch_start_time, batch_id)
                    
                    total_images += len(images)
                    batch_id += 1
                
                # Collect completed batches
                completed_ids = []
                for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                    result = self.model.pipeline_manager.get_completed_batch(pid, timeout=0.001)
                    if result is not None:
                        # Calculate accuracy
                        _, predicted = torch.max(result.data, 1)
                        batch_correct = (predicted == orig_labels).sum().item()
                        num_correct += batch_correct
                        
                        batch_accuracy = (batch_correct / len(orig_labels)) * 100.0
                        
                        # End batch tracking
                        self.metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)
                        
                        completed_ids.append(pid)
                        batch_count += 1
                
                # Remove completed batches
                for pid in completed_ids:
                    del active_batches[pid]
                
                # Small sleep if waiting
                if not completed_ids and len(active_batches) >= max_batches_in_flight:
                    time.sleep(0.01)
            
            # Wait for remaining batches
            while active_batches:
                completed_ids = []
                for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                    result = self.model.pipeline_manager.get_completed_batch(pid, timeout=0.1)
                    if result is not None:
                        _, predicted = torch.max(result.data, 1)
                        batch_correct = (predicted == orig_labels).sum().item()
                        num_correct += batch_correct
                        
                        batch_accuracy = (batch_correct / len(orig_labels)) * 100.0
                        self.metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)
                        
                        completed_ids.append(pid)
                        batch_count += 1
                
                for pid in completed_ids:
                    del active_batches[pid]
        
        elapsed_time = time.time() - start_time
        
        return {
            'total_images': total_images,
            'total_time': elapsed_time,
            'accuracy': (num_correct / total_images * 100.0) if total_images > 0 else 0.0,
            'throughput': total_images / elapsed_time if elapsed_time > 0 else 0.0,
            'batch_count': batch_count,
            'pipeline_stats': self.model.get_pipeline_stats()
        }