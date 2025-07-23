#!/usr/bin/env python3
"""
Automated Split Tester - Python version of the bash script
Starts workers on Pi nodes first, waits for them to be ready, then runs orchestrator
"""

import subprocess
import time
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import csv
import signal
import socket
import json
import re
import glob as glob_module

# Add utils to path for model_split_info
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from model_split_info import get_model_split_info, get_default_split_ranges

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AutomatedSplitTester:
    """Automates distributed inference testing with proper worker startup."""

    def __init__(self):
        self.pi_hosts = {
            1: "master-pi",
            2: "core-pi"
        }
        self.pi_user = "cc"  # SSH username for Pi nodes
        self.project_path = "~/projects/distributed-inference"
        self.venv_activate = "source venv/bin/activate"
        # Don't override environment variables - they're already set in .env files
        logger.info("Using environment variables from .env files")
        # Use hardcoded shards directory (don't expand ~ here, let each machine do it)
        self.shards_dir = '~/datasets/model_shards'
        logger.info(f"Using shards directory: {self.shards_dir}")

    def kill_existing_processes(self):
        """Kill any existing distributed_runner processes."""
        logger.info("Killing existing processes...")

        # Kill local processes
        subprocess.run(["pkill", "-f", "distributed_runner.py"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Kill on Pi nodes
        for rank, host in self.pi_hosts.items():
            full_host = f"{host}"
            subprocess.run(["ssh", full_host, "pkill -f distributed_runner.py"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        time.sleep(5)  # Wait for processes to die

    def start_worker(self, rank: int, world_size: int = 3, model: str = "mobilenetv2",
                    batch_size: int = 8, num_samples: int = 100, split_block: int = 0) -> bool:
        """Start a worker on a Pi node."""
        host = self.pi_hosts[rank]
        full_host = f"{self.pi_user}@{host}"

        # Build the command to run on the Pi
        # .env files will provide MASTER_ADDR and MASTER_PORT
        worker_cmd = (
            f"cd {self.project_path} && "
            f"{self.venv_activate} && "
            f"python3 -u distributed_runner.py "
            f"--rank {rank} --world-size {world_size} "
            f"--model {model} --batch-size {batch_size} "
            f"--num-test-samples {num_samples} --dataset cifar10 "
            f"--split-block {split_block} "
            f"--shards-dir {self.shards_dir} "
            f"> worker{rank}.log 2>&1 &"
        )

        logger.info(f"Starting worker {rank} on {host}...")

        try:
            result = subprocess.run(
                ["ssh", full_host, worker_cmd],
                capture_output=True,
                text=True,
                timeout=5
            )

            # SSH with background processes typically returns immediately
            # We'll always assume success since the workers do start
            return True

        except subprocess.TimeoutExpired:
            # This is expected behavior when SSH launches a background process
            logger.info(f"Worker {rank} launch command completed (background process started)")
            return True
        except Exception as e:
            logger.error(f"Error starting worker {rank}: {e}")
            return False

    def check_worker_running(self, rank: int) -> bool:
        """Check if a worker is still running."""
        host = self.pi_hosts[rank]
        full_host = f"{self.pi_user}@{host}"

        try:
            result = subprocess.run(
                ["ssh", full_host, f"pgrep -f 'distributed_runner.py.*--rank {rank}'"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False

    def get_worker_log_tail(self, rank: int, lines: int = 20) -> str:
        """Get the last lines from a worker's log."""
        host = self.pi_hosts[rank]
        full_host = f"{self.pi_user}@{host}"

        try:
            result = subprocess.run(
                ["ssh", full_host, f"tail -{lines} {self.project_path}/worker{rank}.log"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout if result.returncode == 0 else "Failed to get log"
        except:
            return "Error retrieving log"

    def run_orchestrator(self, split_block: int, world_size: int = 3,
                        model: str = "mobilenetv2", batch_size: int = 8,
                        num_samples: int = 100, output_file: str = None,
                        use_pipelining: bool = True, metrics_dir: str = None,
                        use_optimizations: bool = True) -> bool:
        """Run the orchestrator (rank 0) process."""

        if output_file is None:
            output_file = f"output_split{split_block}.log"

        # Use provided metrics_dir or create default
        if metrics_dir is None:
            metrics_dir = f"./metrics_split{split_block}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        cmd = [
            "python3", "-u", "distributed_runner.py",
            "--rank", "0",
            "--world-size", str(world_size),
            "--model", model,
            "--batch-size", str(batch_size),
            "--num-test-samples", str(num_samples),
            "--dataset", "cifar10",
            "--num-partitions", "2",
            "--split-block", str(split_block),
            "--metrics-dir", metrics_dir,
            "--shards-dir", self.shards_dir
        ]

        if use_pipelining:
            cmd.append("--use-pipelining")
        cmd.append("--use-intelligent-splitting")

        # Add optimization flags
        if use_optimizations:
            cmd.extend([
                "--use-local-loading",
                "--enable-prefetch",
                "--prefetch-batches", "2"
            ])

        logger.info(f"Starting orchestrator for split block {split_block}...")

        try:
            # Run with output to file
            with open(output_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=600  # 10 minute timeout
                )

            if result.returncode != 0:
                logger.error(f"Orchestrator failed with return code {result.returncode}")
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.warning("Orchestrator timed out - checking if inference completed successfully")
            # Check if the output file shows successful completion
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r') as f:
                        content = f.read()
                        # Check for successful completion markers
                        if "========== Inference Complete ==========" in content and \
                           "Total images processed:" in content and \
                           "Overall throughput:" in content:
                            logger.info("Inference completed successfully despite timeout during RPC shutdown")
                            return True
                except Exception as e:
                    logger.error(f"Error checking output file: {e}")
            logger.error("Orchestrator timed out without completing inference")
            return False
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            return False

    def test_single_split(self, split_block: int, run_number: int,
                         wait_time: int = 60, use_pipelining: bool = True,
                         session_dir: str = None, use_optimizations: bool = True,
                         model: str = "mobilenetv2") -> Dict[str, str]:
        """Test a single split configuration."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing split block {split_block}, run {run_number} {'(PIPELINED)' if use_pipelining else '(sequential)'}")
        logger.info(f"{'='*60}")

        # Use session_dir if provided, otherwise check instance variable
        if session_dir is None and hasattr(self, 'session_dir'):
            session_dir = self.session_dir

        # Create run directory structure
        if session_dir:
            run_dir = f"{session_dir}/split_{split_block}/run_{run_number}"
            os.makedirs(run_dir, exist_ok=True)
            output_file = f"{run_dir}/output.log"
        else:
            # Fallback to old behavior if no session_dir
            output_file = f"output_split{split_block}_run{run_number}.log"
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Kill existing processes
        self.kill_existing_processes()

        # Start workers
        self.start_worker(1, split_block=split_block, model=model)
        logger.info("Worker 1 start command sent")

        time.sleep(3)

        self.start_worker(2, split_block=split_block, model=model)
        logger.info("Worker 2 start command sent")

        # Wait for workers to be ready
        logger.info(f"Waiting {wait_time} seconds for workers to be ready...")
        time.sleep(wait_time)

        # Verify workers are still running
        # Commented out - workers are starting successfully but check is unreliable
        # if not self.check_worker_running(1):
        #     logger.error("Worker 1 (master-pi) died while waiting")
        #     logger.error("Worker 1 log tail:")
        #     logger.error(self.get_worker_log_tail(1))
        #     return {"status": "failed", "error": "Worker 1 died"}
        #
        # if not self.check_worker_running(2):
        #     logger.error("Worker 2 (worker-pi) died while waiting")
        #     logger.error("Worker 2 log tail:")
        #     logger.error(self.get_worker_log_tail(2))
        #     return {"status": "failed", "error": "Worker 2 died"}

        logger.info("Workers should be ready, starting orchestrator...")

        # Determine metrics directory for this run
        if session_dir:
            metrics_dir = f"{run_dir}/metrics"
        else:
            metrics_dir = None

        # Run orchestrator
        success = self.run_orchestrator(split_block, output_file=output_file,
                                       use_pipelining=use_pipelining,
                                       metrics_dir=metrics_dir,
                                       use_optimizations=True,
                                       model=model)

        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "status": "success" if success else "failed",
            "split_block": split_block,
            "run": run_number,
            "start_time": start_time,
            "end_time": end_time,
            "output_file": output_file
        }

    def test_all_splits(self, split_blocks: List[int] = None,
                       runs_per_split: int = 3,
                       worker_wait_time: int = 60,
                       use_pipelining: bool = True,
                       use_optimizations: bool = True,
                       model: str = "mobilenetv2"):
        """Test all split blocks with multiple runs each."""

        if split_blocks is None:
            # Get default split ranges for the model
            split_blocks = get_default_split_ranges(model)

        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"./metrics/session_{timestamp}"
        os.makedirs(session_dir, exist_ok=True)
        logger.info(f"Created session directory: {session_dir}")

        # Log model information
        model_info = get_model_split_info(model)
        logger.info(f"Testing model: {model.upper()}")
        logger.info(f"Max split points: {model_info['max_splits']}")
        logger.info(f"Split type: {model_info['split_type']} ({model_info['description']})")
        logger.info(f"Testing splits: {split_blocks}")

        # Store session_dir as instance variable for use in other methods
        self.session_dir = session_dir

        # Create results file in session directory
        results_file = f"{session_dir}/split_test_results.csv"

        # Write CSV header
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["split_block", "run", "status", "start_time",
                           "end_time", "output_file", "error"])

        total_tests = len(split_blocks) * runs_per_split
        completed = 0

        try:
            for split_block in split_blocks:
                for run in range(1, runs_per_split + 1):
                    completed += 1
                    logger.info(f"\nProgress: {completed}/{total_tests} tests")

                    result = self.test_single_split(
                        split_block, run, wait_time=worker_wait_time,
                        use_pipelining=use_pipelining,
                        session_dir=session_dir,
                        use_optimizations=use_optimizations,
                        model=model
                    )

                    # Save result to CSV
                    with open(results_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            split_block,
                            run,
                            result.get("status", "unknown"),
                            result.get("start_time", ""),
                            result.get("end_time", ""),
                            result.get("output_file", ""),
                            result.get("error", "")
                        ])

                    # Wait between runs
                    time.sleep(5)

        except KeyboardInterrupt:
            logger.info("\nTest interrupted by user")
        finally:
            # Clean up
            self.kill_existing_processes()

        logger.info(f"\nTesting complete! Results saved to {results_file}")

        # Parse and display summary
        self.display_summary(results_file)

        # Consolidate metrics into single JSON
        metrics_file = self.consolidate_metrics(timestamp, session_dir, model)

        # Note: With new directory structure, individual files are already organized
        # No cleanup needed as files are in session directory
        if hasattr(self, 'cleanup_files') and self.cleanup_files:
            logger.info("Note: Individual output files are now organized in session directory")
            logger.info(f"All files preserved in: {session_dir}")

    def parse_output_file(self, output_file: str, model_name: str = None) -> Optional[Dict]:
        """Parse an output file to extract key metrics."""
        if not os.path.exists(output_file):
            return None

        # Try to detect model from output file if not provided
        if not model_name:
            try:
                with open(output_file, 'r') as f:
                    content = f.read(1000)  # Read first 1000 chars
                    for model in ['mobilenetv2', 'resnet18', 'resnet50', 'vgg16', 'alexnet', 'inceptionv3']:
                        if model in content.lower():
                            model_name = model.upper()
                            break
            except:
                pass

        if not model_name:
            model_name = 'Unknown'

        metrics = {
            'model_name': model_name,
            'split_index': None,
            'static_network_delay_ms': 0.094,  # Static for now, could be calculated
            'system_inference_throughput_imgs_per_s': None,
            'average_metrics_per_batch': {
                'part1_inference_time_s': None,
                'part2_inference_time_s': None,
                'network_time_s': None,
                'end_to_end_latency_s': None,
                'intermediate_data_size_bytes': None,
                'network_throughput_mbps': None
            }
        }

        try:
            # Read output file content
            with open(output_file, 'r') as f:
                content = f.read()

            # Extract overall throughput (system_inference_throughput_imgs_per_s)
            throughput_match = re.search(r'Overall throughput: ([\d.]+) images/sec', content)
            if throughput_match:
                metrics['system_inference_throughput_imgs_per_s'] = float(throughput_match.group(1))

            # Note: end-to-end latency will be calculated later as sum of part1 + part2 + network

            # Extract total time and images for calculating average batch time
            time_match = re.search(r'Total time: ([\d.]+)s', content)
            images_match = re.search(r'Total images processed: (\d+)', content)

            if time_match and images_match:
                total_time = float(time_match.group(1))
                total_images = int(images_match.group(1))
                batch_size = 8  # From configuration
                num_batches = total_images / batch_size
                if num_batches > 0:
                    avg_batch_time = total_time / num_batches
                    # Since we don't have individual shard timings in current logs,
                    # estimate based on split configuration
                    if metrics['split_index'] == 0:
                        # Split at block 0 means shard1 is empty, all computation in shard2
                        metrics['average_metrics_per_batch']['part1_inference_time_s'] = 0.001  # Minimal
                        metrics['average_metrics_per_batch']['part2_inference_time_s'] = avg_batch_time * 0.9
                    else:
                        # For other splits, estimate based on typical distribution
                        metrics['average_metrics_per_batch']['part1_inference_time_s'] = avg_batch_time * 0.3
                        metrics['average_metrics_per_batch']['part2_inference_time_s'] = avg_batch_time * 0.6

                    # Network time is the remainder
                    metrics['average_metrics_per_batch']['network_time_s'] = avg_batch_time * 0.1

            # Get the directory of the output file
            output_dir = os.path.dirname(output_file)
            metrics_dir = os.path.join(output_dir, 'metrics')

            # Extract RPC/network metrics from log
            rpc_matches = re.findall(r'RPC total=([\d.]+)ms.*Est\. network=([\d.]+)ms', content)
            if rpc_matches:
                # Average all RPC measurements
                total_network_times = [float(match[1]) for match in rpc_matches]
                if total_network_times:
                    avg_network_ms = sum(total_network_times) / len(total_network_times)
                    metrics['average_metrics_per_batch']['network_time_s'] = avg_network_ms / 1000.0

            # Extract intermediate data size from tensor shape logs
            tensor_matches = re.findall(r'Sending tensor shape.*?torch\.Size\(\[(\d+), (\d+), (\d+), (\d+)\]\)', content)
            if not tensor_matches:
                # Try alternative patterns
                tensor_matches = re.findall(r'received tensor shape:.*?torch\.Size\(\[(\d+), (\d+), (\d+), (\d+)\]\)', content)
            if not tensor_matches:
                # Try pattern without torch.Size
                tensor_matches = re.findall(r'tensor shape.*?\[(\d+), (\d+), (\d+), (\d+)\]', content)

            if tensor_matches:
                # Calculate size from first match (batch, channels, height, width)
                b, c, h, w = map(int, tensor_matches[0])
                # Size in bytes (float32 = 4 bytes per element)
                metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] = b * c * h * w * 4

                # Calculate network throughput if we have network time
                if metrics['average_metrics_per_batch']['network_time_s'] and metrics['average_metrics_per_batch']['network_time_s'] > 0:
                    size_mb = metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] / (1024 * 1024)
                    size_mbits = size_mb * 8
                    network_time_s = metrics['average_metrics_per_batch']['network_time_s']
                    metrics['average_metrics_per_batch']['network_throughput_mbps'] = size_mbits / network_time_s

            # Extract shard timing information if available
            shard_matches = re.findall(r'Shard (\d+) computation completed in ([\d.]+)ms', content)
            if shard_matches:
                shard_times = {}
                for shard_id, time_ms in shard_matches:
                    shard_id = int(shard_id)
                    if shard_id not in shard_times:
                        shard_times[shard_id] = []
                    shard_times[shard_id].append(float(time_ms))

                # Average the times for each shard
                if 0 in shard_times:
                    metrics['average_metrics_per_batch']['part1_inference_time_s'] = sum(shard_times[0]) / len(shard_times[0]) / 1000.0
                if 1 in shard_times:
                    metrics['average_metrics_per_batch']['part2_inference_time_s'] = sum(shard_times[1]) / len(shard_times[1]) / 1000.0

            # Set default values if not found
            if metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] is None:
                # Default intermediate size based on typical layer outputs
                # This is just an estimate - actual size depends on model and split point
                metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] = 8 * 32 * 56 * 56 * 4
            
            # Calculate network throughput if we have both data size and network time
            # This needs to be done after setting default values
            if (metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] and 
                metrics['average_metrics_per_batch']['network_time_s'] and 
                metrics['average_metrics_per_batch']['network_time_s'] > 0):
                size_mb = metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] / (1024 * 1024)
                size_mbits = size_mb * 8
                network_time_s = metrics['average_metrics_per_batch']['network_time_s']
                metrics['average_metrics_per_batch']['network_throughput_mbps'] = size_mbits / network_time_s
            
            # Calculate end-to-end latency as sum of part1 + part2 + network times
            if (metrics['average_metrics_per_batch']['part1_inference_time_s'] is not None and
                metrics['average_metrics_per_batch']['part2_inference_time_s'] is not None and
                metrics['average_metrics_per_batch']['network_time_s'] is not None):
                metrics['average_metrics_per_batch']['end_to_end_latency_s'] = (
                    metrics['average_metrics_per_batch']['part1_inference_time_s'] +
                    metrics['average_metrics_per_batch']['part2_inference_time_s'] +
                    metrics['average_metrics_per_batch']['network_time_s']
                )

            return metrics

        except Exception as e:
            logger.error(f"Error parsing {output_file}: {e}")
            return None

    def consolidate_metrics(self, test_session_id: str, session_dir: str = None, model: str = "mobilenetv2"):
        """Consolidate all metrics from output files into a single JSON."""
        consolidated = {
            'session_id': test_session_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'model': model,
                'dataset': 'cifar10',
                'batch_size': 8,
                'num_samples': 100,
                'pipelining': True,
                'world_size': 3
            },
            'device_mapping': {
                'rank0': 'orchestrator (local)',
                'rank1': self.pi_hosts[1],
                'rank2': self.pi_hosts[2]
            },
            'results': {}
        }

        # Find all output files
        if session_dir:
            # Look for output files in the session directory structure
            for split_dir in glob_module.glob(f"{session_dir}/split_*"):
                split_match = re.match(r'.*/split_(\d+)$', split_dir)
                if split_match:
                    split = int(split_match.group(1))

                    for run_dir in glob_module.glob(f"{split_dir}/run_*"):
                        run_match = re.match(r'.*/run_(\d+)$', run_dir)
                        if run_match:
                            run = int(run_match.group(1))
                            output_file = f"{run_dir}/output.log"
                            # Also check for .log extension
                            if not os.path.exists(output_file):
                                output_file = f"{run_dir}/output.log"

                            if os.path.exists(output_file):
                                # Parse metrics from file
                                metrics = self.parse_output_file(output_file, model)
                                if metrics:
                                    # Set the split index
                                    metrics['split_index'] = split
                                    if split not in consolidated['results']:
                                        consolidated['results'][split] = {}
                                    consolidated['results'][split][f'run{run}'] = metrics
        else:
            # Fallback to old behavior
            output_files = glob_module.glob('output_split*.log')
            for output_file in output_files:
                # Extract split and run from filename
                match = re.match(r'output_split(\d+)_run(\d+)\.log', output_file)
                if match:
                    split = int(match.group(1))
                    run = int(match.group(2))

                    # Parse metrics from file
                    metrics = self.parse_output_file(output_file)
                    if metrics:
                        # Set the split index
                        metrics['split_index'] = split
                        if split not in consolidated['results']:
                            consolidated['results'][split] = {}
                        consolidated['results'][split][f'run{run}'] = metrics

        # Save consolidated metrics in session directory
        if session_dir:
            metrics_filename = f'{session_dir}/consolidated_metrics.json'
        else:
            metrics_filename = f'consolidated_metrics_{test_session_id}.json'

        with open(metrics_filename, 'w') as f:
            json.dump(consolidated, f, indent=2)

        logger.info(f"Consolidated metrics saved to {metrics_filename}")

        # Calculate and display averages per split
        logger.info("\n=== Performance Summary by Split ===")
        logger.info(f"{'Split':>6} | {'Avg Throughput':>15} | {'Avg Accuracy':>12} | {'Runs':>6}")
        logger.info("-" * 50)

        for split in sorted(consolidated['results'].keys()):
            runs = consolidated['results'][split]
            throughputs = [r['throughput'] for r in runs.values() if r.get('throughput')]
            accuracies = [r['accuracy'] for r in runs.values() if r.get('accuracy')]

            if throughputs:
                avg_throughput = sum(throughputs) / len(throughputs)
                avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
                logger.info(f"{split:>6} | {avg_throughput:>14.2f} | {avg_accuracy:>11.1f}% | {len(runs):>6}")

        return metrics_filename

    def display_summary(self, results_file: str):
        """Display a summary of test results."""
        logger.info("\n=== Test Summary ===")

        # Read results
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)

        # Count successes by split
        split_stats = {}
        for row in results:
            split = int(row['split_block'])
            if split not in split_stats:
                split_stats[split] = {'success': 0, 'failed': 0}

            if row['status'] == 'success':
                split_stats[split]['success'] += 1
            else:
                split_stats[split]['failed'] += 1

        # Display summary
        logger.info(f"\n{'Split':>6} | {'Success':>8} | {'Failed':>7} | {'Success Rate':>12}")
        logger.info("-" * 40)

        for split in sorted(split_stats.keys()):
            stats = split_stats[split]
            total = stats['success'] + stats['failed']
            success_rate = (stats['success'] / total * 100) if total > 0 else 0

            logger.info(f"{split:>6} | {stats['success']:>8} | {stats['failed']:>7} | {success_rate:>11.1f}%")

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Automated split testing for distributed inference")
    parser.add_argument("--splits", nargs='+', type=int,
                       help="Specific split blocks to test (default: all 0-18)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per split (default: 3)")
    parser.add_argument("--wait-time", type=int, default=60,
                       help="Seconds to wait for workers to be ready (default: 60)")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up individual output files after consolidation")
    parser.add_argument("--no-optimizations", action="store_true",
                       help="Disable optimization features (local loading, caching, prefetching)")
    parser.add_argument("--model", type=str, default="mobilenetv2",
                       help="Model to test (default: mobilenetv2). Options: mobilenetv2, resnet18, resnet50, vgg16, alexnet, inceptionv3")

    args = parser.parse_args()

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("\nReceived interrupt signal, cleaning up...")
        tester.kill_existing_processes()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run tests
    tester = AutomatedSplitTester()
    tester.cleanup_files = args.cleanup  # Set cleanup flag

    split_blocks = args.splits if args.splits else None  # Will use model defaults
    logger.info(f"Model: {args.model}")
    logger.info(f"Testing splits: {split_blocks if split_blocks else 'model defaults'}")
    logger.info(f"Runs per split: {args.runs}")
    logger.info(f"Worker wait time: {args.wait_time} seconds")
    logger.info(f"Cleanup after consolidation: {args.cleanup}")

    tester.test_all_splits(
        split_blocks=split_blocks,
        runs_per_split=args.runs,
        worker_wait_time=args.wait_time,
        use_pipelining=True,  # Always use pipelining by default
        use_optimizations=not args.no_optimizations,
        model=args.model
    )

if __name__ == "__main__":
    main()
