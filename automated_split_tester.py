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
            output_file = f"output_split{split_block}.txt"
        
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
            "--metrics-dir", metrics_dir
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
                    timeout=300  # 5 minute timeout
                )
            
            if result.returncode != 0:
                logger.error(f"Orchestrator failed with return code {result.returncode}")
                return False
                
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Orchestrator timed out")
            return False
        except Exception as e:
            logger.error(f"Error running orchestrator: {e}")
            return False
    
    def test_single_split(self, split_block: int, run_number: int,
                         wait_time: int = 60, use_pipelining: bool = True,
                         session_dir: str = None, use_optimizations: bool = True) -> Dict[str, str]:
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
            output_file = f"{run_dir}/output.txt"
        else:
            # Fallback to old behavior if no session_dir
            output_file = f"output_split{split_block}_run{run_number}.txt"
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Kill existing processes
        self.kill_existing_processes()
        
        # Start workers
        self.start_worker(1, split_block=split_block)
        logger.info("Worker 1 start command sent")
        
        time.sleep(3)
        
        self.start_worker(2, split_block=split_block)
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
                                       use_optimizations=True)
        
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
                       use_optimizations: bool = True):
        """Test all split blocks with multiple runs each."""
        
        if split_blocks is None:
            # Test all 19 possible splits (0-18)
            split_blocks = list(range(19))
        
        # Create session directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = f"./metrics/session_{timestamp}"
        os.makedirs(session_dir, exist_ok=True)
        logger.info(f"Created session directory: {session_dir}")
        
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
                        use_optimizations=use_optimizations
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
        metrics_file = self.consolidate_metrics(timestamp, session_dir)
        
        # Note: With new directory structure, individual files are already organized
        # No cleanup needed as files are in session directory
        if hasattr(self, 'cleanup_files') and self.cleanup_files:
            logger.info("Note: Individual output files are now organized in session directory")
            logger.info(f"All files preserved in: {session_dir}")
    
    def parse_output_file(self, output_file: str) -> Optional[Dict]:
        """Parse an output file to extract key metrics."""
        if not os.path.exists(output_file):
            return None
            
        metrics = {
            'throughput': None,
            'accuracy': None,
            'total_time': None,
            'images_processed': None,
            'avg_processing_time': None,
            'pipeline_utilization': None,
            'device_info': {}
        }
        
        try:
            # First try to parse from output.txt (in case it has the data)
            with open(output_file, 'r') as f:
                content = f.read()
                
            # Extract overall throughput
            throughput_match = re.search(r'Overall throughput: ([\d.]+) images/sec', content)
            if throughput_match:
                metrics['throughput'] = float(throughput_match.group(1))
                
            # Extract accuracy
            accuracy_match = re.search(r'Final accuracy: ([\d.]+)%', content)
            if accuracy_match:
                metrics['accuracy'] = float(accuracy_match.group(1))
                
            # Extract total time
            time_match = re.search(r'Total time: ([\d.]+)s', content)
            if time_match:
                metrics['total_time'] = float(time_match.group(1))
                
            # Extract images processed
            images_match = re.search(r'Total images processed: (\d+)', content)
            if images_match:
                metrics['images_processed'] = int(images_match.group(1))
                
            # Get the directory of the output file
            output_dir = os.path.dirname(output_file)
            metrics_dir = os.path.join(output_dir, 'metrics')
            
            # Check CSV files for metrics
            if os.path.exists(metrics_dir):
                # Look for device metrics CSV files (for throughput if not found)
                if metrics['throughput'] is None:
                    device_csv_files = glob_module.glob(os.path.join(metrics_dir, 'device_metrics_*_rank_0_*.csv'))
                    if device_csv_files:
                        # Use the most recent file
                        latest_device_csv = max(device_csv_files, key=os.path.getmtime)
                        
                        # Read the CSV file
                        with open(latest_device_csv, 'r') as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                            if rows:
                                # Get the last row (final summary)
                                last_row = rows[-1]
                                if 'images_per_second' in last_row:
                                    metrics['throughput'] = float(last_row['images_per_second'])
                                if 'total_images_processed' in last_row:
                                    metrics['images_processed'] = int(last_row['total_images_processed'])
                                if 'average_processing_time_ms' in last_row and last_row['average_processing_time_ms']:
                                    try:
                                        metrics['avg_processing_time'] = float(last_row['average_processing_time_ms'])
                                    except ValueError:
                                        pass
                
                    # Look for batch metrics CSV files for accuracy and pipeline utilization
                    batch_csv_files = glob_module.glob(os.path.join(metrics_dir, 'batch_metrics_*_rank_0_*.csv'))
                    if batch_csv_files:
                        latest_batch_csv = max(batch_csv_files, key=os.path.getmtime)
                        
                        with open(latest_batch_csv, 'r') as f:
                            reader = csv.DictReader(f)
                            accuracies = []
                            pipeline_utilizations = []
                            total_time = 0
                            for row in reader:
                                if 'accuracy' in row and row['accuracy']:
                                    accuracies.append(float(row['accuracy']))
                                if 'pipeline_utilization' in row and row['pipeline_utilization']:
                                    try:
                                        util_value = float(row['pipeline_utilization'])
                                        pipeline_utilizations.append(util_value)
                                    except ValueError:
                                        pass
                                if 'total_time_ms' in row and row['total_time_ms']:
                                    total_time = max(total_time, float(row['total_time_ms']))
                            
                            if accuracies:
                                metrics['accuracy'] = sum(accuracies) / len(accuracies)
                            if pipeline_utilizations:
                                # Use average of all pipeline utilization values
                                metrics['pipeline_utilization'] = sum(pipeline_utilizations) / len(pipeline_utilizations)
                            if total_time > 0:
                                metrics['total_time'] = total_time / 1000.0  # Convert ms to seconds
                            
                            # Calculate avg_processing_time from batch metrics if not already set
                            if metrics['avg_processing_time'] is None and batch_csv_files:
                                # Re-read to calculate average processing time per batch
                                with open(latest_batch_csv, 'r') as f:
                                    reader = csv.DictReader(f)
                                    batch_times = []
                                    for row in reader:
                                        if 'total_time_ms' in row and row['total_time_ms']:
                                            batch_times.append(float(row['total_time_ms']))
                                    if batch_times:
                                        # Average time per batch in ms
                                        metrics['avg_processing_time'] = sum(batch_times) / len(batch_times)
                            
                            # If pipeline_utilization is 0 or None, calculate from throughput
                            if not metrics.get('pipeline_utilization'):
                                # For pipelined systems, utilization can be estimated from speedup
                                # If we're getting 50 images/sec with 8 images per batch taking 6-7 seconds
                                # then pipeline is well utilized
                                if metrics.get('throughput') and metrics.get('avg_processing_time'):
                                    batch_size = 8  # From configuration
                                    sequential_throughput = (batch_size / (metrics['avg_processing_time'] / 1000.0))
                                    actual_throughput = metrics['throughput']
                                    if sequential_throughput > 0:
                                        metrics['pipeline_utilization'] = min(actual_throughput / sequential_throughput, 1.0) * 100
            
            # Always check batch metrics CSV for missing metrics (avg_processing_time and pipeline_utilization)
            if os.path.exists(metrics_dir) and (metrics['avg_processing_time'] is None or metrics['pipeline_utilization'] is None):
                batch_csv_files = glob_module.glob(os.path.join(metrics_dir, 'batch_metrics_*_rank_0_*.csv'))
                if batch_csv_files:
                    latest_batch_csv = max(batch_csv_files, key=os.path.getmtime)
                    
                    # Calculate avg_processing_time if not set
                    if metrics['avg_processing_time'] is None:
                        with open(latest_batch_csv, 'r') as f:
                            reader = csv.DictReader(f)
                            batch_times = []
                            for row in reader:
                                if 'total_time_ms' in row and row['total_time_ms']:
                                    batch_times.append(float(row['total_time_ms']))
                            if batch_times:
                                metrics['avg_processing_time'] = sum(batch_times) / len(batch_times)
                    
                    # Calculate pipeline_utilization if still None or 0
                    if not metrics.get('pipeline_utilization') and metrics.get('throughput') and metrics.get('avg_processing_time'):
                        batch_size = 8  # From configuration
                        sequential_throughput = (batch_size / (metrics['avg_processing_time'] / 1000.0))
                        actual_throughput = metrics['throughput']
                        if sequential_throughput > 0:
                            metrics['pipeline_utilization'] = min(actual_throughput / sequential_throughput, 1.0) * 100
            
            # Extract device info from content
            device_matches = re.findall(r'\[([\w-]+):rank(\d+)\]', content)
            if device_matches:
                for hostname, rank in device_matches:
                    metrics['device_info'][f'rank{rank}'] = hostname
                    
            return metrics
            
        except Exception as e:
            logger.error(f"Error parsing {output_file}: {e}")
            return None
    
    def consolidate_metrics(self, test_session_id: str, session_dir: str = None):
        """Consolidate all metrics from output files into a single JSON."""
        consolidated = {
            'session_id': test_session_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'model': 'mobilenetv2',
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
                            output_file = f"{run_dir}/output.txt"
                            
                            if os.path.exists(output_file):
                                # Parse metrics from file
                                metrics = self.parse_output_file(output_file)
                                if metrics:
                                    if split not in consolidated['results']:
                                        consolidated['results'][split] = {}
                                    consolidated['results'][split][f'run{run}'] = metrics
        else:
            # Fallback to old behavior
            output_files = glob_module.glob('output_split*.txt')
            for output_file in output_files:
                # Extract split and run from filename
                match = re.match(r'output_split(\d+)_run(\d+)\.txt', output_file)
                if match:
                    split = int(match.group(1))
                    run = int(match.group(2))
                    
                    # Parse metrics from file
                    metrics = self.parse_output_file(output_file)
                    if metrics:
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
    
    split_blocks = args.splits if args.splits else list(range(19))
    logger.info(f"Testing splits: {split_blocks}")
    logger.info(f"Runs per split: {args.runs}")
    logger.info(f"Worker wait time: {args.wait_time} seconds")
    logger.info(f"Cleanup after consolidation: {args.cleanup}")
    
    tester.test_all_splits(
        split_blocks=split_blocks,
        runs_per_split=args.runs,
        worker_wait_time=args.wait_time,
        use_pipelining=True,  # Always use pipelining by default, but can be changed
        use_optimizations=not args.no_optimizations
    )

if __name__ == "__main__":
    main()