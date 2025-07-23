#!/usr/bin/env python3
"""
Automated Runner for Refactored Distributed Inference
Handles deployment, testing, and performance comparison
"""

import subprocess
import time
import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import csv
import json
import socket
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutomatedRunner:
    """Automates distributed inference testing and deployment."""
    
    def __init__(self):
        # Pi configuration
        self.pi_hosts = {
            1: "master-pi",
            2: "core-pi"
        }
        self.pi_user = "cc"
        self.project_path = "~/projects/distributed-inference/distributed_inference_refactored"
        self.venv_activate = "source ../venv/bin/activate"
        
        # Test configurations
        self.test_configs = {
            "quick": {
                "num_samples": 16,
                "batch_size": 8,
                "description": "Quick test with 16 samples"
            },
            "standard": {
                "num_samples": 100,
                "batch_size": 8,
                "description": "Standard test with 100 samples"
            },
            "performance": {
                "num_samples": 1000,
                "batch_size": 16,
                "description": "Performance test with 1000 samples"
            },
            "stress": {
                "num_samples": 5000,
                "batch_size": 32,
                "description": "Stress test with 5000 samples"
            }
        }
        
        # Split configurations for MobileNetV2
        self.split_configs = {
            "balanced": 8,      # Default balanced split
            "early": 4,         # More compute on first device
            "late": 12,         # More compute on second device
            "extreme_early": 2, # Almost all on second device
            "extreme_late": 16  # Almost all on first device
        }
        
    def deploy_to_pis(self, force_copy: bool = False):
        """Deploy refactored code to all Pi nodes."""
        logger.info("Deploying refactored code to Pi nodes...")
        
        for rank, host in self.pi_hosts.items():
            full_host = f"{self.pi_user}@{host}"
            logger.info(f"Deploying to {host}...")
            
            # Create directory if it doesn't exist
            ssh_cmd = f"ssh {full_host} 'mkdir -p {self.project_path}'"
            subprocess.run(ssh_cmd, shell=True)
            
            # Sync the refactored code
            if force_copy:
                # Full copy (slower but ensures everything is updated)
                rsync_cmd = (
                    f"rsync -avz --delete "
                    f"./ {full_host}:{self.project_path}/"
                )
            else:
                # Incremental sync (faster)
                rsync_cmd = (
                    f"rsync -avz "
                    f"--exclude='__pycache__' "
                    f"--exclude='*.pyc' "
                    f"--exclude='enhanced_metrics/' "
                    f"./ {full_host}:{self.project_path}/"
                )
            
            result = subprocess.run(rsync_cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✓ Successfully deployed to {host}")
            else:
                logger.error(f"✗ Failed to deploy to {host}: {result.stderr}")
                return False
        
        return True
    
    def kill_existing_processes(self):
        """Kill any existing distributed inference processes."""
        logger.info("Cleaning up existing processes...")
        
        # Kill local processes (both original and refactored)
        subprocess.run(["pkill", "-f", "distributed_runner.py"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-f", "main.py"], 
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Kill on Pi nodes
        for rank, host in self.pi_hosts.items():
            full_host = f"{self.pi_user}@{host}"
            subprocess.run(
                ["ssh", full_host, "pkill -f 'distributed_runner.py|main.py'"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
        
        time.sleep(3)
    
    def start_worker(self, rank: int, config: Dict) -> bool:
        """Start a worker on a Pi node."""
        host = self.pi_hosts[rank]
        full_host = f"{self.pi_user}@{host}"
        
        worker_cmd = (
            f"cd {self.project_path} && "
            f"{self.venv_activate} && "
            f"python3 -u main.py "
            f"--rank {rank} --world-size {config['world_size']} "
            f"--model {config['model']} --batch-size {config['batch_size']} "
            f"--num-test-samples {config['num_samples']} "
            f"--dataset {config['dataset']} "
        )
        
        # Add optional parameters
        if 'split_block' in config:
            worker_cmd += f"--split-block {config['split_block']} "
        if config.get('use_pipelining'):
            worker_cmd += "--use-pipelining "
        if config.get('disable_local_loading'):
            worker_cmd += "--disable-local-loading "
            
        worker_cmd += f"> worker{rank}.log 2>&1 &"
        
        logger.info(f"Starting worker {rank} on {host}...")
        
        try:
            subprocess.run(
                ["ssh", full_host, worker_cmd],
                timeout=5
            )
            return True
        except subprocess.TimeoutExpired:
            # Expected for background processes
            return True
        except Exception as e:
            logger.error(f"Error starting worker {rank}: {e}")
            return False
    
    def check_worker_logs(self, rank: int, check_string: str = "Ready to receive") -> bool:
        """Check worker logs for readiness."""
        host = self.pi_hosts[rank]
        full_host = f"{self.pi_user}@{host}"
        
        try:
            result = subprocess.run(
                ["ssh", full_host, f"cd {self.project_path} && tail -n 50 worker{rank}.log"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and check_string in result.stdout:
                return True
        except:
            pass
        
        return False
    
    def run_test(self, test_name: str, config: Dict) -> Dict:
        """Run a single test configuration."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        logger.info(f"{'='*60}\n")
        
        # Start workers
        for rank in [1, 2]:
            if not self.start_worker(rank, config):
                logger.error(f"Failed to start worker {rank}")
                return {"status": "failed", "error": "Worker startup failed"}
        
        # Wait for workers to be ready
        logger.info("Waiting for workers to initialize...")
        max_wait = 60
        start_wait = time.time()
        workers_ready = False
        
        while time.time() - start_wait < max_wait:
            if all(self.check_worker_logs(rank, "ready and waiting") for rank in [1, 2]):
                workers_ready = True
                break
            time.sleep(2)
        
        if not workers_ready:
            logger.error("Workers failed to initialize in time")
            return {"status": "failed", "error": "Worker initialization timeout"}
        
        logger.info("Workers ready! Starting master...")
        
        # Run master
        master_cmd = [
            "python3", "main.py",
            "--rank", "0",
            "--world-size", str(config['world_size']),
            "--model", config['model'],
            "--batch-size", str(config['batch_size']),
            "--num-test-samples", str(config['num_samples']),
            "--dataset", config['dataset']
        ]
        
        if 'split_block' in config:
            master_cmd.extend(["--split-block", str(config['split_block'])])
        if config.get('use_pipelining'):
            master_cmd.append("--use-pipelining")
        if config.get('disable_local_loading'):
            master_cmd.append("--disable-local-loading")
        
        start_time = time.time()
        result = subprocess.run(master_cmd, capture_output=True, text=True)
        end_time = time.time()
        
        if result.returncode != 0:
            logger.error(f"Master failed: {result.stderr}")
            return {"status": "failed", "error": result.stderr}
        
        # Parse results
        output = result.stdout
        metrics = {
            "status": "success",
            "duration": end_time - start_time,
            "test_name": test_name,
            "config": config
        }
        
        # Extract metrics from output
        for line in output.split('\n'):
            if "Total images:" in line:
                metrics['total_images'] = int(line.split(':')[1].strip())
            elif "Total time:" in line:
                metrics['total_time'] = float(line.split(':')[1].strip().rstrip('s'))
            elif "Accuracy:" in line and "Final" not in line:
                metrics['accuracy'] = float(line.split(':')[1].strip().rstrip('%'))
            elif "Throughput:" in line:
                metrics['throughput'] = float(line.split(':')[1].strip().split()[0])
        
        logger.info(f"\nTest completed successfully!")
        logger.info(f"Throughput: {metrics.get('throughput', 0):.2f} images/sec")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.2f}%")
        
        return metrics
    
    def run_comparison_suite(self):
        """Run a suite of tests comparing different configurations."""
        results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Test 1: Different split points
        logger.info("\n" + "="*80)
        logger.info("TEST SUITE 1: Split Point Comparison")
        logger.info("="*80)
        
        for split_name, split_block in self.split_configs.items():
            config = {
                "world_size": 3,
                "model": "mobilenetv2",
                "batch_size": 8,
                "num_samples": 100,
                "dataset": "cifar10",
                "split_block": split_block
            }
            
            self.kill_existing_processes()
            result = self.run_test(f"split_{split_name}", config)
            results.append(result)
            time.sleep(5)
        
        # Test 2: Pipelining comparison
        logger.info("\n" + "="*80)
        logger.info("TEST SUITE 2: Pipelining Comparison")
        logger.info("="*80)
        
        for use_pipeline in [False, True]:
            config = {
                "world_size": 3,
                "model": "mobilenetv2",
                "batch_size": 16,
                "num_samples": 200,
                "dataset": "cifar10",
                "split_block": 8,
                "use_pipelining": use_pipeline
            }
            
            test_name = "pipelined" if use_pipeline else "sequential"
            self.kill_existing_processes()
            result = self.run_test(test_name, config)
            results.append(result)
            time.sleep(5)
        
        # Save results
        self.save_results(results, f"comparison_results_{timestamp}.json")
        self.generate_report(results, timestamp)
        
        return results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save test results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {filename}")
    
    def generate_report(self, results: List[Dict], timestamp: str):
        """Generate a human-readable report."""
        report_file = f"test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("DISTRIBUTED INFERENCE TEST REPORT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("="*80 + "\n\n")
            
            # Group results by test type
            split_tests = [r for r in results if 'split_' in r.get('test_name', '')]
            pipeline_tests = [r for r in results if 'pipeline' in r.get('test_name', '')]
            
            # Split point comparison
            if split_tests:
                f.write("SPLIT POINT COMPARISON\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Split Type':<20} {'Block':<10} {'Throughput':<15} {'Accuracy':<10}\n")
                f.write("-"*40 + "\n")
                
                for test in split_tests:
                    if test['status'] == 'success':
                        split_type = test['test_name'].replace('split_', '')
                        block = test['config']['split_block']
                        throughput = test.get('throughput', 0)
                        accuracy = test.get('accuracy', 0)
                        f.write(f"{split_type:<20} {block:<10} {throughput:<15.2f} {accuracy:<10.2f}%\n")
                
                f.write("\n")
            
            # Pipeline comparison
            if pipeline_tests:
                f.write("PIPELINE COMPARISON\n")
                f.write("-"*40 + "\n")
                f.write(f"{'Mode':<20} {'Throughput':<15} {'Improvement':<15}\n")
                f.write("-"*40 + "\n")
                
                seq_throughput = 0
                for test in pipeline_tests:
                    if test['status'] == 'success':
                        mode = test['test_name']
                        throughput = test.get('throughput', 0)
                        
                        if mode == 'sequential':
                            seq_throughput = throughput
                            improvement = "baseline"
                        else:
                            improvement = f"{(throughput/seq_throughput - 1)*100:.1f}%" if seq_throughput > 0 else "N/A"
                        
                        f.write(f"{mode:<20} {throughput:<15.2f} {improvement:<15}\n")
        
        logger.info(f"Report saved to {report_file}")
    
    def interactive_menu(self):
        """Interactive menu for running tests."""
        while True:
            print("\n" + "="*60)
            print("DISTRIBUTED INFERENCE AUTOMATED RUNNER")
            print("="*60)
            print("1. Deploy code to Pis")
            print("2. Run quick test")
            print("3. Run standard test")
            print("4. Run performance test")
            print("5. Run comparison suite")
            print("6. Custom test")
            print("7. Kill all processes")
            print("0. Exit")
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '0':
                break
            elif choice == '1':
                self.deploy_to_pis()
            elif choice == '2':
                self.kill_existing_processes()
                config = {
                    "world_size": 3,
                    "model": "mobilenetv2",
                    **self.test_configs['quick'],
                    "dataset": "cifar10",
                    "split_block": 8
                }
                self.run_test("quick_test", config)
            elif choice == '3':
                self.kill_existing_processes()
                config = {
                    "world_size": 3,
                    "model": "mobilenetv2",
                    **self.test_configs['standard'],
                    "dataset": "cifar10",
                    "split_block": 8
                }
                self.run_test("standard_test", config)
            elif choice == '4':
                self.kill_existing_processes()
                config = {
                    "world_size": 3,
                    "model": "mobilenetv2",
                    **self.test_configs['performance'],
                    "dataset": "cifar10",
                    "split_block": 8,
                    "use_pipelining": True
                }
                self.run_test("performance_test", config)
            elif choice == '5':
                self.run_comparison_suite()
            elif choice == '6':
                # Custom test
                print("\nCustom Test Configuration")
                num_samples = int(input("Number of samples (default 100): ") or "100")
                batch_size = int(input("Batch size (default 8): ") or "8")
                split_block = int(input("Split block (default 8): ") or "8")
                use_pipeline = input("Use pipelining? (y/n, default n): ").lower() == 'y'
                
                self.kill_existing_processes()
                config = {
                    "world_size": 3,
                    "model": "mobilenetv2",
                    "batch_size": batch_size,
                    "num_samples": num_samples,
                    "dataset": "cifar10",
                    "split_block": split_block,
                    "use_pipelining": use_pipeline
                }
                self.run_test("custom_test", config)
            elif choice == '7':
                self.kill_existing_processes()
                print("All processes killed.")


def main():
    """Main entry point."""
    runner = AutomatedRunner()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "deploy":
            runner.deploy_to_pis(force_copy=True)
        elif sys.argv[1] == "quick":
            runner.kill_existing_processes()
            config = {
                "world_size": 3,
                "model": "mobilenetv2",
                **runner.test_configs['quick'],
                "dataset": "cifar10",
                "split_block": 8
            }
            runner.run_test("quick_test", config)
        elif sys.argv[1] == "compare":
            runner.run_comparison_suite()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python automated_runner.py [deploy|quick|compare]")
    else:
        runner.interactive_menu()


if __name__ == "__main__":
    main()