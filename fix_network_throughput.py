#!/usr/bin/env python3
"""Fix network_throughput_mbps in existing consolidated_metrics.json files."""

import json
import sys

def calculate_network_throughput(metrics):
    """Calculate network throughput from existing metrics."""
    if (metrics.get('intermediate_data_size_bytes') and 
        metrics.get('network_time_s') and 
        metrics['network_time_s'] > 0):
        
        size_mb = metrics['intermediate_data_size_bytes'] / (1024 * 1024)
        size_mbits = size_mb * 8
        network_time_s = metrics['network_time_s']
        return size_mbits / network_time_s
    return None

def fix_consolidated_metrics(filename):
    """Fix network_throughput_mbps in consolidated metrics file."""
    
    # Read the file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Track if we made any changes
    modified = False
    
    # Process each result
    for split_id, split_results in data.get('results', {}).items():
        for run_id, run_data in split_results.items():
            avg_metrics = run_data.get('average_metrics_per_batch', {})
            
            # Only update if network_throughput_mbps is null but we have the data to calculate it
            if (avg_metrics.get('network_throughput_mbps') is None and
                avg_metrics.get('intermediate_data_size_bytes') is not None and
                avg_metrics.get('network_time_s') is not None):
                
                throughput = calculate_network_throughput(avg_metrics)
                if throughput is not None:
                    avg_metrics['network_throughput_mbps'] = round(throughput, 2)
                    modified = True
                    print(f"Fixed {split_id}/{run_id}: {throughput:.2f} Mbps")
    
    # Save if modified
    if modified:
        # Create backup
        import shutil
        backup_file = filename + '.backup'
        shutil.copy(filename, backup_file)
        print(f"\nCreated backup: {backup_file}")
        
        # Write updated file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated file: {filename}")
    else:
        print("No changes needed - all network_throughput_mbps values are already set or data is missing")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "/home/xfu601/Projects/distributed_inference/metrics/session_20250723_134418/consolidated_metrics.json"
    
    fix_consolidated_metrics(filename)