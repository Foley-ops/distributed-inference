#!/usr/bin/env python3
"""Fix end_to_end_latency_s to be the sum of part1 + part2 + network times."""

import json
import sys

def fix_consolidated_metrics(filename):
    """Fix end_to_end_latency_s in consolidated metrics file."""
    
    # Read the file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Track if we made any changes
    modified = False
    fixed_count = 0
    
    # Process each result
    for split_id, split_results in data.get('results', {}).items():
        for run_id, run_data in split_results.items():
            avg_metrics = run_data.get('average_metrics_per_batch', {})
            
            # Get the component times
            part1_time = avg_metrics.get('part1_inference_time_s')
            part2_time = avg_metrics.get('part2_inference_time_s')
            network_time = avg_metrics.get('network_time_s')
            
            # Calculate correct end-to-end latency if we have all components
            if part1_time is not None and part2_time is not None and network_time is not None:
                correct_latency = part1_time + part2_time + network_time
                old_latency = avg_metrics.get('end_to_end_latency_s', 0)
                
                # Update the value
                avg_metrics['end_to_end_latency_s'] = round(correct_latency, 6)
                modified = True
                fixed_count += 1
                
                print(f"Fixed {split_id}/{run_id}: {old_latency:.6f} -> {correct_latency:.6f} "
                      f"(d1={part1_time:.3f} + d2={part2_time:.3f} + net={network_time:.3f})")
    
    # Save if modified
    if modified:
        # Create backup
        import shutil
        backup_file = filename + '.backup2'
        shutil.copy(filename, backup_file)
        print(f"\nCreated backup: {backup_file}")
        
        # Write updated file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Updated file: {filename}")
        print(f"Fixed {fixed_count} end-to-end latency values")
    else:
        print("No changes needed - missing component times")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "/home/xfu601/Projects/distributed_inference/metrics/session_20250723_134418/consolidated_metrics.json"
    
    fix_consolidated_metrics(filename)