#!/usr/bin/env python3
import torch.distributed.rpc as rpc
import os
import time
import sys

# Force environment variables
os.environ['MASTER_ADDR'] = '10.100.117.1'
os.environ['MASTER_PORT'] = '29501'

if len(sys.argv) > 1 and sys.argv[1] == "worker":
    # Worker mode
    print("Starting as worker...")
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    time.sleep(5)  # Wait for master
    
    try:
        rpc.init_rpc(
            "worker1",
            rank=1,
            world_size=2
        )
        print("Worker initialized!")
        time.sleep(10)
        rpc.shutdown()
    except Exception as e:
        print(f"Worker error: {e}")
else:
    # Master mode
    print("Starting as master...")
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
    
    try:
        rpc.init_rpc(
            "master",
            rank=0,
            world_size=2
        )
        print("Master initialized!")
        time.sleep(15)
        rpc.shutdown()
    except Exception as e:
        print(f"Master error: {e}")