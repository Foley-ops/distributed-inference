#!/usr/bin/env python3
import torch.distributed.rpc as rpc
import os
import sys
import time

def test_master():
    print("Starting master RPC test...")
    os.environ['MASTER_ADDR'] = '10.100.117.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp6s0'
    os.environ['TP_SOCKET_IFNAME'] = 'enp6s0'
    
    print(f"Environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    print(f"GLOO_SOCKET_IFNAME={os.environ['GLOO_SOCKET_IFNAME']}")
    
    try:
        print("Calling rpc.init_rpc for master...")
        rpc.init_rpc(
            "master",
            rank=0,
            world_size=2,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rpc_timeout=30
            )
        )
        print("Master RPC initialized successfully!")
        
        # Keep alive for 60 seconds
        print("Master waiting for 60 seconds...")
        time.sleep(60)
        
        rpc.shutdown()
        print("Master shutdown complete")
        
    except Exception as e:
        print(f"Master RPC failed: {e}")
        import traceback
        traceback.print_exc()

def test_worker():
    print("Starting worker RPC test...")
    os.environ['MASTER_ADDR'] = '10.100.117.1'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
    os.environ['TP_SOCKET_IFNAME'] = 'eth0'
    
    print(f"Environment: MASTER_ADDR={os.environ['MASTER_ADDR']}, MASTER_PORT={os.environ['MASTER_PORT']}")
    print(f"GLOO_SOCKET_IFNAME={os.environ['GLOO_SOCKET_IFNAME']}")
    
    # Wait a bit for master to start
    print("Waiting 5 seconds for master to initialize...")
    time.sleep(5)
    
    try:
        print("Calling rpc.init_rpc for worker1...")
        rpc.init_rpc(
            "worker1",
            rank=1,
            world_size=2,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
                rpc_timeout=30
            )
        )
        print("Worker RPC initialized successfully!")
        
        # Try to communicate with master
        master_rref = rpc.remote("master", lambda: "Hello from worker!")
        result = master_rref.to_here()
        print(f"Communication test result: {result}")
        
        rpc.shutdown()
        print("Worker shutdown complete")
        
    except Exception as e:
        print(f"Worker RPC failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_rpc.py [master|worker]")
        sys.exit(1)
    
    role = sys.argv[1]
    if role == "master":
        test_master()
    elif role == "worker":
        test_worker()
    else:
        print(f"Unknown role: {role}")