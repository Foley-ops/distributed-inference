#!/usr/bin/env python3
import torch
import torch.distributed as dist
import os
import socket

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# Check network
print("\nNetwork configuration:")
print(f"Hostname: {socket.gethostname()}")
print(f"MASTER_ADDR: {os.getenv('MASTER_ADDR', 'not set')}")
print(f"MASTER_PORT: {os.getenv('MASTER_PORT', 'not set')}")

# Try to bind to the port
try:
    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    test_socket.bind(('', 29501))
    print("Successfully bound to port 29501")
    test_socket.close()
except Exception as e:
    print(f"Failed to bind to port 29501: {e}")

# Check if we can resolve the master address
try:
    addr_info = socket.getaddrinfo('10.100.117.1', 29501)
    print(f"\nAddress resolution for 10.100.117.1:29501 successful")
    print(f"Address info: {addr_info[0]}")
except Exception as e:
    print(f"Failed to resolve address: {e}")

# Check network interfaces
import subprocess
result = subprocess.run(['ip', 'addr', 'show'], capture_output=True, text=True)
print("\nNetwork interfaces:")
for line in result.stdout.split('\n'):
    if 'inet ' in line and not '127.0.0.1' in line:
        print(line.strip())