#!/usr/bin/env python
"""
Launcher script for distributed DeepGEMM operations.

This script provides a convenient way to launch distributed benchmarks or tests
across multiple GPUs on a single machine or across multiple machines.

Usage:
    python tools/launch_distributed.py --script benchmarks/benchmark_distributed.py --world-size 2
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch distributed DeepGEMM operations"
    )
    parser.add_argument(
        "--script", 
        type=str, 
        required=True,
        help="Path to the script to be launched in distributed mode"
    )
    parser.add_argument(
        "--world-size", 
        type=int, 
        default=2,
        help="Number of processes to launch (default: 2)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=12355,
        help="Port for distributed communication (default: 12355)"
    )
    parser.add_argument(
        "--node-rank", 
        type=int, 
        default=0,
        help="Rank of this node in multi-node setup (default: 0)"
    )
    parser.add_argument(
        "--master-addr", 
        type=str, 
        default="localhost",
        help="Master node address for multi-node setup (default: localhost)"
    )
    parser.add_argument(
        "--use-mpi", 
        action="store_true",
        help="Use MPI for distributed communication instead of Python multiprocessing"
    )
    parser.add_argument(
        "--backend", 
        type=str, 
        default="nccl",
        choices=["nccl", "gloo"],
        help="Backend for PyTorch distributed (default: nccl)"
    )
    parser.add_argument(
        "--script-args", 
        type=str, 
        default="",
        help="Additional arguments to pass to the script (use quotes, e.g. '--arg1 val1 --arg2 val2')"
    )
    
    return parser.parse_args()


def check_cuda_availability(world_size):
    """Check CUDA availability and warn if not enough GPUs."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, falling back to CPU mode")
        return False
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < world_size:
        print(f"WARNING: Requested {world_size} GPUs but only {available_gpus} are available")
        print(f"Reducing world_size to {available_gpus}")
        return available_gpus
    
    return world_size


def launch_with_multiprocessing(args, adjusted_world_size):
    """Launch script using Python multiprocessing."""
    # Set environment variables
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = str(args.port)
    
    # Construct command
    base_cmd = [
        sys.executable,
        "-m", "torch.distributed.launch",
        f"--nproc_per_node={adjusted_world_size}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.port}",
        f"--node_rank={args.node_rank}",
        f"--use_env",
        args.script
    ]
    
    # Add additional script arguments if provided
    if args.script_args:
        base_cmd.extend(args.script_args.split())
    
    # Launch the command
    print(f"Launching command: {' '.join(base_cmd)}")
    subprocess.run(base_cmd, check=True)


def launch_with_mpi(args, adjusted_world_size):
    """Launch script using MPI."""
    # Construct command
    base_cmd = [
        "mpirun",
        "-np", str(adjusted_world_size),
        "--allow-run-as-root",  # Add this if running as root
        sys.executable,
        args.script,
        f"--port={args.port}",
        f"--backend={args.backend}"
    ]
    
    # Add additional script arguments if provided
    if args.script_args:
        base_cmd.extend(args.script_args.split())
    
    # Launch the command
    print(f"Launching command: {' '.join(base_cmd)}")
    subprocess.run(base_cmd, check=True)


def create_virtual_env_if_needed():
    """Create virtual environment using uv if not already active."""
    # Check if we're already in a virtual environment
    if sys.prefix == sys.base_prefix:
        print("No active virtual environment detected. Creating one with uv...")
        
        venv_path = Path(".venv")
        if not venv_path.exists():
            subprocess.run(["uv", "venv", ".venv"], check=True)
        
        # Activate the virtual environment by modifying PATH and PYTHONPATH
        venv_bin = venv_path / "bin"
        os.environ["PATH"] = f"{venv_bin}:{os.environ['PATH']}"
        
        # Install requirements if needed
        if Path("requirements.txt").exists():
            subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"], check=True)
        
        print("Virtual environment created and activated using uv")
    else:
        print(f"Using existing virtual environment: {sys.prefix}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Import torch here to avoid import error if it's not installed
    try:
        import torch
    except ImportError:
        print("WARNING: PyTorch not found, attempting to install...")
        create_virtual_env_if_needed()
        subprocess.run(["uv", "pip", "install", "torch"], check=True)
        import torch
    
    # Create virtual environment if needed
    create_virtual_env_if_needed()
    
    # Check CUDA availability
    adjusted_world_size = check_cuda_availability(args.world_size)
    
    if args.use_mpi:
        try:
            import mpi4py
        except ImportError:
            print("mpi4py not found, installing...")
            subprocess.run(["uv", "pip", "install", "mpi4py"], check=True)
        launch_with_mpi(args, adjusted_world_size)
    else:
        launch_with_multiprocessing(args, adjusted_world_size)


if __name__ == "__main__":
    main() 