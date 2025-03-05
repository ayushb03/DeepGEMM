#!/usr/bin/env python
"""
Distributed DeepGEMM Launcher

This script launches distributed DeepGEMM operations across multiple nodes and processes.
It supports various distributed backends (PyTorch, MPI) and handles environment setup.

Usage:
    python -m tools.launch_distributed --nnodes=2 --nproc_per_node=4 \
        --master_addr=localhost --master_port=12355 \
        --module examples.production_deployment \
        --args="--matrix-size 8192 --precision fp16"

For MPI launcher:
    python -m tools.launch_distributed --backend=mpi --nnodes=2 --nproc_per_node=4 \
        --module examples.production_deployment \
        --args="--matrix-size 8192 --precision fp16"
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("launch_distributed")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed DeepGEMM Launcher")
    
    # Launcher arguments
    parser.add_argument("--backend", type=str, default="pytorch",
                        choices=["pytorch", "mpi"],
                        help="Launcher backend")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--node_rank", type=int, default=0,
                        help="Rank of this node")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="Number of processes per node")
    parser.add_argument("--master_addr", type=str, default="localhost",
                        help="Master node address")
    parser.add_argument("--master_port", type=int, default=12355,
                        help="Master node port")
    
    # Python environment arguments
    parser.add_argument("--python_path", type=str, default=sys.executable,
                        help="Path to Python executable")
    parser.add_argument("--use_venv", action="store_true",
                        help="Use virtual environment")
    parser.add_argument("--venv_path", type=str, default=".venv",
                        help="Path to virtual environment")
    parser.add_argument("--use_uv", action="store_true",
                        help="Use uv package manager")
    
    # Script arguments
    parser.add_argument("--module", type=str, required=True,
                        help="Python module to run")
    parser.add_argument("--args", type=str, default="",
                        help="Arguments to pass to the module")
    
    return parser.parse_args()


def setup_environment(args: argparse.Namespace) -> Dict[str, str]:
    """
    Set up environment variables for distributed training.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary of environment variables
    """
    env = os.environ.copy()
    
    # Add distributed environment variables
    env["MASTER_ADDR"] = args.master_addr
    env["MASTER_PORT"] = str(args.master_port)
    env["WORLD_SIZE"] = str(args.nnodes * args.nproc_per_node)
    env["NODE_RANK"] = str(args.node_rank)
    
    # Add PYTHONPATH to include the current project
    project_root = Path(__file__).parent.parent
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{python_path}" if python_path else str(project_root)
    
    return env


def get_python_command(args: argparse.Namespace) -> List[str]:
    """
    Get the Python command to run.
    
    Args:
        args: Command line arguments
        
    Returns:
        List of command parts
    """
    if args.use_venv:
        if args.use_uv:
            return ["uv", "run", "--python", args.python_path, "-m"]
        else:
            return [os.path.join(args.venv_path, "bin", "python"), "-m"]
    else:
        return [args.python_path, "-m"]


def launch_pytorch(args: argparse.Namespace, env: Dict[str, str]) -> int:
    """
    Launch distributed training using PyTorch's launcher.
    
    Args:
        args: Command line arguments
        env: Environment variables
        
    Returns:
        Exit code
    """
    cmd = [
        *get_python_command(args),
        "torch.distributed.launch",
        f"--nnodes={args.nnodes}",
        f"--node_rank={args.node_rank}",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_addr={args.master_addr}",
        f"--master_port={args.master_port}",
        "-m", args.module
    ]
    
    if args.args:
        cmd.extend(args.args.split())
    
    logger.info(f"Launching with PyTorch: {' '.join(cmd)}")
    
    # Launch the process
    try:
        process = subprocess.Popen(cmd, env=env)
        
        # Handle signals
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, terminating process...")
            process.terminate()
            process.wait()
            sys.exit(128 + sig)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process to complete
        process.wait()
        return process.returncode
    
    except Exception as e:
        logger.error(f"Error launching with PyTorch: {e}")
        return 1


def launch_mpi(args: argparse.Namespace, env: Dict[str, str]) -> int:
    """
    Launch distributed training using MPI.
    
    Args:
        args: Command line arguments
        env: Environment variables
        
    Returns:
        Exit code
    """
    cmd = [
        "mpirun",
        "-np", str(args.nnodes * args.nproc_per_node),
        "--allow-run-as-root",
        "-x", "MASTER_ADDR",
        "-x", "MASTER_PORT",
        "-x", "WORLD_SIZE",
        "-x", "PYTHONPATH",
    ]
    
    # Add Python command
    python_cmd = get_python_command(args)
    cmd.extend(python_cmd)
    
    # Add module and arguments
    cmd.extend([args.module])
    if args.args:
        cmd.extend(args.args.split())
    
    logger.info(f"Launching with MPI: {' '.join(cmd)}")
    
    # Launch the process
    try:
        process = subprocess.Popen(cmd, env=env)
        
        # Handle signals
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, terminating process...")
            process.terminate()
            process.wait()
            sys.exit(128 + sig)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process to complete
        process.wait()
        return process.returncode
    
    except Exception as e:
        logger.error(f"Error launching with MPI: {e}")
        return 1


def check_prerequisites(args: argparse.Namespace) -> bool:
    """
    Check prerequisites for launching distributed training.
    
    Args:
        args: Command line arguments
        
    Returns:
        True if prerequisites are met, False otherwise
    """
    # Check if virtual environment exists
    if args.use_venv:
        venv_path = Path(args.venv_path)
        if not venv_path.exists():
            logger.error(f"Virtual environment not found: {venv_path}")
            logger.info("Creating virtual environment...")
            
            if args.use_uv:
                try:
                    cmd = ["uv", "venv", str(venv_path)]
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create virtual environment: {e}")
                    return False
            else:
                try:
                    cmd = [sys.executable, "-m", "venv", str(venv_path)]
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to create virtual environment: {e}")
                    return False
    
    # Check if MPI is available
    if args.backend == "mpi":
        try:
            subprocess.run(["mpirun", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("MPI not found. Please install MPI (e.g., Open MPI) before using the MPI backend.")
            return False
    
    return True


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check prerequisites
    if not check_prerequisites(args):
        sys.exit(1)
    
    # Set up environment
    env = setup_environment(args)
    
    # Launch training
    if args.backend == "pytorch":
        exit_code = launch_pytorch(args, env)
    elif args.backend == "mpi":
        exit_code = launch_mpi(args, env)
    else:
        logger.error(f"Unsupported backend: {args.backend}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 