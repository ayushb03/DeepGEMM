#!/usr/bin/env python
"""
Production Deployment Example for DeepGEMM

This example demonstrates how to use the distributed GEMM operations
in a production environment with proper error handling, logging,
and configuration management.

Usage:
    python -m examples.production_deployment --config-path config.yaml --world-size 4
    
For multi-node deployment:
    # On the master node
    python -m examples.production_deployment --config-path config.yaml --world-size 8 --node-rank 0 --nnodes 2
    
    # On the worker node
    python -m examples.production_deployment --config-path config.yaml --world-size 8 --node-rank 1 --nnodes 2 --master-addr <master-ip>
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist

# Add parent directory to sys.path to allow importing deep_gemm
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_gemm.distributed import (
    DistributedGEMM, 
    DistributedConfig,
    ShardingStrategy,
    init_from_config
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("production_example")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Production deployment example for distributed GEMM")
    
    # Configuration arguments
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to configuration file")
    
    # Distributed arguments
    parser.add_argument("--world-size", type=int, default=None,
                        help="Total number of distributed processes")
    parser.add_argument("--nnodes", type=int, default=1,
                        help="Number of nodes")
    parser.add_argument("--node-rank", type=int, default=0,
                        help="Rank of this node")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Master node address")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Master node port")
    parser.add_argument("--backend", type=str, default=None,
                        help="Distributed backend (nccl, gloo, etc.)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["row", "column", "fully_sharded"],
                        help="Sharding strategy")
    
    # Matrix arguments
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Size of square matrices for benchmarking")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "bf16"],
                        help="Data precision for GEMM operation")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations")
    parser.add_argument("--iterations", type=int, default=10,
                        help="Number of benchmark iterations")
    
    return parser.parse_args()


def create_matrices(
    size: int, 
    strategy: ShardingStrategy,
    rank: int, 
    world_size: int,
    precision: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create input matrices based on sharding strategy.
    
    Args:
        size: Matrix size (for square matrices)
        strategy: Sharding strategy
        rank: Current process rank
        world_size: Total number of processes
        precision: Data precision ("fp16" or "bf16")
        
    Returns:
        Tuple of input matrices (A, B)
    """
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    
    if strategy == ShardingStrategy.ROW_PARALLEL:
        # For row parallel, shard matrix A by rows
        local_rows = size // world_size
        if rank == world_size - 1:
            # Last rank may get extra rows
            local_rows = size - (world_size - 1) * local_rows
            
        A = torch.randn(local_rows, size, dtype=dtype, device="cuda")
        B = torch.randn(size, size, dtype=dtype, device="cuda")
        
    elif strategy == ShardingStrategy.COLUMN_PARALLEL:
        # For column parallel, shard matrix B by columns
        local_cols = size // world_size
        if rank == world_size - 1:
            # Last rank may get extra columns
            local_cols = size - (world_size - 1) * local_cols
            
        A = torch.randn(size, size, dtype=dtype, device="cuda")
        B = torch.randn(size, local_cols, dtype=dtype, device="cuda")
        
    elif strategy == ShardingStrategy.FULLY_SHARDED:
        # For fully sharded, shard both matrices
        local_size = int(size / (world_size ** 0.5))
        grid_size = int(world_size ** 0.5)
        row_rank = rank // grid_size
        col_rank = rank % grid_size
        
        row_start = row_rank * local_size
        col_start = col_rank * local_size
        
        # Handle edge cases
        if row_rank == grid_size - 1:
            local_rows = size - row_start
        else:
            local_rows = local_size
            
        if col_rank == grid_size - 1:
            local_cols = size - col_start
        else:
            local_cols = local_size
            
        A = torch.randn(local_rows, size, dtype=dtype, device="cuda")
        B = torch.randn(size, local_cols, dtype=dtype, device="cuda")
    
    else:
        # No sharding (for single device)
        A = torch.randn(size, size, dtype=dtype, device="cuda")
        B = torch.randn(size, size, dtype=dtype, device="cuda")
        
    return A, B


def benchmark_gemm(
    distributed_gemm: DistributedGEMM,
    size: int,
    precision: str,
    warmup: int = 5,
    iterations: int = 10
) -> float:
    """
    Benchmark distributed GEMM operations.
    
    Args:
        distributed_gemm: DistributedGEMM instance
        size: Matrix size (for square matrices)
        precision: Data precision ("fp16" or "bf16")
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        
    Returns:
        Average execution time in milliseconds
    """
    strategy = distributed_gemm.config.strategy
    rank = distributed_gemm.rank
    world_size = distributed_gemm.world_size
    
    # Create matrices based on sharding strategy
    A, B = create_matrices(size, strategy, rank, world_size, precision)
    
    logger.info(f"Running benchmark with matrices of size {A.shape} x {B.shape}")
    
    # Warmup
    for _ in range(warmup):
        if precision == "fp16":
            result = distributed_gemm.gemm_fp16_fp32_nt(A, B)
        else:
            result = distributed_gemm.gemm_bf16_fp32_nt(A, B)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    for _ in range(iterations):
        if precision == "fp16":
            result = distributed_gemm.gemm_fp16_fp32_nt(A, B)
        else:
            result = distributed_gemm.gemm_bf16_fp32_nt(A, B)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate average time
    avg_time = (end_time - start_time) * 1000 / iterations
    
    # Only rank 0 prints the result shape
    if rank == 0:
        logger.info(f"Result shape: {result.shape}")
        logger.info(f"Average execution time: {avg_time:.2f} ms")
    
    return avg_time


def run_with_error_handling(args: argparse.Namespace) -> int:
    """
    Run the example with proper error handling.
    
    Args:
        args: Command line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. This example requires CUDA.")
            return 3
            
        # Create override args from command line
        override_args = {}
        if args.world_size is not None:
            override_args["world_size"] = args.world_size
        if args.master_addr is not None:
            override_args["master_addr"] = args.master_addr
        if args.master_port is not None:
            override_args["master_port"] = args.master_port
        if args.backend is not None:
            override_args["backend"] = args.backend
        if args.strategy is not None:
            override_args["strategy"] = args.strategy
            
        # Initialize from config
        config = init_from_config(args.config_path, override_args)
        
        # Create DistributedGEMM instance as context manager
        with DistributedGEMM(config) as dgemm:
            # Get local rank for setting device
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            
            # Print information about the distributed setup
            if dgemm.rank == 0:
                logger.info(f"Running with {dgemm.world_size} processes")
                logger.info(f"Sharding strategy: {dgemm.config.strategy}")
                logger.info(f"Backend: {dgemm.config.backend}")
                logger.info(f"Device: {torch.cuda.get_device_name()}")
                
            # Benchmark GEMM operations
            benchmark_gemm(
                dgemm,
                args.matrix_size,
                args.precision,
                args.warmup,
                args.iterations
            )
            
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    
    except ValueError as e:
        logger.error(f"Invalid argument: {e}")
        return 2
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 3
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 4


def main():
    """Main entry point."""
    args = parse_args()
    exit_code = run_with_error_handling(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main() 