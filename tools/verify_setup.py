#!/usr/bin/env python
"""
Distributed Setup Verification

This script verifies that the distributed setup is working correctly by
performing a simple distributed GEMM operation and checking the results.

Usage:
    python -m tools.verify_setup --config-path config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# Add parent directory to sys.path to allow importing deep_gemm
sys.path.insert(0, str(Path(__file__).parent.parent))

from deep_gemm.distributed import (
    DistributedGEMM,
    init_from_config,
    get_rank,
    get_world_size,
    ShardingStrategy,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("verify_setup")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify distributed setup")
    
    # Configuration arguments
    parser.add_argument("--config-path", type=str, default=None,
                        help="Path to configuration file")
    
    # Override arguments
    parser.add_argument("--world-size", type=int, default=None,
                        help="Number of processes")
    parser.add_argument("--master-addr", type=str, default=None,
                        help="Master node address")
    parser.add_argument("--master-port", type=int, default=None,
                        help="Master node port")
    parser.add_argument("--backend", type=str, default=None,
                        help="Distributed backend")
    
    # Test parameters
    parser.add_argument("--matrix-size", type=int, default=1024,
                        help="Size of matrices to use for testing")
    parser.add_argument("--verify-result", action="store_true",
                        help="Verify result against PyTorch reference")
    parser.add_argument("--test-all-strategies", action="store_true",
                        help="Test all sharding strategies")
    
    return parser.parse_args()


def verify_distributed_gemm(
    dgemm: DistributedGEMM,
    matrix_size: int,
    verify_result: bool = True
) -> bool:
    """
    Verify that distributed GEMM works correctly.
    
    Args:
        dgemm: DistributedGEMM instance
        matrix_size: Size of matrices to use for testing
        verify_result: Whether to verify result against PyTorch reference
        
    Returns:
        True if verification succeeded, False otherwise
    """
    rank = dgemm.rank
    world_size = dgemm.world_size
    strategy = dgemm.config.strategy
    
    try:
        # Create matrices based on sharding strategy
        if strategy == ShardingStrategy.ROW_PARALLEL:
            local_rows = matrix_size // world_size
            if rank == world_size - 1:
                local_rows = matrix_size - (world_size - 1) * local_rows
                
            A = torch.randn(local_rows, matrix_size, dtype=torch.float16, device="cuda")
            B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device="cuda")
            
        elif strategy == ShardingStrategy.COLUMN_PARALLEL:
            local_cols = matrix_size // world_size
            if rank == world_size - 1:
                local_cols = matrix_size - (world_size - 1) * local_cols
                
            A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device="cuda")
            B = torch.randn(matrix_size, local_cols, dtype=torch.float16, device="cuda")
            
        else:  # FULLY_SHARDED or default
            # For simplicity, we'll use the same matrix sizes for all ranks
            A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device="cuda")
            B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device="cuda")
        
        # Perform distributed GEMM
        logger.info(f"Rank {rank}: Running distributed GEMM with matrices of shape {A.shape} x {B.shape}")
        
        result = dgemm.gemm_fp16_fp32_nt(A, B)
        
        logger.info(f"Rank {rank}: Result shape: {result.shape}")
        
        # Verify result against PyTorch reference
        if verify_result and rank == 0:
            logger.info("Verifying result against PyTorch reference")
            
            # Gather complete matrices for reference computation
            if strategy == ShardingStrategy.ROW_PARALLEL:
                full_A = dgemm.all_gather(A, dim=0)
                full_B = B
            elif strategy == ShardingStrategy.COLUMN_PARALLEL:
                full_A = A
                full_B = dgemm.all_gather(B, dim=1)
            else:
                full_A = A
                full_B = B
            
            # Compute reference result
            reference = torch.matmul(full_A.float(), full_B.float().t())
            
            # Compare results
            if strategy == ShardingStrategy.ROW_PARALLEL:
                local_result = reference[:result.shape[0]]
                max_diff = torch.max(torch.abs(result - local_result)).item()
            elif strategy == ShardingStrategy.COLUMN_PARALLEL:
                local_result = reference[:, :result.shape[1]]
                max_diff = torch.max(torch.abs(result - local_result)).item()
            else:
                max_diff = torch.max(torch.abs(result - reference)).item()
            
            logger.info(f"Maximum difference from PyTorch reference: {max_diff:.6f}")
            
            if max_diff > 1e-3:
                logger.warning("Result differs significantly from PyTorch reference")
                return False
            
        return True
    
    except Exception as e:
        logger.error(f"Error verifying distributed GEMM: {e}", exc_info=True)
        return False


def verify_all_strategies(config, matrix_size: int, verify_result: bool) -> bool:
    """
    Verify all sharding strategies.
    
    Args:
        config: Distributed configuration
        matrix_size: Size of matrices to use for testing
        verify_result: Whether to verify result against PyTorch reference
        
    Returns:
        True if all strategies passed verification, False otherwise
    """
    strategies = [
        ShardingStrategy.ROW_PARALLEL,
        ShardingStrategy.COLUMN_PARALLEL,
    ]
    
    # Only test FULLY_SHARDED if world_size is a perfect square
    world_size = get_world_size()
    if (int(world_size ** 0.5) ** 2) == world_size and world_size > 1:
        strategies.append(ShardingStrategy.FULLY_SHARDED)
    
    all_passed = True
    
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")
        
        # Update strategy in configuration
        config.strategy = strategy
        
        # Create a new DistributedGEMM instance with this strategy
        with DistributedGEMM(config) as dgemm:
            # Verify distributed GEMM
            passed = verify_distributed_gemm(dgemm, matrix_size, verify_result)
            
            if not passed:
                logger.error(f"Verification failed for strategy: {strategy}")
                all_passed = False
    
    return all_passed


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
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
        
        # Initialize from config
        config = init_from_config(args.config_path, override_args)
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Verification requires CUDA.")
            logger.error("Skipping verification.")
            sys.exit(2)
            
        # Get local rank for setting device
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        
        if args.test_all_strategies:
            # Test all sharding strategies
            all_passed = verify_all_strategies(
                config, 
                args.matrix_size, 
                args.verify_result
            )
            
            if all_passed:
                logger.info("All strategies passed verification")
                sys.exit(0)
            else:
                logger.error("Some strategies failed verification")
                sys.exit(1)
        else:
            # Test only the configured strategy
            with DistributedGEMM(config) as dgemm:
                # Print information about the distributed setup
                if dgemm.rank == 0:
                    logger.info(f"Running with {dgemm.world_size} processes")
                    logger.info(f"Sharding strategy: {dgemm.config.strategy}")
                    logger.info(f"Backend: {dgemm.config.backend}")
                    logger.info(f"Device: {torch.cuda.get_device_name()}")
                
                # Verify distributed GEMM
                passed = verify_distributed_gemm(
                    dgemm, 
                    args.matrix_size, 
                    args.verify_result
                )
                
                if passed:
                    logger.info("Verification passed")
                    sys.exit(0)
                else:
                    logger.error("Verification failed")
                    sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error during verification: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main() 