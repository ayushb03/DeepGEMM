"""
Unit tests for distributed multi-GPU GEMM operations.
"""

import os
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from deep_gemm.distributed import (
    ShardingStrategy,
    distributed_gemm_fp16_fp32_nt,
    distributed_gemm_bf16_fp32_nt,
    distribute_gemm,
    get_rank,
    get_world_size,
    initialize_process_group
)


def init_distributed_test_env(rank, world_size, fn, args=(), port=12355):
    """
    Initialize distributed test environment and run test function.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        fn: Test function to run
        args: Arguments to pass to test function
        port: Port for distributed communication
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    
    # Run test function
    fn(rank, world_size, *args)
    
    # Cleanup
    dist.destroy_process_group()


class TestDistributedGEMM(unittest.TestCase):
    """Test cases for distributed GEMM operations."""
    
    def _test_distribute_gemm_fp16(self, rank, world_size):
        """Test distributed GEMM with FP16 inputs."""
        # Create test matrices
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        m, n, k = 64, 64, 64
        
        # Use the same seed across all ranks for deterministic results
        torch.manual_seed(42)
        
        # Create matrices - same on all ranks for this test
        a = torch.randn((m, k), dtype=torch.float16, device=device)
        b = torch.randn((n, k), dtype=torch.float16, device=device)
        
        # Create expected output using PyTorch's matmul for reference
        expected = torch.matmul(a, b.transpose(0, 1))
        
        # Get shard of matrix
        strategies = [
            ShardingStrategy.ROW_PARALLEL,
            ShardingStrategy.COLUMN_PARALLEL,
            ShardingStrategy.FULLY_SHARDED,
        ]
        
        for strategy in strategies:
            # Compute distributed result
            distributed_result = distributed_gemm_fp16_fp32_nt(a, b, strategy=strategy)
            
            # Gather results from all ranks for comparison
            if strategy == ShardingStrategy.ROW_PARALLEL:
                # For row parallel, each rank has a horizontal slice of the result
                # We need to gather all slices to compare with expected
                all_results = [torch.zeros_like(distributed_result) for _ in range(world_size)]
                dist.all_gather(all_results, distributed_result)
                
                # Only check on rank 0
                if rank == 0:
                    # Concatenate results from all ranks
                    gathered_result = torch.cat(all_results, dim=0)
                    
                    # Compare with expected
                    # We use a relative tolerance due to floating point precision issues
                    rtol = 1e-3 if device.startswith("cuda") else 1e-2
                    torch.testing.assert_close(gathered_result[:m], expected, rtol=rtol)
                    
            elif strategy == ShardingStrategy.COLUMN_PARALLEL:
                # For column parallel, each rank has a vertical slice of the result
                # We need to gather all slices to compare with expected
                all_results = [torch.zeros_like(distributed_result) for _ in range(world_size)]
                dist.all_gather(all_results, distributed_result)
                
                # Only check on rank 0
                if rank == 0:
                    # Concatenate results from all ranks
                    gathered_result = torch.cat(all_results, dim=1)
                    
                    # Compare with expected
                    rtol = 1e-3 if device.startswith("cuda") else 1e-2
                    torch.testing.assert_close(gathered_result[:, :n], expected, rtol=rtol)
            
            # Synchronize before next test
            dist.barrier()
    
    def _test_distribute_gemm_bf16(self, rank, world_size):
        """Test distributed GEMM with BF16 inputs."""
        # Skip if BF16 is not supported
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            if rank == 0:
                print("BF16 not supported on this device, skipping test")
            return
        
        # Create test matrices
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        m, n, k = 64, 64, 64
        
        # Use the same seed across all ranks for deterministic results
        torch.manual_seed(42)
        
        # Create matrices - same on all ranks for this test
        a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
        b = torch.randn((n, k), dtype=torch.bfloat16, device=device)
        
        # Create expected output using PyTorch's matmul for reference
        expected = torch.matmul(a, b.transpose(0, 1))
        
        # Test row parallel strategy
        distributed_result = distributed_gemm_bf16_fp32_nt(
            a, b, strategy=ShardingStrategy.ROW_PARALLEL
        )
        
        # Gather results from all ranks for comparison
        all_results = [torch.zeros_like(distributed_result) for _ in range(world_size)]
        dist.all_gather(all_results, distributed_result)
        
        # Only check on rank 0
        if rank == 0:
            # Concatenate results from all ranks
            gathered_result = torch.cat(all_results, dim=0)
            
            # Compare with expected (using a larger tolerance for BF16)
            rtol = 1e-2 if device.startswith("cuda") else 5e-2
            torch.testing.assert_close(gathered_result[:m], expected, rtol=rtol)
    
    def _test_world_size_and_rank(self, rank, world_size):
        """Test the world size and rank utilities."""
        # Test with initialized process group
        self.assertEqual(get_world_size(), world_size)
        self.assertEqual(get_rank(), rank)
    
    def test_distribute_gemm_fp16(self):
        """Run FP16 distributed GEMM test in multiple processes."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Get world size (number of GPUs)
        world_size = min(torch.cuda.device_count(), 2)  # Use at most 2 GPUs
        
        # Spawn processes
        mp.spawn(
            init_distributed_test_env,
            args=(world_size, self._test_distribute_gemm_fp16),
            nprocs=world_size,
            join=True
        )
    
    def test_distribute_gemm_bf16(self):
        """Run BF16 distributed GEMM test in multiple processes."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Get world size (number of GPUs)
        world_size = min(torch.cuda.device_count(), 2)  # Use at most 2 GPUs
        
        # Spawn processes
        mp.spawn(
            init_distributed_test_env,
            args=(world_size, self._test_distribute_gemm_bf16),
            nprocs=world_size,
            join=True
        )
    
    def test_world_size_and_rank(self):
        """Run world size and rank test in multiple processes."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Get world size (number of GPUs)
        world_size = min(torch.cuda.device_count(), 2)  # Use at most 2 GPUs
        
        # Spawn processes
        mp.spawn(
            init_distributed_test_env,
            args=(world_size, self._test_world_size_and_rank),
            nprocs=world_size,
            join=True
        )
    
    def test_initialize_process_group(self):
        """Test initialize_process_group function."""
        # Mock dist.is_initialized
        with mock.patch('torch.distributed.is_initialized', return_value=False), \
             mock.patch('torch.distributed.init_process_group') as mock_init:
            
            # Call our initialize function
            initialize_process_group(backend="test_backend")
            
            # Check that init_process_group was called with correct backend
            mock_init.assert_called_once_with(backend="test_backend")
            
        # Case where it's already initialized
        with mock.patch('torch.distributed.is_initialized', return_value=True), \
             mock.patch('torch.distributed.init_process_group') as mock_init:
            
            # Call our initialize function
            initialize_process_group()
            
            # Check that init_process_group was not called
            mock_init.assert_not_called()


if __name__ == "__main__":
    unittest.main() 