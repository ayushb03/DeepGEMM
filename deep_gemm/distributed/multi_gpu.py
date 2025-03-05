"""
Multi-GPU GEMM operations for DeepGEMM.

This module provides optimized GEMM kernels distributed across multiple GPUs
for larger matrix multiplications and improved performance.
"""

import enum
import math
from typing import Optional, Tuple, Union, List

import torch
import torch.distributed as dist

from ..mixed_precision import gemm_fp16_fp32_nt, gemm_bf16_fp32_nt
from ..utils import bench


class ShardingStrategy(enum.Enum):
    """
    Enum for different matrix sharding strategies.
    
    - ROW_PARALLEL: Shard the first input matrix along rows (M dimension).
      Each GPU computes a portion of the output matrix rows.
    
    - COLUMN_PARALLEL: Shard the second input matrix along columns (N dimension).
      Each GPU computes a portion of the output matrix columns.
    
    - FULLY_SHARDED: Shard both matrices (M and N dimensions).
      Each GPU computes a portion of the output matrix.
    """
    ROW_PARALLEL = 'row'
    COLUMN_PARALLEL = 'column'
    FULLY_SHARDED = 'fully_sharded'


def initialize_process_group(backend: str = 'nccl'):
    """
    Initialize the distributed process group if not already initialized.
    
    Args:
        backend: The backend to use for distributed operations ('nccl', 'gloo', etc.)
    """
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)


def get_world_size() -> int:
    """
    Get the number of processes in the current process group.
    
    Returns:
        Number of processes in the process group
    """
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """
    Get the rank of the current process in the process group.
    
    Returns:
        Rank of the current process
    """
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def distribute_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    gemm_fn,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    strategy: ShardingStrategy = ShardingStrategy.ROW_PARALLEL,
    sync: bool = True
) -> torch.Tensor:
    """
    Generic function to distribute any GEMM operation across multiple GPUs.
    
    Args:
        a: First input tensor
        b: Second input tensor
        gemm_fn: The GEMM function to use (e.g., gemm_fp16_fp32_nt)
        out: Output tensor, will be created if not provided
        alpha: Scalar multiplier for the product of input tensors
        beta: Scalar multiplier for the output
        strategy: Sharding strategy to use
        sync: Whether to synchronize at the end of the operation
    
    Returns:
        The result tensor
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # If not distributed or only one process, perform normal GEMM
    if world_size == 1:
        return gemm_fn(a, b, out=out, alpha=alpha, beta=beta)
    
    # Get dimensions
    m, k = a.shape
    n, k2 = b.shape
    
    assert k == k2, f"Inner dimensions must match: got {k} and {k2}"
    
    if strategy == ShardingStrategy.ROW_PARALLEL:
        # Shard matrix A along M dimension
        chunk_size = math.ceil(m / world_size)
        start_row = rank * chunk_size
        end_row = min((rank + 1) * chunk_size, m)
        
        if start_row >= m:
            # This rank has no work to do
            if out is not None:
                # Still need to synchronize for consistency
                dist.barrier()
                return out
            else:
                # Return empty tensor for consistency
                result = torch.empty((0, n), dtype=a.dtype, device=a.device)
                dist.barrier()
                return result
        
        # Extract my shard of matrix A
        a_shard = a[start_row:end_row]
        
        # Create output shard or use provided output
        if out is None:
            out_shard = torch.empty((end_row - start_row, n), 
                                  dtype=torch.float32, 
                                  device=a.device)
        else:
            out_shard = out[start_row:end_row]
        
        # Perform GEMM on my shard
        result_shard = gemm_fn(a_shard, b, out=out_shard, alpha=alpha, beta=beta)
        
        # Synchronize when required
        if sync:
            dist.barrier()
        
        return result_shard
    
    elif strategy == ShardingStrategy.COLUMN_PARALLEL:
        # Shard matrix B along N dimension
        chunk_size = math.ceil(n / world_size)
        start_col = rank * chunk_size
        end_col = min((rank + 1) * chunk_size, n)
        
        if start_col >= n:
            # This rank has no work to do
            if out is not None:
                # Still need to synchronize for consistency
                dist.barrier()
                return out
            else:
                # Return empty tensor for consistency
                result = torch.empty((m, 0), dtype=a.dtype, device=a.device)
                dist.barrier()
                return result
        
        # Extract my shard of matrix B
        b_shard = b[start_col:end_col]
        
        # Create output shard or use provided output
        if out is None:
            out_shard = torch.empty((m, end_col - start_col), 
                                  dtype=torch.float32, 
                                  device=a.device)
        else:
            out_shard = out[:, start_col:end_col]
        
        # Perform GEMM on my shard
        result_shard = gemm_fn(a, b_shard, out=out_shard, alpha=alpha, beta=beta)
        
        # Synchronize when required
        if sync:
            dist.barrier()
        
        return result_shard
    
    elif strategy == ShardingStrategy.FULLY_SHARDED:
        # For fully sharded, we need a 2D grid of processes
        # This is a simplified implementation using a 1D process group
        # In a real-world implementation, you'd want to create 2D process groups
        
        # Determine grid dimensions - try to make it as square as possible
        grid_size = int(math.sqrt(world_size))
        while world_size % grid_size != 0:
            grid_size -= 1
        
        grid_x = grid_size
        grid_y = world_size // grid_size
        
        # Calculate my position in the grid
        grid_row = rank // grid_x
        grid_col = rank % grid_x
        
        # Shard matrix A along M dimension
        m_chunk_size = math.ceil(m / grid_y)
        start_row = grid_row * m_chunk_size
        end_row = min((grid_row + 1) * m_chunk_size, m)
        
        # Shard matrix B along N dimension
        n_chunk_size = math.ceil(n / grid_x)
        start_col = grid_col * n_chunk_size
        end_col = min((grid_col + 1) * n_chunk_size, n)
        
        # Skip if this rank has no work
        if start_row >= m or start_col >= n:
            if out is not None:
                dist.barrier()
                return out
            else:
                result = torch.empty((0, 0), dtype=a.dtype, device=a.device)
                dist.barrier()
                return result
        
        # Extract my shards
        a_shard = a[start_row:end_row]
        b_shard = b[start_col:end_col]
        
        # Create output shard or use provided output
        if out is None:
            out_shard = torch.empty((end_row - start_row, end_col - start_col), 
                                  dtype=torch.float32, 
                                  device=a.device)
        else:
            out_shard = out[start_row:end_row, start_col:end_col]
        
        # Perform GEMM on my shard
        result_shard = gemm_fn(a_shard, b_shard, out=out_shard, alpha=alpha, beta=beta)
        
        # Synchronize when required
        if sync:
            dist.barrier()
        
        return result_shard
    
    else:
        raise ValueError(f"Unsupported sharding strategy: {strategy}")


def distributed_gemm_fp16_fp32_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    strategy: ShardingStrategy = ShardingStrategy.ROW_PARALLEL,
    sync: bool = True
) -> torch.Tensor:
    """
    Distributed GEMM operation with FP16 inputs and FP32 accumulation.
    
    Args:
        a: First input tensor in FP16 format
        b: Second input tensor in FP16 format
        out: Output tensor in FP32 format, will be created if not provided
        alpha: Scalar multiplier for the product of input tensors
        beta: Scalar multiplier for the output
        strategy: Sharding strategy to use
        sync: Whether to synchronize at the end of the operation
    
    Returns:
        The result tensor in FP32 format
    """
    return distribute_gemm(
        a=a, 
        b=b, 
        gemm_fn=gemm_fp16_fp32_nt,
        out=out, 
        alpha=alpha, 
        beta=beta, 
        strategy=strategy,
        sync=sync
    )


def distributed_gemm_bf16_fp32_nt(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    strategy: ShardingStrategy = ShardingStrategy.ROW_PARALLEL,
    sync: bool = True
) -> torch.Tensor:
    """
    Distributed GEMM operation with BF16 inputs and FP32 accumulation.
    
    Args:
        a: First input tensor in BF16 format
        b: Second input tensor in BF16 format
        out: Output tensor in FP32 format, will be created if not provided
        alpha: Scalar multiplier for the product of input tensors
        beta: Scalar multiplier for the output
        strategy: Sharding strategy to use
        sync: Whether to synchronize at the end of the operation
    
    Returns:
        The result tensor in FP32 format
    """
    return distribute_gemm(
        a=a, 
        b=b, 
        gemm_fn=gemm_bf16_fp32_nt,
        out=out, 
        alpha=alpha, 
        beta=beta, 
        strategy=strategy,
        sync=sync
    )


def benchmark_distributed_gemm(
    m: int, 
    n: int, 
    k: int, 
    dtype: torch.dtype = torch.float16,
    num_warmups: int = 5, 
    num_runs: int = 10,
    strategy: ShardingStrategy = ShardingStrategy.ROW_PARALLEL
) -> float:
    """
    Benchmark the distributed GEMM operation.
    
    Args:
        m, n, k: Matrix dimensions
        dtype: Data type for the input tensors
        num_warmups: Number of warmup iterations
        num_runs: Number of benchmark iterations
        strategy: Sharding strategy to use
    
    Returns:
        Average execution time in milliseconds
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # Create test matrices on the appropriate device
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    # Create full matrices on rank 0 and distribute
    if rank == 0:
        a_full = torch.randn((m, k), dtype=dtype, device=device)
        b_full = torch.randn((n, k), dtype=dtype, device=device)
    else:
        a_full = None
        b_full = None
    
    # Broadcast tensors to all ranks (in a real implementation, you'd use more efficient sharding)
    if world_size > 1:
        if a_full is not None:
            a_full = a_full.contiguous()
            b_full = b_full.contiguous()
        
        # Allocate tensors on other ranks
        if rank != 0:
            a_full = torch.empty((m, k), dtype=dtype, device=device)
            b_full = torch.empty((n, k), dtype=dtype, device=device)
        
        # Broadcast from rank 0 to all other ranks
        dist.broadcast(a_full, 0)
        dist.broadcast(b_full, 0)
    
    # Select the appropriate GEMM function
    if dtype == torch.float16:
        distributed_gemm_fn = distributed_gemm_fp16_fp32_nt
    elif dtype == torch.bfloat16:
        distributed_gemm_fn = distributed_gemm_bf16_fp32_nt
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # Run the benchmark
    def run_distributed_gemm():
        distributed_gemm_fn(a_full, b_full, strategy=strategy, sync=True)
    
    # Only measure on rank 0
    if rank == 0:
        return bench(run_distributed_gemm, num_warmups=num_warmups, num_tests=num_runs)
    else:
        # Run but don't measure on other ranks
        for _ in range(num_warmups):
            run_distributed_gemm()
        for _ in range(num_runs):
            run_distributed_gemm()
        return 0.0 