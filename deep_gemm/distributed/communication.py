"""
Optimized communication primitives for distributed GEMM operations.

This module provides efficient communication operations for transferring
matrix data between multiple GPUs during distributed GEMM computation.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .multi_gpu import get_rank, get_world_size


def all_gather_matrix(
    local_matrix: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """
    Gather matrices from all processes and concatenate them along the specified dimension.
    
    Args:
        local_matrix: Local matrix tensor on the current process
        dim: Dimension along which to concatenate tensors (0 for row-wise, 1 for column-wise)
    
    Returns:
        Concatenated tensor containing matrices from all processes
    """
    world_size = get_world_size()
    
    # If only one process, just return the input
    if world_size == 1:
        return local_matrix
    
    # Ensure the tensor is contiguous
    local_matrix = local_matrix.contiguous()
    
    # Get the list of tensors from all ranks
    gather_list = [torch.zeros_like(local_matrix) for _ in range(world_size)]
    
    # All-gather operation
    dist.all_gather(gather_list, local_matrix)
    
    # Concatenate the gathered tensors
    return torch.cat(gather_list, dim=dim)


def reduce_scatter_matrix(
    full_matrix: torch.Tensor,
    dim: int = 0,
    op: dist.ReduceOp = dist.ReduceOp.SUM
) -> torch.Tensor:
    """
    Perform a reduce-scatter operation on the matrix.
    
    Each process contributes a full matrix and receives a shard of the reduced result.
    
    Args:
        full_matrix: Full matrix tensor on the current process
        dim: Dimension along which to split the tensor (0 for row-wise, 1 for column-wise)
        op: Reduction operation (default: SUM)
    
    Returns:
        Local shard of the reduced matrix
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # If only one process, just return the input
    if world_size == 1:
        return full_matrix
    
    # Get the size of the full matrix
    full_size = full_matrix.size(dim)
    
    # Calculate shard size and start/end positions
    shard_size = math.ceil(full_size / world_size)
    start_idx = rank * shard_size
    end_idx = min((rank + 1) * shard_size, full_size)
    
    # Handle case where this rank's shard would be empty
    if start_idx >= full_size:
        # Return an empty tensor with proper dimensions
        shape = list(full_matrix.shape)
        shape[dim] = 0
        return torch.empty(shape, dtype=full_matrix.dtype, device=full_matrix.device)
    
    # Split the tensor into shards
    shards = list(torch.tensor_split(full_matrix, world_size, dim=dim))
    
    # Make sure each shard is the same size by padding if necessary
    for i in range(len(shards)):
        shard_shape = list(shards[i].shape)
        target_shape = list(shards[0].shape)
        if shard_shape != target_shape:
            # Create a new tensor with the target shape
            new_shard = torch.zeros(target_shape, 
                                  dtype=full_matrix.dtype, 
                                  device=full_matrix.device)
            # Copy the data
            slices = [slice(None)] * len(target_shape)
            slices[dim] = slice(0, shard_shape[dim])
            new_shard[slices] = shards[i]
            shards[i] = new_shard
    
    # Perform reduce-scatter operation
    output = torch.zeros_like(shards[0])
    dist.reduce_scatter(output, shards, op=op)
    
    # If the last rank's shard was padded, trim it to the correct size
    if rank == world_size - 1 and end_idx - start_idx < shard_size:
        slices = [slice(None)] * len(output.shape)
        slices[dim] = slice(0, end_idx - start_idx)
        output = output[slices]
    
    return output


def broadcast_matrix(
    matrix: torch.Tensor,
    src: int = 0
) -> torch.Tensor:
    """
    Broadcast a matrix from the source rank to all other ranks.
    
    Args:
        matrix: Matrix tensor to broadcast (only significant on src)
        src: Source rank for the broadcast
    
    Returns:
        Broadcasted matrix tensor on all ranks
    """
    # Ensure the tensor is contiguous
    matrix = matrix.contiguous()
    
    # Broadcast the tensor
    dist.broadcast(matrix, src=src)
    
    return matrix


def scatter_matrix(
    matrix: Optional[torch.Tensor] = None,
    dim: int = 0,
    src: int = 0
) -> torch.Tensor:
    """
    Scatter a matrix from source rank to all ranks.
    
    Args:
        matrix: Full matrix on source rank, None on other ranks
        dim: Dimension along which to split the tensor
        src: Source rank for the scatter
    
    Returns:
        Local shard of the matrix
    """
    world_size = get_world_size()
    rank = get_rank()
    
    # If only one process, just return the input
    if world_size == 1:
        return matrix
    
    if rank == src:
        assert matrix is not None, "Source rank must provide a matrix"
        
        # Get the size of the full matrix
        full_size = matrix.size(dim)
        
        # Calculate shard size for each rank
        shard_sizes = []
        for r in range(world_size):
            r_start = r * math.ceil(full_size / world_size)
            r_end = min((r + 1) * math.ceil(full_size / world_size), full_size)
            shard_sizes.append(r_end - r_start)
        
        # Split the tensor into chunks
        chunks = []
        start_idx = 0
        for size in shard_sizes:
            if size > 0:
                slices = [slice(None)] * matrix.dim()
                slices[dim] = slice(start_idx, start_idx + size)
                chunks.append(matrix[slices].contiguous())
                start_idx += size
            else:
                # Create an empty shard with the right shape
                shape = list(matrix.shape)
                shape[dim] = 0
                chunks.append(torch.empty(shape, dtype=matrix.dtype, device=matrix.device))
        
        # Calculate the output shape for this rank
        my_start = rank * math.ceil(full_size / world_size)
        my_end = min((rank + 1) * math.ceil(full_size / world_size), full_size)
        
        # If this rank would get an empty tensor, create an empty one with the right shape
        if my_start >= full_size:
            shape = list(matrix.shape)
            shape[dim] = 0
            output = torch.empty(shape, dtype=matrix.dtype, device=matrix.device)
        else:
            # Otherwise, initialize the output with the right shape
            shape = list(matrix.shape)
            shape[dim] = my_end - my_start
            output = chunks[rank]
        
        # Scatter the chunks to all ranks
        dist.scatter(output, chunks if rank == src else None, src=src)
        
        return output
    else:
        # Non-source ranks need to figure out what shape they'll receive
        # This requires communication since they don't have the original matrix
        
        # First, broadcast the matrix shape from source
        if rank == src:
            shape_tensor = torch.tensor(matrix.shape, dtype=torch.long, device="cpu")
        else:
            shape_tensor = torch.zeros(8, dtype=torch.long, device="cpu")  # Assume max 8D tensor
        
        dist.broadcast(shape_tensor, src=src)
        
        # Get the actual shape
        shape = shape_tensor.tolist()
        while len(shape) > 0 and shape[-1] == 0:
            shape.pop()
        
        # Also broadcast the full size of the dimension we're scattering on
        if rank == src:
            full_size_tensor = torch.tensor([matrix.size(dim)], dtype=torch.long, device="cpu")
        else:
            full_size_tensor = torch.zeros(1, dtype=torch.long, device="cpu")
        
        dist.broadcast(full_size_tensor, src=src)
        full_size = full_size_tensor.item()
        
        # Calculate this rank's shard size
        my_start = rank * math.ceil(full_size / world_size)
        my_end = min((rank + 1) * math.ceil(full_size / world_size), full_size)
        
        # If this rank would get an empty tensor, create an empty one with the right shape
        if my_start >= full_size:
            shape = list(shape)
            shape[dim] = 0
            output = torch.empty(shape, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        else:
            # Otherwise, initialize the output with the right shape
            shape = list(shape)
            shape[dim] = my_end - my_start
            output = torch.empty(shape, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Receive the data from source rank
        dist.scatter(output, None, src=src)
        
        return output


def allreduce_matrix(
    matrix: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM
) -> torch.Tensor:
    """
    Perform an all-reduce operation on a matrix.
    
    Args:
        matrix: Local matrix tensor on the current process
        op: Reduction operation (default: SUM)
    
    Returns:
        Reduced matrix tensor on all processes
    """
    # Ensure the tensor is contiguous
    matrix = matrix.contiguous()
    
    # Perform all-reduce operation
    dist.all_reduce(matrix, op=op)
    
    return matrix 