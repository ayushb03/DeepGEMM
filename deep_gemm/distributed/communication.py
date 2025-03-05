"""
Distributed communication primitives for DeepGEMM.

This module provides efficient communication operations for distributed GEMM,
with robust error handling and validation checks.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

# Configure logging
logger = logging.getLogger("deep_gemm.distributed.comm")


class DistributedError(Exception):
    """Base exception for distributed communication errors."""
    pass


class ProcessGroupError(DistributedError):
    """Exception raised for process group errors."""
    pass


class CommunicationError(DistributedError):
    """Exception raised for communication operation errors."""
    pass


class DeviceMismatchError(DistributedError):
    """Exception raised when tensors are not on the expected device."""
    pass


class ShapeMismatchError(DistributedError):
    """Exception raised when tensor shapes don't match expected dimensions."""
    pass


def initialize_process_group(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    timeout: float = 1800.0,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> None:
    """
    Initialize the process group for distributed operations with robust error handling.
    
    Args:
        backend: The backend to use (nccl, gloo, etc.)
        init_method: The URL to use for initialization
        timeout: Timeout in seconds for operations
        world_size: Number of processes participating
        rank: Rank of this process
        
    Raises:
        ProcessGroupError: If initialization fails
    """
    # Use environment variables if parameters are not provided
    if world_size is None:
        world_size_env = os.environ.get("WORLD_SIZE")
        if world_size_env is None:
            raise ProcessGroupError(
                "world_size not provided and WORLD_SIZE environment variable not set"
            )
        world_size = int(world_size_env)
    
    if rank is None:
        rank_env = os.environ.get("RANK")
        if rank_env is None:
            raise ProcessGroupError(
                "rank not provided and RANK environment variable not set"
            )
        rank = int(rank_env)
    
    # Default initialization method
    if init_method is None:
        # Use environment variables if available
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", "12355")
        init_method = f"tcp://{master_addr}:{master_port}"
    
    try:
        # Check if already initialized
        if dist.is_initialized():
            logger.warning("Process group already initialized, skipping initialization")
            return
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(seconds=timeout)
        )
        
        logger.info(
            f"Initialized process group: backend={backend}, "
            f"world_size={world_size}, rank={rank}"
        )
    except (RuntimeError, ValueError) as e:
        raise ProcessGroupError(f"Failed to initialize process group: {e}") from e


def cleanup_process_group() -> None:
    """
    Clean up the process group for distributed operations.
    
    Raises:
        ProcessGroupError: If cleanup fails
    """
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Process group cleaned up")
        else:
            logger.warning("Process group not initialized, skipping cleanup")
    except RuntimeError as e:
        raise ProcessGroupError(f"Failed to clean up process group: {e}") from e


def get_process_group(
    ranks: Optional[List[int]] = None,
    timeout: float = 1800.0
) -> Any:
    """
    Get a process group for collective operations.
    
    Args:
        ranks: List of ranks to include in the process group (None for all)
        timeout: Timeout in seconds for operations
        
    Returns:
        Process group for collective operations
        
    Raises:
        ProcessGroupError: If getting process group fails
    """
    try:
        if not dist.is_initialized():
            raise ProcessGroupError("Process group not initialized")
        
        if ranks is None:
            return dist.group.WORLD
        
        return dist.new_group(
            ranks=ranks,
            timeout=datetime.timedelta(seconds=timeout)
        )
    except RuntimeError as e:
        raise ProcessGroupError(f"Failed to get process group: {e}") from e


def validate_tensor(
    tensor: torch.Tensor,
    expected_ndim: Optional[int] = None,
    expected_dtype: Optional[torch.dtype] = None,
    expected_device_type: Optional[str] = None,
    name: str = "tensor"
) -> None:
    """
    Validate tensor properties.
    
    Args:
        tensor: Tensor to validate
        expected_ndim: Expected number of dimensions
        expected_dtype: Expected data type
        expected_device_type: Expected device type (cuda, cpu)
        name: Name of the tensor for error messages
        
    Raises:
        ValueError: If tensor is invalid
        ShapeMismatchError: If tensor doesn't have expected dimensions
        DeviceMismatchError: If tensor is not on expected device
    """
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    
    if expected_ndim is not None and tensor.ndim != expected_ndim:
        raise ShapeMismatchError(
            f"{name} must have {expected_ndim} dimensions, got {tensor.ndim}"
        )
    
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        logger.warning(
            f"{name} expected dtype {expected_dtype}, got {tensor.dtype}. "
            "This may affect performance or accuracy."
        )
    
    if expected_device_type is not None:
        device_type = tensor.device.type
        if device_type != expected_device_type:
            raise DeviceMismatchError(
                f"{name} must be on {expected_device_type} device, got {device_type}"
            )


def all_gather_matrix(
    local_matrix: torch.Tensor,
    dim: int = 0,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    Gather matrices from all processes and concatenate them along the specified dimension.
    
    Args:
        local_matrix: Local matrix from this process
        dim: Dimension to concatenate along
        group: Process group
        
    Returns:
        Gathered matrix
        
    Raises:
        CommunicationError: If all_gather operation fails
        ShapeMismatchError: If tensor doesn't have expected dimensions
        DeviceMismatchError: If tensor is not on expected device
    """
    validate_tensor(
        local_matrix,
        expected_ndim=2,
        expected_device_type="cuda",
        name="local_matrix"
    )
    
    if not dist.is_initialized():
        logger.warning("Process group not initialized, returning local matrix")
        return local_matrix
    
    world_size = dist.get_world_size(group)
    
    # Skip if only one process
    if world_size == 1:
        return local_matrix
    
    try:
        # Get tensor shape and device
        local_shape = local_matrix.shape
        device = local_matrix.device
        
        # Create list to store output tensors
        output_tensors = [torch.zeros_like(local_matrix) for _ in range(world_size)]
        
        # Perform all_gather
        dist.all_gather(output_tensors, local_matrix, group=group)
        
        # Concatenate tensors
        gathered_matrix = torch.cat(output_tensors, dim=dim)
        
        return gathered_matrix
    except (RuntimeError, ValueError) as e:
        raise CommunicationError(f"Failed to all_gather matrix: {e}") from e


def reduce_scatter_matrix(
    input_matrix: torch.Tensor,
    dim: int = 0,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    Reduce matrices from all processes and scatter chunks to each process.
    
    Args:
        input_matrix: Input matrix to reduce and scatter
        dim: Dimension to scatter along
        op: Reduction operation
        group: Process group
        
    Returns:
        Reduced and scattered matrix
        
    Raises:
        CommunicationError: If reduce_scatter operation fails
        ShapeMismatchError: If tensor doesn't have expected dimensions
        DeviceMismatchError: If tensor is not on expected device
    """
    validate_tensor(
        input_matrix,
        expected_ndim=2,
        expected_device_type="cuda",
        name="input_matrix"
    )
    
    if not dist.is_initialized():
        logger.warning("Process group not initialized, returning input matrix")
        return input_matrix
    
    world_size = dist.get_world_size(group)
    
    # Skip if only one process
    if world_size == 1:
        return input_matrix
    
    try:
        # Get tensor shape
        input_shape = input_matrix.shape
        
        # Verify tensor can be evenly divided
        scatter_dim_size = input_shape[dim]
        if scatter_dim_size % world_size != 0:
            raise ShapeMismatchError(
                f"Cannot evenly scatter dimension {dim} of size {scatter_dim_size} "
                f"across {world_size} processes"
            )
        
        # Calculate output size
        output_shape = list(input_shape)
        output_shape[dim] = scatter_dim_size // world_size
        
        # Create output tensor
        output_tensor = torch.empty(output_shape, dtype=input_matrix.dtype, device=input_matrix.device)
        
        # Perform reduce-scatter (implemented using scatter and all-reduce since PyTorch doesn't have direct reduce_scatter)
        input_chunks = torch.chunk(input_matrix, world_size, dim=dim)
        chunk_size = input_chunks[0].size(dim)
        
        # Scatter
        rank = dist.get_rank(group)
        local_chunk = input_chunks[rank]
        
        # All-reduce the local chunk
        dist.all_reduce(local_chunk, op=op, group=group)
        
        return local_chunk
    except (RuntimeError, ValueError) as e:
        raise CommunicationError(f"Failed to reduce_scatter matrix: {e}") from e


def broadcast_matrix(
    matrix: torch.Tensor,
    src: int = 0,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    Broadcast matrix from specified source to all processes.
    
    Args:
        matrix: Matrix to broadcast or receive
        src: Source rank
        group: Process group
        
    Returns:
        Broadcasted matrix
        
    Raises:
        CommunicationError: If broadcast operation fails
        DeviceMismatchError: If tensor is not on expected device
    """
    validate_tensor(
        matrix,
        expected_device_type="cuda",
        name="matrix"
    )
    
    if not dist.is_initialized():
        logger.warning("Process group not initialized, returning input matrix")
        return matrix
    
    world_size = dist.get_world_size(group)
    
    # Skip if only one process
    if world_size == 1:
        return matrix
    
    try:
        # Perform broadcast
        dist.broadcast(matrix, src=src, group=group)
        return matrix
    except (RuntimeError, ValueError) as e:
        raise CommunicationError(f"Failed to broadcast matrix: {e}") from e


def scatter_matrix(
    scatter_list: Optional[List[torch.Tensor]] = None,
    src: int = 0,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    Scatter matrices from source to all processes.
    
    Args:
        scatter_list: List of tensors to scatter (only needed on source process)
        src: Source rank
        group: Process group
        
    Returns:
        Scattered matrix for this process
        
    Raises:
        CommunicationError: If scatter operation fails
        DeviceMismatchError: If tensor is not on expected device
    """
    if not dist.is_initialized():
        logger.warning("Process group not initialized, returning first matrix from list")
        if scatter_list is not None and len(scatter_list) > 0:
            return scatter_list[0]
        raise ValueError("scatter_list is empty or None")
    
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    
    # Skip if only one process
    if world_size == 1:
        if scatter_list is not None and len(scatter_list) > 0:
            return scatter_list[0]
        raise ValueError("scatter_list is empty or None")
    
    try:
        # If not source, we only need an output tensor
        if rank != src:
            if scatter_list is not None:
                # Use the expected shape if provided
                output_tensor = torch.zeros_like(scatter_list[0])
            else:
                # Wait to receive the tensor shape
                raise ValueError("Non-source process needs scatter_list with correct shape")
        else:
            # Validate input
            if scatter_list is None or len(scatter_list) != world_size:
                raise ValueError(f"scatter_list must contain {world_size} tensors")
            
            # Validate each tensor
            for i, tensor in enumerate(scatter_list):
                validate_tensor(
                    tensor,
                    expected_device_type="cuda",
                    name=f"scatter_list[{i}]"
                )
            
            # Create output tensor
            output_tensor = torch.zeros_like(scatter_list[rank])
            
        # Perform scatter
        dist.scatter(output_tensor, scatter_list if rank == src else None, src=src, group=group)
        return output_tensor
    except (RuntimeError, ValueError) as e:
        raise CommunicationError(f"Failed to scatter matrix: {e}") from e


def all_reduce_matrix(
    matrix: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[Any] = None
) -> torch.Tensor:
    """
    Reduce matrix from all processes and distribute result back to all processes.
    
    Args:
        matrix: Matrix to reduce
        op: Reduction operation
        group: Process group
        
    Returns:
        Reduced matrix
        
    Raises:
        CommunicationError: If all_reduce operation fails
        DeviceMismatchError: If tensor is not on expected device
    """
    validate_tensor(
        matrix,
        expected_device_type="cuda",
        name="matrix"
    )
    
    if not dist.is_initialized():
        logger.warning("Process group not initialized, returning input matrix")
        return matrix
    
    world_size = dist.get_world_size(group)
    
    # Skip if only one process
    if world_size == 1:
        return matrix
    
    try:
        # Perform all_reduce
        dist.all_reduce(matrix, op=op, group=group)
        return matrix
    except (RuntimeError, ValueError) as e:
        raise CommunicationError(f"Failed to all_reduce matrix: {e}") from e
        
        
# Additional imports for datetime
import datetime 