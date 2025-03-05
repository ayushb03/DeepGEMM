"""
Production-ready API for DeepGEMM distributed operations.

This module provides simplified, robust interfaces for using distributed GEMM
operations in production environments.
"""

import contextlib
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from .multi_gpu import (
    ShardingStrategy,
    distributed_gemm_fp16_fp32_nt,
    distributed_gemm_bf16_fp32_nt,
    initialize_process_group,
    get_rank,
    get_world_size
)
from .communication import (
    all_gather_matrix,
    reduce_scatter_matrix,
    broadcast_matrix,
    scatter_matrix,
    all_reduce_matrix
)


# Configure logging
logger = logging.getLogger("deep_gemm.distributed")


class DistributedConfig:
    """Configuration for distributed operations."""
    
    def __init__(
        self,
        strategy: ShardingStrategy = ShardingStrategy.ROW_PARALLEL,
        backend: str = "nccl", 
        master_addr: str = "localhost",
        master_port: int = 12355,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        local_rank: Optional[int] = None,
        timeout: float = 1800.0,
        device_type: str = "cuda"
    ):
        """
        Initialize distributed configuration.
        
        Args:
            strategy: Sharding strategy for distributed GEMM operations
            backend: PyTorch distributed backend (nccl, gloo, etc.)
            master_addr: Address of the master node
            master_port: Port for distributed communication
            world_size: Total number of processes (if None, read from environment)
            rank: Global rank of this process (if None, read from environment)
            local_rank: Local rank on this node (if None, read from environment)
            timeout: Timeout in seconds for operations
            device_type: Device type to use (cuda, cpu)
        """
        self.strategy = strategy
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.timeout = timeout
        self.device_type = device_type
        
        # Resolve environment variables if values not provided
        if self.world_size is None:
            self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.rank is None:
            self.rank = int(os.environ.get("RANK", 0))
            
        if self.local_rank is None:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
        self._initialized = False
        
    def __repr__(self) -> str:
        """Return string representation of config."""
        return (
            f"DistributedConfig(strategy={self.strategy}, "
            f"backend={self.backend}, "
            f"rank={self.rank}/{self.world_size})"
        )


class DistributedGEMM:
    """
    Production-ready interface for distributed GEMM operations.
    
    This class provides a high-level interface for distributed GEMM operations
    with proper error handling and logging.
    
    Example:
        ```python
        config = DistributedConfig(strategy=ShardingStrategy.ROW_PARALLEL)
        with DistributedGEMM(config) as dgemm:
            result = dgemm.gemm_fp16_fp32_nt(a, b)
        ```
    """
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        """
        Initialize the distributed GEMM interface.
        
        Args:
            config: Distributed configuration
        """
        self.config = config or DistributedConfig()
        self._initialized = False
        self.logger = logging.getLogger("deep_gemm.distributed.api")
    
    def __enter__(self):
        """Enter context manager."""
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def initialize(self) -> bool:
        """
        Initialize the distributed environment.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        if self._initialized:
            logger.info("Distributed environment already initialized")
            return True
            
        if self.config._initialized:
            self._initialized = True
            logger.info("Using previously initialized distributed environment")
            return True
            
        try:
            # Set environment variables for PyTorch distributed
            os.environ["MASTER_ADDR"] = self.config.master_addr
            os.environ["MASTER_PORT"] = str(self.config.master_port)
            
            # Check if we're already initialized (e.g., by torch.distributed.launch)
            if not dist.is_initialized():
                logger.info(
                    f"Initializing distributed environment: "
                    f"rank {self.config.rank}/{self.config.world_size}, "
                    f"backend={self.config.backend}"
                )
                
                # Initialize process group
                initialize_process_group(backend=self.config.backend)
                
                # Set device for this process if using CUDA
                if self.config.device_type == "cuda" and torch.cuda.is_available():
                    torch.cuda.set_device(self.config.local_rank)
            else:
                logger.info("Using existing distributed environment")
            
            # Verify that initialization was successful
            if dist.is_initialized():
                actual_rank = get_rank()
                actual_world_size = get_world_size()
                logger.info(f"Distributed environment initialized: rank {actual_rank}/{actual_world_size}")
                self._initialized = True
                self.config._initialized = True
                return True
            else:
                logger.error("Failed to initialize distributed environment")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing distributed environment: {str(e)}")
            return False
    
    def is_distributed(self) -> bool:
        """
        Check if we're running in a distributed environment.
        
        Returns:
            True if distributed (world_size > 1), False otherwise
        """
        return get_world_size() > 1
    
    def get_rank(self) -> int:
        """Get the rank of this process."""
        return get_rank()
    
    def get_world_size(self) -> int:
        """Get the total number of processes."""
        return get_world_size()
    
    def cleanup(self) -> None:
        """Clean up the distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
            self.config._initialized = False
            logger.info("Distributed environment cleaned up")
    
    @contextlib.contextmanager
    def distributed_context(self):
        """
        Context manager for distributed operations.
        
        Usage:
            with dist_gemm.distributed_context():
                # Do distributed operations here
        """
        success = False
        try:
            success = self.initialize()
            if not success:
                raise RuntimeError("Failed to initialize distributed environment")
            yield
        finally:
            if success:
                self.cleanup()
    
    def gemm_fp16(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        strategy: Optional[ShardingStrategy] = None,
        sync: bool = True
    ) -> torch.Tensor:
        """
        Perform distributed FP16 to FP32 GEMM operation.
        
        Args:
            a: First input tensor (FP16)
            b: Second input tensor (FP16)
            out: Output tensor (FP32), will be created if not provided
            alpha: Scalar multiplier for the product of input tensors
            beta: Scalar multiplier for the output
            strategy: Sharding strategy to use (defaults to config strategy)
            sync: Whether to synchronize at the end of the operation
            
        Returns:
            Result tensor in FP32 format
        """
        if not self._initialized and not self.initialize():
            raise RuntimeError("Distributed environment not initialized")
            
        # Ensure inputs are in FP16 format
        if a.dtype != torch.float16:
            logger.warning(f"Input tensor 'a' has dtype {a.dtype}, converting to float16")
            a = a.to(torch.float16)
            
        if b.dtype != torch.float16:
            logger.warning(f"Input tensor 'b' has dtype {b.dtype}, converting to float16")
            b = b.to(torch.float16)
        
        # Use strategy from config if not provided
        if strategy is None:
            strategy = self.config.strategy
            
        try:
            return distributed_gemm_fp16_fp32_nt(
                a=a,
                b=b,
                out=out,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
                sync=sync
            )
        except Exception as e:
            logger.error(f"Error in distributed FP16 GEMM: {str(e)}")
            raise
    
    def gemm_bf16(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        strategy: Optional[ShardingStrategy] = None,
        sync: bool = True
    ) -> torch.Tensor:
        """
        Perform distributed BF16 to FP32 GEMM operation.
        
        Args:
            a: First input tensor (BF16)
            b: Second input tensor (BF16)
            out: Output tensor (FP32), will be created if not provided
            alpha: Scalar multiplier for the product of input tensors
            beta: Scalar multiplier for the output
            strategy: Sharding strategy to use (defaults to config strategy)
            sync: Whether to synchronize at the end of the operation
            
        Returns:
            Result tensor in FP32 format
        """
        if not self._initialized and not self.initialize():
            raise RuntimeError("Distributed environment not initialized")
            
        # Check if BF16 is supported
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is not supported on this device")
            
        # Ensure inputs are in BF16 format
        if a.dtype != torch.bfloat16:
            logger.warning(f"Input tensor 'a' has dtype {a.dtype}, converting to bfloat16")
            a = a.to(torch.bfloat16)
            
        if b.dtype != torch.bfloat16:
            logger.warning(f"Input tensor 'b' has dtype {b.dtype}, converting to bfloat16")
            b = b.to(torch.bfloat16)
        
        # Use strategy from config if not provided
        if strategy is None:
            strategy = self.config.strategy
            
        try:
            return distributed_gemm_bf16_fp32_nt(
                a=a,
                b=b,
                out=out,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
                sync=sync
            )
        except Exception as e:
            logger.error(f"Error in distributed BF16 GEMM: {str(e)}")
            raise
    
    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Gather tensor from all processes along specified dimension.
        
        Args:
            tensor: Local tensor on current process
            dim: Dimension along which to concatenate (0 for rows, 1 for columns)
            
        Returns:
            Concatenated tensor from all processes
        """
        if not self._initialized and not self.initialize():
            raise RuntimeError("Distributed environment not initialized")
            
        try:
            return all_gather_matrix(tensor, dim=dim)
        except Exception as e:
            logger.error(f"Error in all_gather: {str(e)}")
            raise
    
    def reduce_scatter(
        self, 
        tensor: torch.Tensor, 
        dim: int = 0,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Reduce-scatter tensor along specified dimension.
        
        Args:
            tensor: Full tensor on current process
            dim: Dimension along which to split (0 for rows, 1 for columns)
            op: Reduction operation (default: SUM)
            
        Returns:
            Local shard of reduced tensor
        """
        if not self._initialized and not self.initialize():
            raise RuntimeError("Distributed environment not initialized")
            
        try:
            return reduce_scatter_matrix(tensor, dim=dim, op=op)
        except Exception as e:
            logger.error(f"Error in reduce_scatter: {str(e)}")
            raise
    
    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """
        Broadcast tensor from source rank to all ranks.
        
        Args:
            tensor: Tensor to broadcast (only significant on src)
            src: Source rank for broadcast
            
        Returns:
            Broadcasted tensor on all ranks
        """
        if not self._initialized and not self.initialize():
            raise RuntimeError("Distributed environment not initialized")
            
        try:
            return broadcast_matrix(tensor, src=src)
        except Exception as e:
            logger.error(f"Error in broadcast: {str(e)}")
            raise
    
    def all_reduce(
        self, 
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        """
        Reduce tensor from all processes and distribute back to all processes.
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed():
            return tensor
            
        try:
            return all_reduce_matrix(tensor, op=op)
        except Exception as e:
            logger.error(f"Error in all_reduce: {str(e)}")
            raise

    def scatter(
        self,
        scatter_list=None,
        src: int = 0
    ) -> torch.Tensor:
        """
        Scatter tensors from source rank to all ranks.
        
        Args:
            scatter_list: List of tensors to scatter (only needed on source rank)
            src: Source rank
            
        Returns:
            Scattered tensor for this rank
        """
        if not self.is_distributed():
            if scatter_list is not None and len(scatter_list) > 0:
                return scatter_list[0]
            raise ValueError("scatter_list is empty or None")
            
        try:
            from .communication import scatter_matrix
            return scatter_matrix(scatter_list, src=src)
        except Exception as e:
            logger.error(f"Error in scatter: {str(e)}")
            raise


# Function aliases for backward compatibility
def initialize_distributed_environment(*args, **kwargs):
    """Alias for initialize_process_group for backward compatibility."""
    return initialize_process_group(*args, **kwargs)

def cleanup_distributed_environment():
    """Alias for cleanup_process_group for backward compatibility."""
    from .communication import cleanup_process_group
    return cleanup_process_group()

def get_local_rank():
    """Get local rank within the current node."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    return 0

# Create default instance
default_distributed_gemm = DistributedGEMM()

# Expose instance methods at module level
initialize = default_distributed_gemm.initialize
cleanup = default_distributed_gemm.cleanup
get_rank = default_distributed_gemm.get_rank
get_world_size = default_distributed_gemm.get_world_size
gemm_fp16_fp32_nt = default_distributed_gemm.gemm_fp16
gemm_bf16_fp32_nt = default_distributed_gemm.gemm_bf16
all_gather = default_distributed_gemm.all_gather
reduce_scatter = default_distributed_gemm.reduce_scatter
broadcast = default_distributed_gemm.broadcast
scatter = default_distributed_gemm.scatter
all_reduce = default_distributed_gemm.all_reduce 