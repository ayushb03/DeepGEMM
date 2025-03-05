"""
Distributed GEMM operations for DeepGEMM.

This module provides functionality for distributed matrix multiplication
across multiple GPUs, including both model and data parallelism.

The module is structured as follows:
1. Core implementations for distributed GEMM operations.
2. Communication primitives for efficient tensor distribution.
3. High-level API for production-ready use.

For production use, it is recommended to use the DistributedGEMM class
and related utilities provided by the public API.
"""

# Core implementation
from .multi_gpu import (
    distribute_gemm,
    distributed_gemm_fp16_fp32_nt,
    distributed_gemm_bf16_fp32_nt,
    benchmark_distributed_gemm,
    ShardingStrategy,
)

# Communication primitives
from .communication import (
    all_gather_matrix,
    reduce_scatter_matrix,
    broadcast_matrix,
    scatter_matrix,
    all_reduce_matrix,
    get_process_group,
    initialize_process_group,
    cleanup_process_group,
)

# Public API for production use
from .api import (
    # Configuration
    DistributedConfig,
    
    # Main class
    DistributedGEMM,
    
    # Helper functions
    initialize_distributed_environment,
    cleanup_distributed_environment,
    get_local_rank,
    get_world_size,
    get_rank,
    
    # GEMM operations
    gemm_fp16_fp32_nt,
    gemm_bf16_fp32_nt,
    
    # Communication operations
    all_gather,
    reduce_scatter,
    broadcast,
    scatter,
    all_reduce,
)

# Configuration utilities
from .config_loader import (
    load_config_from_file,
    create_distributed_config,
    init_from_config,
    configure_logging,
)

# Set default logging configuration
import logging
logging.getLogger("deep_gemm.distributed").setLevel(logging.INFO) 