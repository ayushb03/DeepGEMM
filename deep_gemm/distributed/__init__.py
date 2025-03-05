"""
Distributed GEMM operations for DeepGEMM.

This module provides functionality for distributed matrix multiplication
across multiple GPUs, including model parallelism and data parallelism strategies.
"""

from .multi_gpu import (
    distribute_gemm,
    distributed_gemm_fp16_fp32_nt,
    distributed_gemm_bf16_fp32_nt,
    ShardingStrategy,
    initialize_process_group,
    get_rank,
    get_world_size,
    benchmark_distributed_gemm
)

from .communication import (
    all_gather_matrix,
    reduce_scatter_matrix,
    broadcast_matrix,
    scatter_matrix,
    allreduce_matrix
) 