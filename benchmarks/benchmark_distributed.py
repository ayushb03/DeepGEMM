"""
Benchmarks for distributed multi-GPU GEMM operations.
"""

import argparse
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

from deep_gemm.distributed import (
    ShardingStrategy,
    distributed_gemm_fp16_fp32_nt,
    distributed_gemm_bf16_fp32_nt,
    initialize_process_group,
    get_rank,
    get_world_size,
    benchmark_distributed_gemm
)


def setup_distributed(rank, world_size, port):
    """
    Initialize the distributed environment.
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        port: Port for distributed communication
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Initialize process group
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, 
                           world_size=world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        
    print(f"Rank {rank}/{world_size} initialized")


def cleanup():
    """Clean up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_benchmarks_worker(
    rank: int,
    world_size: int,
    port: int,
    matrix_sizes: List[Tuple[int, int, int]],
    strategies: List[ShardingStrategy],
    num_warmups: int = 5,
    num_runs: int = 10,
):
    """
    Worker function for running benchmarks on a single process.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed communication
        matrix_sizes: List of (M, N, K) matrix sizes to benchmark
        strategies: List of sharding strategies to benchmark
        num_warmups: Number of warmup iterations
        num_runs: Number of benchmark iterations
    """
    # Initialize distributed environment
    setup_distributed(rank, world_size, port)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print(f"Rank {rank}/{world_size} initialized")
        print("CUDA is not available, skipping benchmark")
        cleanup()
        return
    
    # Set device
    torch.cuda.set_device(rank)
    
    # Results storage
    fp16_results = {}
    bf16_results = {}
    
    # Only rank 0 prints progress
    if rank == 0:
        print(f"Running benchmarks on {world_size} GPUs")
        print(f"Matrix sizes: {matrix_sizes}")
        print(f"Strategies: {strategies}")
        print(f"Warmup iterations: {num_warmups}")
        print(f"Benchmark iterations: {num_runs}")
    
    # Run benchmarks for each strategy
    for strategy in strategies:
        strategy_name = str(strategy).split('.')[-1]
        
        if rank == 0:
            print(f"\nBenchmarking strategy: {strategy_name}")
        
        fp16_results[strategy_name] = []
        bf16_results[strategy_name] = []
        
        # Run benchmarks for each matrix size
        for m, n, k in matrix_sizes:
            if rank == 0:
                print(f"  Matrix size: {m}x{n}x{k}")
            
            # FP16 benchmark
            try:
                fp16_time = benchmark_distributed_gemm(
                    m, n, k, 
                    strategy=strategy,
                    dtype=torch.float16,
                    num_warmups=num_warmups,
                    num_runs=num_runs
                )
                
                if rank == 0:
                    print(f"    FP16: {fp16_time:.2f} ms")
                    fp16_results[strategy_name].append((m, n, k, fp16_time))
            except Exception as e:
                if rank == 0:
                    print(f"    FP16 benchmark failed: {e}")
            
            # BF16 benchmark
            try:
                bf16_time = benchmark_distributed_gemm(
                    m, n, k, 
                    strategy=strategy,
                    dtype=torch.bfloat16,
                    num_warmups=num_warmups,
                    num_runs=num_runs
                )
                
                if rank == 0:
                    print(f"    BF16: {bf16_time:.2f} ms")
                    bf16_results[strategy_name].append((m, n, k, bf16_time))
            except Exception as e:
                if rank == 0:
                    print(f"    BF16 benchmark failed: {e}")
    
    # Plot results (only on rank 0)
    if rank == 0:
        plot_results(fp16_results, bf16_results, world_size, matrix_sizes, strategies)
    
    # Clean up
    cleanup()


def plot_results(
    fp16_results: Dict[str, List[Tuple[int, int, int, float]]],
    bf16_results: Dict[str, List[Tuple[int, int, int, float]]],
    world_size: int,
    matrix_sizes: List[Tuple[int, int, int]],
    strategies: List[ShardingStrategy]
):
    """
    Plot benchmark results.
    
    Args:
        fp16_results: FP16 benchmark results by strategy
        bf16_results: BF16 benchmark results by strategy
        world_size: Number of GPUs used
        matrix_sizes: List of matrix dimensions benchmarked
        strategies: List of sharding strategies benchmarked
    """
    # Create figure for plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Create labels for x-axis
    x_labels = [f"{m}x{n}x{k}" for m, n, k in matrix_sizes]
    x = np.arange(len(x_labels))
    width = 0.8 / len(strategies)  # Width of bars
    
    # Plot FP16 results
    for i, strategy in enumerate(strategies):
        strategy_name = strategy.value
        times = [result[3] for result in fp16_results[strategy_name]]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax1.bar(x + offset, times, width, label=f"{strategy_name}")
    
    ax1.set_title(f"FP16 GEMM Performance ({world_size} GPUs)")
    ax1.set_xlabel("Matrix Dimensions (MxNxK)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot BF16 results
    for i, strategy in enumerate(strategies):
        strategy_name = strategy.value
        times = [result[3] for result in bf16_results[strategy_name]]
        offset = (i - len(strategies) / 2 + 0.5) * width
        ax2.bar(x + offset, times, width, label=f"{strategy_name}")
    
    ax2.set_title(f"BF16 GEMM Performance ({world_size} GPUs)")
    ax2.set_xlabel("Matrix Dimensions (MxNxK)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"distributed_gemm_benchmark_{world_size}_gpus.png")
    plt.close()

    print(f"Benchmark results saved to distributed_gemm_benchmark_{world_size}_gpus.png")


def run_benchmarks(
    world_size: int,
    port: int = 12355,
    matrix_sizes: List[Tuple[int, int, int]] = None,
    strategies: List[str] = None,
    num_warmups: int = 5,
    num_runs: int = 10,
):
    """
    Run benchmarks for distributed GEMM operations.
    
    Args:
        world_size: Number of processes to use
        port: Port for distributed communication
        matrix_sizes: List of (M, N, K) matrix sizes to benchmark
        strategies: List of sharding strategies to benchmark
        num_warmups: Number of warmup iterations
        num_runs: Number of benchmark iterations
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Distributed benchmarks require CUDA.")
        print("Skipping distributed benchmarks.")
        return
        
    print(f"Starting benchmark with {world_size} processes")
    
    # Default matrix sizes if not provided
    if matrix_sizes is None:
        matrix_sizes = [
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            (8192, 8192, 8192)
        ]
    
    # Default strategies if not provided
    if strategies is None:
        strategies = [
            ShardingStrategy.ROW_PARALLEL,
            ShardingStrategy.COLUMN_PARALLEL
        ]
        
        # Only add FULLY_SHARDED if world_size is a perfect square
        if (int(world_size ** 0.5) ** 2) == world_size and world_size > 1:
            strategies.append(ShardingStrategy.FULLY_SHARDED)
    
    # Convert string strategies to enum
    strategy_map = {
        "row": ShardingStrategy.ROW_PARALLEL,
        "column": ShardingStrategy.COLUMN_PARALLEL,
        "fully_sharded": ShardingStrategy.FULLY_SHARDED
    }
    
    strategies_enum = []
    for strategy in strategies:
        if isinstance(strategy, str):
            if strategy.lower() in strategy_map:
                strategies_enum.append(strategy_map[strategy.lower()])
            else:
                print(f"Warning: Unknown strategy '{strategy}', skipping")
        else:
            strategies_enum.append(strategy)
    
    # Start processes
    mp.spawn(
        run_benchmarks_worker,
        args=(world_size, port, matrix_sizes, strategies_enum, num_warmups, num_runs),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark distributed GEMM operations")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs to use")
    parser.add_argument("--port", type=int, default=12355, help="Port for distributed communication")
    parser.add_argument("--warmups", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark iterations")
    parser.add_argument("--strategies", type=str, nargs="+", choices=["row", "column", "fully_sharded"],
                        default=["row", "column", "fully_sharded"], help="Sharding strategies to benchmark")
    
    args = parser.parse_args()
    
    # Run the benchmarks
    run_benchmarks(
        world_size=args.world_size,
        port=args.port,
        strategies=args.strategies,
        num_warmups=args.warmups,
        num_runs=args.runs
    ) 