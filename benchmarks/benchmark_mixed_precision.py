"""
Benchmarks for mixed precision GEMM operations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from deep_gemm.mixed_precision import (
    gemm_fp16_fp32_nt, benchmark_fp16_fp32,
    gemm_bf16_fp32_nt, benchmark_bf16_fp32
)

def benchmark_pytorch_fp16_fp32(m, n, k, num_warmups=5, num_runs=10):
    """Benchmark PyTorch's implementation for FP16->FP32 GEMM."""
    a = torch.randn((m, k), dtype=torch.float16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(num_warmups):
        torch.matmul(a.float(), b.float().t())
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        torch.matmul(a.float(), b.float().t())
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def benchmark_pytorch_bf16_fp32(m, n, k, num_warmups=5, num_runs=10):
    """Benchmark PyTorch's implementation for BF16->FP32 GEMM."""
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
    
    # Warmup
    for _ in range(num_warmups):
        torch.matmul(a.float(), b.float().t())
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(num_runs):
        torch.matmul(a.float(), b.float().t())
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def run_benchmarks():
    """Run benchmarks for different matrix sizes and create plots."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    # Matrix sizes to benchmark
    sizes = [(128, 128, 128), (256, 256, 256), (512, 512, 512), 
             (1024, 1024, 1024), (2048, 2048, 2048)]
    
    # Results storage
    fp16_pytorch_times = []
    fp16_custom_times = []
    bf16_pytorch_times = []
    bf16_custom_times = []
    
    print("Running benchmarks...")
    for size in sizes:
        m, n, k = size
        print(f"Benchmarking size {m}x{n}x{k}...")
        
        # FP16 benchmarks
        pytorch_time = benchmark_pytorch_fp16_fp32(m, n, k)
        custom_time = benchmark_fp16_fp32(m, n, k)
        fp16_pytorch_times.append(pytorch_time)
        fp16_custom_times.append(custom_time)
        
        # BF16 benchmarks
        pytorch_time = benchmark_pytorch_bf16_fp32(m, n, k)
        custom_time = benchmark_bf16_fp32(m, n, k)
        bf16_pytorch_times.append(pytorch_time)
        bf16_custom_times.append(custom_time)
    
    # Create plot for FP16
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    x_labels = [f"{s[0]}x{s[1]}x{s[2]}" for s in sizes]
    x = np.arange(len(x_labels))
    width = 0.35
    
    plt.bar(x - width/2, fp16_pytorch_times, width, label='PyTorch')
    plt.bar(x + width/2, fp16_custom_times, width, label='DeepGEMM')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.title('FP16->FP32 GEMM Performance')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    
    # Create plot for BF16
    plt.subplot(1, 2, 2)
    plt.bar(x - width/2, bf16_pytorch_times, width, label='PyTorch')
    plt.bar(x + width/2, bf16_custom_times, width, label='DeepGEMM')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (ms)')
    plt.title('BF16->FP32 GEMM Performance')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('mixed_precision_benchmarks.png')
    plt.close()
    
    print("Benchmarks complete. Results saved to mixed_precision_benchmarks.png")
    
    # Print speedup
    print("\nSpeedup over PyTorch:")
    print("FP16->FP32 GEMM:")
    for i, size in enumerate(sizes):
        speedup = fp16_pytorch_times[i] / fp16_custom_times[i]
        print(f"  {size[0]}x{size[1]}x{size[2]}: {speedup:.2f}x")
    
    print("\nBF16->FP32 GEMM:")
    for i, size in enumerate(sizes):
        speedup = bf16_pytorch_times[i] / bf16_custom_times[i]
        print(f"  {size[0]}x{size[1]}x{size[2]}: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmarks() 