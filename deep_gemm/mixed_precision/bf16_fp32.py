"""
BF16 to FP32 GEMM operations for DeepGEMM.

This module provides optimized GEMM kernels that operate on BF16 inputs
and accumulate in FP32 for higher precision.
"""

import torch
from ..jit import compiler
from .. import jit_kernels
from ..utils import bench

# Template for BF16->FP32 GEMM kernel
BF16_FP32_GEMM_TEMPLATE = """
#include <cuda_bf16.h>

extern "C" __global__ void ${kernel_name}(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta
) {
    const int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (m_idx < M && n_idx < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; ++k) {
            sum += __bfloat162float(A[m_idx * K + k]) * __bfloat162float(B[n_idx * K + k]);
        }
        
        const int c_idx = m_idx * N + n_idx;
        C[c_idx] = alpha * sum + beta * C[c_idx];
    }
}
"""

def gemm_bf16_fp32_nt(a, b, out=None, alpha=1.0, beta=0.0):
    """
    Performs GEMM operation with BF16 inputs and FP32 accumulation.
    
    Args:
        a: First input tensor in BF16 format
        b: Second input tensor in BF16 format
        out: Output tensor in FP32 format, will be created if not provided
        alpha: Scalar multiplier for the product of input tensors
        beta: Scalar multiplier for the output
    
    Returns:
        The result tensor in FP32 format
    """
    # Convert inputs to correct format if needed
    if a.dtype != torch.bfloat16:
        a = a.to(torch.bfloat16)
    if b.dtype != torch.bfloat16:
        b = b.to(torch.bfloat16)
    
    # Get dimensions
    m, k = a.shape
    n, k2 = b.shape
    
    assert k == k2, f"Inner dimensions must match: got {k} and {k2}"
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    else:
        assert out.shape == (m, n), f"Output shape mismatch: expected {(m, n)}, got {out.shape}"
        assert out.dtype == torch.float32, f"Output must be float32, got {out.dtype}"
    
    # Get kernel
    kernel_name = "gemm_bf16_fp32_nt"
    module = compiler.get_cached_module(
        BF16_FP32_GEMM_TEMPLATE,
        kernel_name=kernel_name,
        dtype=(torch.bfloat16, torch.bfloat16, torch.float32),
    )
    
    # Set grid and block dimensions
    # This is a simple implementation; for production use, would need tuning
    block_dim = (16, 16, 1)
    grid_dim = ((m + block_dim[0] - 1) // block_dim[0], 
                (n + block_dim[1] - 1) // block_dim[1], 
                1)
    
    # Execute kernel
    getattr(module, kernel_name)[grid_dim, block_dim](
        a, b, out, m, n, k, alpha, beta
    )
    
    return out

def benchmark_bf16_fp32(m, n, k, num_warmups=5, num_runs=10):
    """
    Benchmark the BF16->FP32 GEMM operation.
    
    Args:
        m, n, k: Matrix dimensions
        num_warmups: Number of warmup iterations
        num_runs: Number of benchmark iterations
    
    Returns:
        Average execution time in milliseconds
    """
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
    
    def run_gemm():
        gemm_bf16_fp32_nt(a, b)
    
    return bench(run_gemm, num_warmups=num_warmups, num_tests=num_runs) 