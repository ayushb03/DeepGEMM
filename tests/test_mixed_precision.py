"""
Tests for mixed precision GEMM operations.
"""

import pytest
import torch
import numpy as np
from deep_gemm.mixed_precision import gemm_fp16_fp32_nt, gemm_bf16_fp32_nt

# Skip tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

@requires_cuda
def test_fp16_fp32_gemm_correctness():
    """Test FP16->FP32 GEMM against PyTorch reference implementation."""
    # Create test matrices
    m, n, k = 32, 32, 32
    a = torch.randn((m, k), dtype=torch.float32, device='cuda')
    b = torch.randn((n, k), dtype=torch.float32, device='cuda')
    
    # Convert to FP16 for our implementation
    a_fp16 = a.to(torch.float16)
    b_fp16 = b.to(torch.float16)
    
    # Reference implementation using PyTorch
    reference = torch.matmul(a, b.t())
    
    # Our implementation
    result = gemm_fp16_fp32_nt(a_fp16, b_fp16)
    
    # Check results
    # We expect some precision differences due to FP16 conversion
    assert torch.allclose(reference, result, rtol=1e-2, atol=1e-2)

@requires_cuda
def test_bf16_fp32_gemm_correctness():
    """Test BF16->FP32 GEMM against PyTorch reference implementation."""
    # Create test matrices
    m, n, k = 32, 32, 32
    a = torch.randn((m, k), dtype=torch.float32, device='cuda')
    b = torch.randn((n, k), dtype=torch.float32, device='cuda')
    
    # Convert to BF16 for our implementation
    a_bf16 = a.to(torch.bfloat16)
    b_bf16 = b.to(torch.bfloat16)
    
    # Reference implementation using PyTorch
    reference = torch.matmul(a, b.t())
    
    # Our implementation
    result = gemm_bf16_fp32_nt(a_bf16, b_bf16)
    
    # Check results
    # We expect some precision differences due to BF16 conversion
    assert torch.allclose(reference, result, rtol=1e-1, atol=1e-1)

@requires_cuda
def test_fp16_fp32_gemm_dimensions():
    """Test FP16->FP32 GEMM with different matrix dimensions."""
    for m, n, k in [(16, 32, 64), (64, 32, 16), (128, 128, 128)]:
        a = torch.randn((m, k), dtype=torch.float16, device='cuda')
        b = torch.randn((n, k), dtype=torch.float16, device='cuda')
        
        # Check output shape
        result = gemm_fp16_fp32_nt(a, b)
        assert result.shape == (m, n)
        assert result.dtype == torch.float32

@requires_cuda
def test_bf16_fp32_gemm_dimensions():
    """Test BF16->FP32 GEMM with different matrix dimensions."""
    for m, n, k in [(16, 32, 64), (64, 32, 16), (128, 128, 128)]:
        a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
        b = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
        
        # Check output shape
        result = gemm_bf16_fp32_nt(a, b)
        assert result.shape == (m, n)
        assert result.dtype == torch.float32

@requires_cuda
def test_fp16_fp32_gemm_alpha_beta():
    """Test FP16->FP32 GEMM with different alpha and beta values."""
    m, n, k = 32, 32, 32
    a = torch.randn((m, k), dtype=torch.float16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float16, device='cuda')
    c = torch.randn((m, n), dtype=torch.float32, device='cuda')
    
    alpha, beta = 2.0, 0.5
    
    # Reference implementation using PyTorch
    a_ref = a.to(torch.float32)
    b_ref = b.to(torch.float32)
    reference = alpha * torch.matmul(a_ref, b_ref.t()) + beta * c
    
    # Our implementation
    result = gemm_fp16_fp32_nt(a, b, out=c.clone(), alpha=alpha, beta=beta)
    
    # Check results
    assert torch.allclose(reference, result, rtol=1e-2, atol=1e-2) 