# Production Deployment Guide

This guide provides detailed instructions for deploying DeepGEMM distributed operations in production environments.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Deployment Options](#deployment-options)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Examples](#examples)
8. [Troubleshooting](#troubleshooting)

## Requirements

### Hardware Requirements

- CUDA-compatible GPUs (recommended: NVIDIA A100, V100, or newer)
- Recommended: NVLink or high-bandwidth inter-GPU connections
- For multi-node: High-speed networking (InfiniBand preferred, at least 100 Gb Ethernet)

### Software Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA 11.0+
- NCCL 2.10+
- For MPI support: OpenMPI 4.0+ or MPICH 3.4+

## Installation

### Using pip

```bash
pip install deep-gemm
```

### From Source

For the latest features or custom builds:

```bash
# Clone the repository
git clone https://github.com/yourusername/DeepGEMM.git
cd DeepGEMM

# Install dependencies (with uv for better performance)
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Docker Deployment

We provide Docker images for easy deployment:

```bash
# Pull the Docker image
docker pull yourusername/deep-gemm:latest

# Run with GPU support
docker run --gpus all -it yourusername/deep-gemm:latest
```

## Configuration

DeepGEMM distributed operations can be configured using YAML configuration files. Below is an example configuration:

```yaml
distributed:
  # Sharding strategy: row, column, or fully_sharded
  strategy: "row"
  
  # Backend for PyTorch distributed: nccl, gloo, mpi
  backend: "nccl"
  
  # Master node address and port
  master_addr: "localhost"
  master_port: 12355
  
  # Timeout in seconds
  timeout_seconds: 1800
  
  # Device type: cuda, cpu
  device_type: "cuda"

performance:
  # Use automatic mixed precision
  use_amp: true
  
  # Use tensor cores for matrix multiplication
  use_tensor_cores: true
  
  # Default values for benchmarks
  benchmark_warmup: 5
  benchmark_iterations: 10

logging:
  # Logging level: DEBUG, INFO, WARNING, ERROR
  level: "INFO"
  
  # Logging options
  log_distributed_init: true
  log_tensor_shapes: true
  log_performance: true
```

Save this configuration to a file (e.g., `config.yaml`) and use it with the DeepGEMM API:

```python
from deep_gemm.distributed import init_from_config, DistributedGEMM

# Initialize from configuration file
config = init_from_config("config.yaml")

# Create a DistributedGEMM instance
with DistributedGEMM(config) as dgemm:
    # Use distributed GEMM operations
    result = dgemm.gemm_fp16_fp32_nt(A, B)
```

## Deployment Options

### Single-Node Multi-GPU

For deploying on a single node with multiple GPUs:

```bash
# Using the launcher script
python -m tools.launch_distributed \
    --nproc_per_node=4 \
    --module examples.production_deployment \
    --args="--matrix-size 8192 --precision fp16"
```

### Multi-Node Deployment

For deploying across multiple nodes:

```bash
# On the master node (node_rank=0)
python -m tools.launch_distributed \
    --nnodes=2 \
    --node_rank=0 \
    --nproc_per_node=4 \
    --master_addr="192.168.1.10" \
    --module examples.production_deployment \
    --args="--matrix-size 8192 --precision fp16"

# On the worker node (node_rank=1)
python -m tools.launch_distributed \
    --nnodes=2 \
    --node_rank=1 \
    --nproc_per_node=4 \
    --master_addr="192.168.1.10" \
    --module examples.production_deployment \
    --args="--matrix-size 8192 --precision fp16"
```

### Using MPI

For deployment with MPI (useful for HPC environments):

```bash
# Using the launcher script with MPI backend
python -m tools.launch_distributed \
    --backend=mpi \
    --nnodes=2 \
    --nproc_per_node=4 \
    --module examples.production_deployment \
    --args="--matrix-size 8192 --precision fp16"

# Or directly with mpirun
mpirun -np 8 python -m examples.production_deployment \
    --matrix-size 8192 --precision fp16
```

## Performance Tuning

### Sharding Strategies

DeepGEMM supports three sharding strategies:

1. **Row-Parallel (default)**: Shards matrix A by rows
   - Best for models with large batch sizes
   - Minimizes communication during forward pass

2. **Column-Parallel**: Shards matrix B by columns
   - Best for models with large hidden dimensions
   - Minimizes communication during backward pass

3. **Fully-Sharded**: Shards both matrices
   - Best for very large models
   - Requires a number of GPUs that is a perfect square

Choose the strategy based on your model architecture and communication patterns.

### Memory Optimization

- Use mixed precision (FP16 or BF16) to reduce memory usage and improve performance
- Adjust tensor layouts for optimal memory access patterns
- Consider using tensor cores for faster matrix multiplication

### Communication Optimization

- Use NCCL backend for GPU-to-GPU communication
- Enable NVLink if available for faster intra-node communication
- For multi-node deployments, use InfiniBand or high-speed networking

## Monitoring and Debugging

### Logging

DeepGEMM includes a comprehensive logging system. You can configure logging levels and outputs in the configuration file:

```yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_distributed_init: true
  log_tensor_shapes: true
  log_performance: true
```

### Debugging Tools

- Use the verification script to check if your distributed setup is working correctly:

```bash
python -m tools.verify_setup --config-path config.yaml --verify-result
```

- Enable detailed logging for debugging:

```bash
python -m tools.launch_distributed \
    --nproc_per_node=4 \
    --module examples.production_deployment \
    --args="--config-path config.yaml --log-level DEBUG"
```

## Examples

### Basic Usage

```python
import torch
from deep_gemm.distributed import DistributedGEMM, DistributedConfig, ShardingStrategy

# Create configuration
config = DistributedConfig(
    strategy=ShardingStrategy.ROW_PARALLEL,
    world_size=4
)

# Initialize distributed environment
with DistributedGEMM(config) as dgemm:
    # Create input matrices
    A = torch.randn(1024, 4096, dtype=torch.float16, device="cuda")
    B = torch.randn(4096, 4096, dtype=torch.float16, device="cuda")
    
    # Perform distributed GEMM
    result = dgemm.gemm_fp16_fp32_nt(A, B)
    
    # Use the result
    print(f"Result shape: {result.shape}")
```

### Advanced Usage

See the `examples/production_deployment.py` script for a complete example of production deployment.

## Troubleshooting

### Common Issues

1. **Initialization Timeout**
   
   ```
   ProcessGroupError: Failed to initialize process group: Timed out waiting for response
   ```
   
   **Solution**: Check network connectivity between nodes and ensure the master node is accessible. Increase timeout in the configuration.

2. **NCCL Errors**
   
   ```
   RuntimeError: NCCL error in: /pytorch/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:XXX, unhandled system error
   ```
   
   **Solution**: Make sure all GPUs are visible and NCCL can communicate between them. Check GPU isolation in containers.

3. **Uneven Tensor Division**
   
   ```
   ShapeMismatchError: Cannot evenly scatter dimension X of size Y across Z processes
   ```
   
   **Solution**: Ensure matrix dimensions are divisible by the number of processes, or use padding.

### Support

For additional support:

- File an issue on the GitHub repository
- Check the API documentation
- Join the community Discord/Slack channel

---

## Further Reading

- [API Reference](api_reference.md)
- [Performance Benchmarks](benchmarks.md)
- [Contributing Guide](contributing.md) 