#!/usr/bin/env python
"""
Example of using DeepGEMM's distributed operations in a simple model training scenario.

This script demonstrates how to use distributed GEMM operations for a multi-layer
perceptron model trained on random data, showing how to leverage multiple GPUs
for faster matrix multiplications.

Run this with:
    python tools/launch_distributed.py --script examples/distributed_training.py --world-size 2
"""

import argparse
import math
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import deep_gemm.distributed as dist_gemm


class DistributedLinear(nn.Module):
    """
    Linear layer that uses distributed GEMM operations for forward pass.
    
    Implements a distributed version of nn.Linear by sharding the weight matrix
    across multiple GPUs.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        strategy: dist_gemm.ShardingStrategy = dist_gemm.ShardingStrategy.ROW_PARALLEL
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.strategy = strategy
        
        # Create local shard of the weight matrix
        world_size = dist_gemm.get_world_size()
        rank = dist_gemm.get_rank()
        
        # Determine shard size based on strategy
        if strategy == dist_gemm.ShardingStrategy.ROW_PARALLEL:
            # Shard output features
            shard_size = (out_features + world_size - 1) // world_size
            start_idx = rank * shard_size
            end_idx = min((rank + 1) * shard_size, out_features)
            
            if start_idx < out_features:
                self.weight = nn.Parameter(torch.empty(end_idx - start_idx, in_features))
                if bias:
                    self.bias = nn.Parameter(torch.empty(end_idx - start_idx))
                else:
                    self.register_parameter('bias', None)
            else:
                # This rank doesn't need parameters
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
                
        elif strategy == dist_gemm.ShardingStrategy.COLUMN_PARALLEL:
            # Shard input features
            shard_size = (in_features + world_size - 1) // world_size
            start_idx = rank * shard_size
            end_idx = min((rank + 1) * shard_size, in_features)
            
            if start_idx < in_features:
                self.weight = nn.Parameter(torch.empty(out_features, end_idx - start_idx))
                if bias and rank == 0:
                    # Only rank 0 needs the bias
                    self.bias = nn.Parameter(torch.empty(out_features))
                else:
                    self.register_parameter('bias', None)
            else:
                # This rank doesn't need parameters
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
            
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")
        
        # Initialize parameters if they exist
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize the parameters using Kaiming uniform initialization."""
        if self.weight is not None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using distributed GEMM operations.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.strategy == dist_gemm.ShardingStrategy.ROW_PARALLEL:
            # Each rank computes a portion of the output rows
            if self.weight is not None:
                # Convert to appropriate precision
                if x.dtype == torch.float16 or x.dtype == torch.float32:
                    x_half = x.to(torch.float16)
                    weight_half = self.weight.to(torch.float16)
                    
                    # Compute local output
                    local_output = dist_gemm.distributed_gemm_fp16_fp32_nt(
                        x_half, weight_half, strategy=self.strategy, sync=False
                    )
                    
                    # Add bias if necessary
                    if self.bias is not None:
                        local_output = local_output + self.bias.unsqueeze(0)
                        
                elif x.dtype == torch.bfloat16:
                    x_bf16 = x
                    weight_bf16 = self.weight.to(torch.bfloat16)
                    
                    # Compute local output
                    local_output = dist_gemm.distributed_gemm_bf16_fp32_nt(
                        x_bf16, weight_bf16, strategy=self.strategy, sync=False
                    )
                    
                    # Add bias if necessary
                    if self.bias is not None:
                        local_output = local_output + self.bias.unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported dtype: {x.dtype}")
            else:
                # This rank doesn't have parameters, return empty tensor
                local_output = torch.empty(
                    (x.size(0), 0), 
                    dtype=torch.float32, 
                    device=x.device
                )
            
            # Gather results from all ranks
            return dist_gemm.communication.all_gather_matrix(local_output, dim=1)
            
        elif self.strategy == dist_gemm.ShardingStrategy.COLUMN_PARALLEL:
            # Each rank computes using a portion of the input features
            if self.weight is not None:
                # Convert to appropriate precision
                if x.dtype == torch.float16 or x.dtype == torch.float32:
                    x_half = x.to(torch.float16)
                    weight_half = self.weight.to(torch.float16)
                    
                    # Compute local output
                    local_output = dist_gemm.distributed_gemm_fp16_fp32_nt(
                        x_half, weight_half, strategy=self.strategy, sync=False
                    )
                    
                elif x.dtype == torch.bfloat16:
                    x_bf16 = x
                    weight_bf16 = self.weight.to(torch.bfloat16)
                    
                    # Compute local output
                    local_output = dist_gemm.distributed_gemm_bf16_fp32_nt(
                        x_bf16, weight_bf16, strategy=self.strategy, sync=False
                    )
                else:
                    raise ValueError(f"Unsupported dtype: {x.dtype}")
            else:
                # This rank doesn't have parameters, return zeros
                local_output = torch.zeros(
                    (x.size(0), self.out_features), 
                    dtype=torch.float32, 
                    device=x.device
                )
            
            # Sum results from all ranks
            output = dist_gemm.communication.allreduce_matrix(local_output)
            
            # Add bias if necessary (only on rank 0)
            if self.bias is not None:
                output = output + self.bias.unsqueeze(0)
                
            return output


class DistributedMLP(nn.Module):
    """
    Multi-layer perceptron model using distributed GEMM operations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        strategy: dist_gemm.ShardingStrategy = dist_gemm.ShardingStrategy.ROW_PARALLEL
    ):
        super().__init__()
        
        # Create layers
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                DistributedLinear(
                    layer_dims[i], 
                    layer_dims[i + 1],
                    strategy=strategy
                )
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP model."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply ReLU to all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                
        return x


def generate_data(
    batch_size: int,
    input_dim: int,
    output_dim: int,
    dtype: torch.dtype = torch.float16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate random data for training.
    
    Args:
        batch_size: Number of examples
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        dtype: Data type for the input data
        
    Returns:
        Tuple of input data and target data
    """
    device = f"cuda:{dist_gemm.get_rank()}" if torch.cuda.is_available() else "cpu"
    
    # Generate random input data
    x = torch.randn(batch_size, input_dim, dtype=dtype, device=device)
    
    # Generate random target data
    y = torch.randn(batch_size, output_dim, dtype=torch.float32, device=device)
    
    return x, y


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    num_batches: int,
    dtype: torch.dtype = torch.float16
) -> float:
    """
    Train the model for one epoch.
    
    Args:
        model: Model to train
        optimizer: Optimizer to use
        batch_size: Batch size
        input_dim: Input dimension
        output_dim: Output dimension
        num_batches: Number of batches in the epoch
        dtype: Data type for inputs
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    # Train for num_batches batches
    for _ in range(num_batches):
        # Generate random data
        x, y = generate_data(batch_size, input_dim, output_dim, dtype)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        
        # Compute loss (MSE)
        loss = F.mse_loss(output, y)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
    
    return total_loss / num_batches


def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Distributed GEMM MLP Example")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--input-dim", type=int, default=1024, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=512, help="Output dimension")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers")
    parser.add_argument("--num-epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of batches per epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--strategy", type=str, default="row", choices=["row", "column"], 
                       help="Sharding strategy (row or column)")
    parser.add_argument("--port", type=int, default=12355, help="Port for distributed communication")
    
    args = parser.parse_args()
    
    # Initialize distributed environment
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # Initialize process group if not using torch.distributed.launch
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(args.port)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                               rank=rank, 
                               world_size=world_size)
    
    # Set device for this process
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cpu"
    
    # Convert strategy string to enum
    if args.strategy == "row":
        strategy = dist_gemm.ShardingStrategy.ROW_PARALLEL
    else:
        strategy = dist_gemm.ShardingStrategy.COLUMN_PARALLEL
    
    # Create model
    hidden_dims = [args.hidden_dim] * args.num_layers
    model = DistributedMLP(
        input_dim=args.input_dim,
        hidden_dims=hidden_dims,
        output_dim=args.output_dim,
        strategy=strategy
    ).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    for epoch in range(args.num_epochs):
        start_time = time.time()
        avg_loss = train_epoch(
            model=model,
            optimizer=optimizer,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            num_batches=args.num_batches,
            dtype=torch.float16
        )
        elapsed_time = time.time() - start_time
        
        # Print results (only on rank 0)
        if rank == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}, "
                 f"Loss: {avg_loss:.6f}, "
                 f"Time: {elapsed_time:.2f}s")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main() 