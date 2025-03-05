"""
Configuration loader for distributed GEMM operations.

This module provides utilities for loading and validating configuration files
for distributed GEMM operations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
import torch.distributed as dist

from .api import DistributedConfig, ShardingStrategy


# Configure logging
logger = logging.getLogger("deep_gemm.distributed.config")


# Default configuration path
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), 
    "configs", 
    "default_config.yaml"
)


def load_config_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def create_distributed_config(
    config_path: Optional[Union[str, Path]] = None,
    override_args: Optional[Dict[str, Any]] = None
) -> DistributedConfig:
    """
    Create a DistributedConfig object from a configuration file.
    
    Args:
        config_path: Path to configuration file (uses default if None)
        override_args: Dictionary of arguments to override from config file
        
    Returns:
        DistributedConfig object
        
    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If configuration file is invalid
        ValueError: If configuration is invalid
    """
    # Load default configuration
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    # Load configuration from file
    try:
        config = load_config_from_file(config_path)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.warning(f"Error loading configuration file: {e}")
        logger.warning("Using default configuration")
        config = {}
    
    # Extract distributed configuration
    distributed_config = config.get("distributed", {})
    
    # Apply overrides
    if override_args:
        for key, value in override_args.items():
            if key in distributed_config:
                distributed_config[key] = value
    
    # Map strategy string to enum
    strategy_map = {
        "row": ShardingStrategy.ROW_PARALLEL,
        "column": ShardingStrategy.COLUMN_PARALLEL,
        "fully_sharded": ShardingStrategy.FULLY_SHARDED
    }
    
    strategy = distributed_config.get("strategy", "row")
    if isinstance(strategy, str):
        if strategy not in strategy_map:
            valid_strategies = ", ".join(strategy_map.keys())
            raise ValueError(f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}")
        strategy = strategy_map[strategy]
    
    # Create distributed configuration
    return DistributedConfig(
        strategy=strategy,
        backend=distributed_config.get("backend", "nccl"),
        master_addr=distributed_config.get("master_addr", "localhost"),
        master_port=int(distributed_config.get("master_port", 12355)),
        timeout=float(distributed_config.get("timeout_seconds", 1800.0)),
        device_type=distributed_config.get("device_type", "cuda")
    )


def configure_logging(config_path: Optional[Union[str, Path]] = None) -> None:
    """
    Configure logging for distributed operations.
    
    Args:
        config_path: Path to configuration file (uses default if None)
    """
    # Load configuration
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
        
    try:
        config = load_config_from_file(config_path)
    except (FileNotFoundError, yaml.YAMLError):
        config = {}
    
    # Get logging configuration
    logging_config = config.get("logging", {})
    log_level = logging_config.get("level", "INFO")
    
    # Set up logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure deep_gemm logger
    deep_gemm_logger = logging.getLogger("deep_gemm")
    deep_gemm_logger.setLevel(level)
    
    logger.info(f"Configured logging with level {log_level}")


def init_from_config(
    config_path: Optional[Union[str, Path]] = None,
    override_args: Optional[Dict[str, Any]] = None
) -> DistributedConfig:
    """
    Initialize distributed environment from configuration file.
    
    Args:
        config_path: Path to configuration file (uses default if None)
        override_args: Dictionary of arguments to override from config file
        
    Returns:
        DistributedConfig object
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Configure logging
    configure_logging(config_path)
    
    # Create distributed configuration
    config = create_distributed_config(config_path, override_args)
    
    logger.info(f"Initialized distributed configuration: {config}")
    return config 