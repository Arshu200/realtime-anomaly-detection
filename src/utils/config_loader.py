"""
Configuration loading and management utilities.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from loguru import logger


class ConfigLoader:
    """Configuration loader with support for multiple config files and environment variables."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_name: Name of config file (without .yaml extension)
            
        Returns:
            Configuration dictionary
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            logger.warning(f"Configuration file not found: {config_file}")
            return {}
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Process environment variable substitutions
            config = self._substitute_env_vars(config)
            
            self.configs[config_name] = config
            logger.info(f"Loaded configuration from {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_file}: {e}")
            return {}
    
    def load_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files from the config directory.
        
        Returns:
            Dictionary mapping config names to their configurations
        """
        if not self.config_dir.exists():
            logger.warning(f"Configuration directory not found: {self.config_dir}")
            return {}
        
        configs = {}
        
        for config_file in self.config_dir.glob("*.yaml"):
            config_name = config_file.stem
            config = self.load_config(config_name)
            if config:
                configs[config_name] = config
        
        return configs
    
    def get_config(self, config_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Get configuration, loading if not already loaded.
        
        Args:
            config_name: Name of configuration
            reload: Whether to reload even if already loaded
            
        Returns:
            Configuration dictionary
        """
        if config_name not in self.configs or reload:
            return self.load_config(config_name)
        
        return self.configs[config_name]
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations into one.
        
        Args:
            *config_names: Names of configurations to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for config_name in config_names:
            config = self.get_config(config_name)
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _substitute_env_vars(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Args:
            config: Configuration value (dict, list, or string)
            
        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_var_string(config)
        else:
            return config
    
    def _substitute_env_var_string(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: String that may contain environment variable references
            
        Returns:
            String with environment variables substituted
        """
        # Handle ${VAR_NAME} format
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            default_value = None
            
            # Handle ${VAR_NAME:default_value} format
            if ":" in env_var:
                env_var, default_value = env_var.split(":", 1)
            
            return os.getenv(env_var, default_value or value)
        
        return value
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result


def load_config(config_path: Union[str, Path] = "config/config.yaml", 
                config_dir: str = "config") -> Dict[str, Any]:
    """
    Convenience function to load a single configuration file.
    
    Args:
        config_path: Path to configuration file
        config_dir: Configuration directory for ConfigLoader
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    # If config_path is absolute or doesn't exist, try direct loading
    if config_path.is_absolute() or config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Process environment variable substitutions
            loader = ConfigLoader(config_dir)
            config = loader._substitute_env_vars(config)
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            return {}
    
    # Otherwise, use ConfigLoader
    loader = ConfigLoader(config_dir)
    config_name = config_path.stem if config_path.suffix else str(config_path)
    return loader.load_config(config_name)


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "database.host")
        default: Default value if key not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split(".")
    value = config
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default