import json
import os
from typing import Dict, Any
from decouple import config
import logging

logger = logging.getLogger("ServerConfig")

class ServerConfig:
    """Manages server configurations and environment variables"""
    
    def __init__(self, config_path: str = "server_config.json"):
        self.config_path = config_path
        self.config_cache = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load and parse server configuration file
        
        Returns:
            Dict containing server configurations
            
        Raises:
            ValueError: If the configuration file cannot be loaded or parsed
        """
        if self.config_cache is not None:
            return self.config_cache
            
        try:
            with open(self.config_path, 'r') as f:
                server_config_json = json.load(f)
                
            # Check for mcpServers structure (Claude Desktop format)
            if "mcpServers" in server_config_json:
                self.config_cache = server_config_json["mcpServers"]
            else:
                self.config_cache = server_config_json
                
            return self.config_cache
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load server configuration: {e}")
            
    def get_server_config(self, server_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific server
        
        Args:
            server_name: Name of the server in the configuration
            
        Returns:
            Dict containing server configuration
            
        Raises:
            ValueError: If the server is not found in configuration
        """
        config = self.load_config()
        
        if server_name not in config:
            raise ValueError(f"Server '{server_name}' not found in configuration")
            
        return config[server_name]
        
    def process_environment_variables(self, env_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Process environment variables with ${VAR} syntax
        
        Args:
            env_dict: Dictionary of environment variables to process
            
        Returns:
            Dict with processed environment variables
        """
        processed_env = {}
        
        for key, value in env_dict.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_value = os.getenv(env_var_name)
                if env_value:
                    processed_env[key] = env_value
                    if key == 'BRAVE_API_KEY':
                        logger.info(f"[API KEY SOURCE] Using {key} from environment variable: ${{{env_var_name}}}")
                    else:
                        logger.info(f"Resolved env var {env_var_name} for {key}")
                else:
                    logger.warning(f"Environment variable {env_var_name} not found")
            else:
                processed_env[key] = value
                
        return processed_env
