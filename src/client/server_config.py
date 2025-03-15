"""
Server Configuration module for managing server configurations and environment variables.
"""
import json
import os
from typing import Dict, Any, Optional
import logging

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("ServerConfig")

class EnvVar(BaseModel):
    """Model for environment variable configuration"""
    name: str
    value: str
    required: bool = False
    
class ServerConfigModel(BaseModel):
    """Model for server configuration"""
    command: str = Field(..., description="Command to run the server")
    args: list[str] = Field(default_factory=list, description="Arguments to pass to the command")
    env: Dict[str, str] = Field(default_factory=dict, description="Environment variables for the server")
    
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
    
    def get_server_config_as_model(self, server_name: str) -> Optional[ServerConfigModel]:
        """
        Get configuration for a specific server as a Pydantic model
        
        Args:
            server_name: Name of the server in the configuration
            
        Returns:
            ServerConfigModel or None if validation fails
        """
        try:
            config_dict = self.get_server_config(server_name)
            return ServerConfigModel(**config_dict)
        except ValidationError as e:
            logger.error(f"Error validating server config for {server_name}: {e}")
            return None
        except ValueError as e:
            logger.error(f"Error getting server config: {e}")
            return None
        
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
                    if key != 'BRAVE_API_KEY':
                        logger.info(f"Resolved env var {env_var_name} for {key}")
                else:
                    logger.warning(f"Environment variable {env_var_name} not found")
            else:
                processed_env[key] = value
                
        return processed_env
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save server configuration to file
        
        Args:
            config: Server configuration dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a structure that matches the expected format
            output_config = {}
            if "mcpServers" in self.load_config():
                output_config["mcpServers"] = config
            else:
                output_config = config
                
            with open(self.config_path, 'w') as f:
                json.dump(output_config, f, indent=2)
                
            # Update cache
            self.config_cache = config
            return True
        except Exception as e:
            logger.error(f"Error saving server configuration: {e}")
            return False