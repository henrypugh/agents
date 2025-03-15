"""
Server Registry module for managing MCP server connections.

This module provides Pydantic-validated registry for MCP server connections,
managing connection lifecycle and tool discovery.
"""

import logging
import os
import shutil
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Set, cast, TypeVar, Type, Protocol
from contextlib import AsyncExitStack
from enum import Enum, auto
from pydantic import TypeAdapter, ValidationError

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.schemas import (
    ServerInfo,
    ServerToolInfo,
    ServerStatus,
    ServerName
)
from src.client.server_instance import ServerInstance
from src.client.server_config import ServerConfig
from src.utils.decorators import server_connection, resource_cleanup

logger = logging.getLogger(__name__)

# Default connection settings
DEFAULT_CONNECTION_TIMEOUT = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_DELAY = 1.0

class ConnectionSettings:
    """Settings for server connections"""
    def __init__(
        self,
        timeout: float = DEFAULT_CONNECTION_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay


class ServerRegistryConfig:
    """Configuration for ServerRegistry"""
    def __init__(
        self,
        connection_settings: Optional[ConnectionSettings] = None,
        validate_servers: bool = True,
        ignore_connection_errors: bool = False
    ):
        self.connection_settings = connection_settings or ConnectionSettings()
        self.validate_servers = validate_servers
        self.ignore_connection_errors = ignore_connection_errors


class ServerConnectionError(Exception):
    """Exception raised for server connection errors"""
    def __init__(self, server_name: str, message: str):
        self.server_name = server_name
        self.message = message
        super().__init__(f"Error connecting to server '{server_name}': {message}")


class ServerRegistry:
    """
    Manages connections to MCP servers using Pydantic validation.
    
    This class handles:
    - Server discovery and connection management
    - Tool collection across servers
    - Server configuration and environment management
    - Resource cleanup
    """
    
    def __init__(self, config: Optional[ServerRegistryConfig] = None):
        """
        Initialize the server registry
        
        Args:
            config: Optional configuration for the registry
        """
        self.servers: Dict[str, ServerInstance] = {}  # Dictionary to store server connections
        self.exit_stack = AsyncExitStack()  # For shared/global resources
        self.config_manager = ServerConfig()
        self.config = config or ServerRegistryConfig()
        
        # Create type adapters for validation
        self.server_info_validator = TypeAdapter(ServerInfo)
        self.server_tool_info_validator = TypeAdapter(ServerToolInfo)
        
        logger.info("ServerRegistry initialized")
        
    def get_server(self, server_name: str) -> Optional[ServerInstance]:
        """
        Get a server connection by name
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server instance or None if not found
        """
        return self.servers.get(server_name)
        
    def collect_all_tools(self) -> List[Dict[str, Any]]:
        """
        Collect tools from all connected servers in OpenAI format
        
        Returns:
            List of tools in OpenAI format
        """
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.get_openai_format_tools())
        
        return all_tools
    
    def collect_all_tools_as_models(self) -> List[ServerToolInfo]:
        """
        Collect tools from all connected servers as Pydantic models
        
        Returns:
            List of ServerToolInfo models
        """
        all_tools = []
        try:
            for server in self.servers.values():
                server_tools = server.get_tools_as_models()
                
                # Validate tools if configured
                if self.config.validate_servers:
                    for tool in server_tools:
                        try:
                            validated_tool = self.server_tool_info_validator.validate_python(tool)
                            all_tools.append(validated_tool)
                        except ValidationError as e:
                            logger.warning(f"Invalid tool from server {server.server_name}: {e}")
                else:
                    all_tools.extend(server_tools)
                    
            return all_tools
        except Exception as e:
            logger.error(f"Error collecting tools as models: {e}")
            return []
        
    def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """
        Get tools from a specific server
        
        Args:
            server_name: Name of the server
            
        Returns:
            List of tools in OpenAI format
        """
        server = self.get_server(server_name)
        if server:
            return server.get_openai_format_tools()
        return []
    
    def get_connected_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about connected servers
        
        Returns:
            Dictionary of server information
        """
        result = {}
        for server_name, server in self.servers.items():
            result[server_name] = {
                "tools": server.get_openai_format_tools(),
                "connected_at": server.connected_at.isoformat(),
                "status": ServerStatus.CONNECTED
            }
        return result
    
    def get_connected_servers_as_models(self) -> List[ServerInfo]:
        """
        Get connected servers as Pydantic models
        
        Returns:
            List of ServerInfo models
        """
        try:
            server_infos = []
            for server in self.servers.values():
                server_info = server.get_server_info_as_model()
                
                # Validate if configured
                if self.config.validate_servers:
                    try:
                        validated_info = self.server_info_validator.validate_python(server_info)
                        server_infos.append(validated_info)
                    except ValidationError as e:
                        logger.warning(f"Invalid server info for {server.server_name}: {e}")
                else:
                    server_infos.append(server_info)
                    
            return server_infos
        except Exception as e:
            logger.error(f"Error getting servers as models: {e}")
            return []
    
    def get_primary_server(self) -> Optional[str]:
        """
        Get the name of the primary server if any
        
        Returns:
            Server name or None
        """
        # Use the first server as primary if available
        if self.servers:
            return next(iter(self.servers.keys()))
        return None
    
    async def get_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available servers from configuration
        
        Returns:
            Dictionary of server information
        """
        try:
            # Get server configurations
            config = self.config_manager.load_config()
            
            # Format for response
            available_servers = {}
            for server_name, server_config in config.items():
                is_connected = server_name in self.servers
                available_servers[server_name] = {
                    "connected": is_connected,
                    "type": "configured",
                    "source": "config",
                    "command": server_config.get("command", "unknown"),
                    "status": ServerStatus.CONNECTED if is_connected else ServerStatus.DISCONNECTED
                }
                
            return available_servers
        except Exception as e:
            logger.error(f"Error getting available servers: {e}")
            return {}
    
    async def get_available_configured_servers(self) -> List[str]:
        """
        Get list of available configured servers
        
        Returns:
            List of server names
        """
        try:
            # Get server configurations
            config = self.config_manager.load_config()
            return list(config.keys())
        except Exception as e:
            logger.error(f"Error getting configured servers: {e}")
            return []
    
    async def get_available_servers_as_models(self) -> List[ServerInfo]:
        """
        Get all available servers as Pydantic models
        
        Returns:
            List of ServerInfo models
        """
        try:
            available_servers = []
            
            # Get server configurations
            config = self.config_manager.load_config()
            
            # Convert to ServerInfo models
            for server_name, _ in config.items():
                is_connected = server_name in self.servers
                
                if is_connected:
                    # Server is already connected, use its model
                    available_servers.append(self.servers[server_name].get_server_info_as_model())
                else:
                    # Server is not connected
                    try:
                        server_info = ServerInfo(
                            name=server_name,
                            connected=False,
                            status=ServerStatus.DISCONNECTED,
                            tools=[]
                        )
                        
                        # Validate if configured
                        if self.config.validate_servers:
                            validated_info = self.server_info_validator.validate_python(server_info)
                            available_servers.append(validated_info)
                        else:
                            available_servers.append(server_info)
                            
                    except ValidationError as e:
                        logger.warning(f"Invalid server info for {server_name}: {e}")
                
            return available_servers
        except Exception as e:
            logger.error(f"Error getting available servers as models: {e}")
            return []
    
    @server_connection
    async def disconnect_server(self, server_name: str) -> Dict[str, Any]:
        """
        Disconnect a specific server
        
        Args:
            server_name: Name of the server to disconnect
            
        Returns:
            Status dictionary
        """
        logger.info(f"Disconnecting server: {server_name}")
        
        # Track the disconnection operation
        Traceloop.set_association_properties({
            "server_name": server_name,
            "disconnect_operation": "started"
        })
        
        if server_name not in self.servers:
            return {
                "status": "not_connected",
                "server": server_name,
                "message": f"Server '{server_name}' is not connected"
            }
            
        try:
            # Get server instance
            server = self.servers[server_name]
            
            # Cleanup server resources
            await server.cleanup()
            
            # Remove from servers dictionary
            self.servers.pop(server_name)
            
            return {
                "status": "disconnected",
                "server": server_name,
                "message": f"Successfully disconnected from server: {server_name}"
            }
        except Exception as e:
            error_msg = f"Error disconnecting from server '{server_name}': {str(e)}"
            logger.error(error_msg)
            
            # Remove from servers dictionary even if cleanup failed
            self.servers.pop(server_name, None)
            
            return {
                "status": "error",
                "server": server_name,
                "error": error_msg
            }
        
    @resource_cleanup
    async def cleanup(self) -> None:
        """Clean up all resources"""
        logger.info("Cleaning up server connections")
        
        # Track cleanup start
        Traceloop.set_association_properties({
            "cleanup_operation": "all_servers",
            "server_count": len(self.servers)
        })
        
        # Keep track of any errors that occur
        errors = []
        
        # Clean up each server individually
        for server_name, server in list(self.servers.items()):
            try:
                await server.cleanup()
            except Exception as e:
                errors.append((server_name, str(e)))
                logger.error(f"Error cleaning up server '{server_name}': {e}")
        
        # Clear servers dictionary after all cleanup attempts
        self.servers.clear()
        
        # Clean up global resources with the main exit stack
        await self.exit_stack.aclose()
        
        # Log any errors that occurred
        if errors:
            logger.warning(f"Encountered {len(errors)} errors during cleanup: {errors}")
    
    @server_connection
    async def connect_to_server(self, server_script_path: str) -> str:
        """
        Connect to an MCP server using a script path
        
        Args:
            server_script_path: Path to server script
            
        Returns:
            Server name
            
        Raises:
            ServerConnectionError: If connection fails
        """
        logger.info(f"Connecting to server with script: {server_script_path}")
        
        # Validate script file extension and get command
        command, server_params = self._create_script_server_params(server_script_path)
        
        # Generate server name from script path
        server_name = os.path.basename(server_script_path).split('.')[0]
        
        # Apply connection retry logic
        retries = 0
        last_error = None
        
        while retries <= self.config.connection_settings.max_retries:
            try:
                # Connect and create server connection
                session = await self._create_server_session(
                    server_params, 
                    timeout=self.config.connection_settings.timeout
                )
                server = ServerInstance(server_name, session)
                await server.initialize()
                
                # Store server connection
                self.servers[server_name] = server
                
                logger.info(f"Connected to server {server_name} with tools: {server.get_tool_names()}")
                return server_name
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.config.connection_settings.max_retries:
                    logger.warning(f"Connection attempt {retries} failed for {server_name}: {e}")
                    await asyncio.sleep(self.config.connection_settings.retry_delay * retries)
                else:
                    logger.error(f"Failed to connect to server {server_name} after {retries} attempts: {e}")
                    break
                    
        # If we get here, all retries failed
        error_message = f"Failed to connect after {retries} attempts: {last_error}"
        
        if self.config.ignore_connection_errors:
            logger.error(error_message)
            return f"error:{server_name}"
        else:
            raise ServerConnectionError(server_name, error_message)

    @server_connection
    async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
        """
        Connect to an MCP server defined in configuration
        
        Args:
            server_name: Name of the server
            
        Returns:
            Status dictionary
            
        Raises:
            ServerConnectionError: If connection fails and ignore_errors is False
        """
        logger.info(f"Connecting to configured server: {server_name}")
        
        # Check if already connected
        if server_name in self.servers:
            logger.info(f"Server {server_name} is already connected")
            tools = self.servers[server_name].get_tool_names()
            
            return {
                "status": "already_connected",
                "server": server_name,
                "tools": tools,
                "tool_count": len(tools)
            }
        
        # Apply connection retry logic
        retries = 0
        last_error = None
        
        while retries <= self.config.connection_settings.max_retries:
            try:
                # Get server configuration
                server_config = self.config_manager.get_server_config(server_name)
                
                # Create server parameters
                server_params = self._create_config_server_params(server_name, server_config)
                
                # Connect and create server connection
                connection_start = time.time()
                session = await self._create_server_session(
                    server_params,
                    timeout=self.config.connection_settings.timeout
                )
                server = ServerInstance(server_name, session)
                await server.initialize()
                connection_time = time.time() - connection_start
                
                # Store server connection
                self.servers[server_name] = server
                
                # Get tool names for response
                tool_names = server.get_tool_names()
                
                # Track connection metrics
                Traceloop.set_association_properties({
                    "connection_time": connection_time,
                    "tool_count": len(tool_names)
                })
                
                return {
                    "status": "connected",
                    "server": server_name,
                    "tools": tool_names,
                    "tool_count": len(tool_names)
                }
                
            except Exception as e:
                last_error = e
                retries += 1
                
                if retries <= self.config.connection_settings.max_retries:
                    logger.warning(f"Connection attempt {retries} failed for {server_name}: {e}")
                    await asyncio.sleep(self.config.connection_settings.retry_delay * retries)
                else:
                    logger.error(f"Failed to connect to server {server_name} after {retries} attempts: {e}")
                    break
                    
        # If we get here, all retries failed
        error_message = f"Failed to connect after {retries} attempts: {last_error}"
        
        if self.config.ignore_connection_errors:
            logger.error(error_message)
            return {
                "status": "error",
                "server": server_name,
                "error": error_message
            }
        else:
            raise ServerConnectionError(server_name, error_message)

    @task(name="create_server_session")
    async def _create_server_session(
        self, 
        server_params: StdioServerParameters,
        timeout: float = DEFAULT_CONNECTION_TIMEOUT
    ) -> ClientSession:
        """
        Create and initialize a server session
        
        Args:
            server_params: Server parameters
            timeout: Connection timeout in seconds
            
        Returns:
            Initialized client session
            
        Raises:
            TimeoutError: If connection times out
            Exception: Other connection errors
        """
        # Track the server session creation
        if hasattr(server_params, 'command') and hasattr(server_params, 'args'):
            Traceloop.set_association_properties({
                "command": server_params.command,
                "args": " ".join(server_params.args) if server_params.args else ""
            })
        
        try:
            # Connect with timeout
            stdio_transport = await asyncio.wait_for(
                self.exit_stack.enter_async_context(stdio_client(server_params)),
                timeout=timeout
            )
            
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))
            
            # Initialize with timeout
            await asyncio.wait_for(
                session.initialize(),
                timeout=timeout
            )
            
            return session
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Connection timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Failed to create server session: {e}")
        
    @task(name="create_script_server_params")
    def _create_script_server_params(self, script_path: str) -> Tuple[str, StdioServerParameters]:
        """
        Create server parameters for a script path
        
        Args:
            script_path: Path to server script
            
        Returns:
            Tuple of (command, server_params)
            
        Raises:
            ValueError: If script path is invalid
        """
        # Validate script path exists
        if not os.path.exists(script_path):
            raise ValueError(f"Server script not found: {script_path}")
            
        # Validate script file extension
        is_python = script_path.endswith('.py')
        is_js = script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        # Set up server parameters
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None
        )
        
        return command, server_params
        
    @task(name="create_config_server_params")
    def _create_config_server_params(self, server_name: str, server_config: Dict[str, Any]) -> StdioServerParameters:
        """
        Create server parameters from configuration
        
        Args:
            server_name: Name of the server
            server_config: Server configuration dictionary
            
        Returns:
            Server parameters
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate required fields
        if 'command' not in server_config:
            raise ValueError(f"Server '{server_name}' configuration missing 'command' field")
            
        # Find command in PATH
        command = server_config['command']
        command_path = shutil.which(command)
        if not command_path:
            raise ValueError(f"Command '{command}' not found in PATH")
            
        # Set up server parameters
        args = server_config.get('args', [])
        env = server_config.get('env', {}).copy() if server_config.get('env') else {}
        
        # Process environment variables
        processed_env = self.config_manager.process_environment_variables(env)
        
        # Merge with current environment
        env = {**os.environ, **processed_env}
        
        return StdioServerParameters(
            command=command_path,
            args=args,
            env=env
        )