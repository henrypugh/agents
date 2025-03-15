"""
Server Manager module for handling MCP server connections.
"""

import logging
import os
import shutil
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from .server_config import ServerConfig
from .server_instance import ServerInstance
from src.utils.decorators import server_connection, resource_cleanup

logger = logging.getLogger("ServerRegistry")

class ServerRegistry:
    """Manages connections to MCP servers"""
    
    def __init__(self):
        """Initialize the server manager"""
        self.servers = {}  # Dictionary to store server connections
        self.exit_stack = AsyncExitStack()  # For shared/global resources
        self.config_manager = ServerConfig()
        logger.info("ServerRegistry initialized")
        
    def get_server(self, server_name: str) -> Optional[ServerInstance]:
        """
        Get a server connection by name
        
        Args:
            server_name: Name of the server
            
        Returns:
            Server connection or None if not found
        """
        return self.servers.get(server_name)
        
    def collect_all_tools(self) -> List[Dict[str, Any]]:
        """
        Collect tools from all connected servers
        
        Returns:
            List of all available tools in OpenAI format
        """
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.get_openai_format_tools())
        
        num_servers = len(self.servers)
        num_tools = len(all_tools)
        if num_servers > 0:
            logger.info(f"Collected {num_tools} tools from {num_servers} servers")
        else:
            logger.info("No servers connected yet. Only server management tools will be available.")
            
        return all_tools
        
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
    
    @server_connection
    async def disconnect_server(self, server_name: str) -> Dict[str, Any]:
        """
        Disconnect a specific server
        
        Args:
            server_name: Name of the server to disconnect
            
        Returns:
            Dictionary with status information
        """
        logger.info(f"Disconnecting server: {server_name}")
        
        # Track the disconnection operation
        Traceloop.set_association_properties({
            "server_name": server_name,
            "disconnect_operation": "started"
        })
        
        if server_name not in self.servers:
            logger.warning(f"Server '{server_name}' not found, cannot disconnect")
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
            
            # Track successful disconnection
            Traceloop.set_association_properties({
                "disconnect_operation": "success"
            })
            
            logger.info(f"Successfully disconnected from server: {server_name}")
            return {
                "status": "disconnected",
                "server": server_name,
                "message": f"Successfully disconnected from server: {server_name}"
            }
        except Exception as e:
            error_msg = f"Error disconnecting from server '{server_name}': {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Track disconnection error
            Traceloop.set_association_properties({
                "disconnect_operation": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
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
                logger.info(f"Cleaning up server: {server_name}")
                await server.cleanup()
                logger.info(f"Successfully cleaned up server: {server_name}")
            except Exception as e:
                error_msg = f"Error cleaning up server '{server_name}': {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append((server_name, str(e)))
        
        # Clear servers dictionary after all cleanup attempts
        self.servers.clear()
        
        # Clean up global resources with the main exit stack
        await self.exit_stack.aclose()
        
        # Track cleanup completion
        Traceloop.set_association_properties({
            "cleanup_complete": True,
            "error_count": len(errors)
        })
        
        # Log summary of cleanup
        if errors:
            logger.warning(f"Completed cleanup with {len(errors)} errors")
        else:
            logger.info("Successfully cleaned up all server connections")
    
    @server_connection
    async def connect_to_server(self, server_script_path: str) -> str:
        """
        Connect to an MCP server using a script path
        
        Args:
            server_script_path: Path to the server script (.py or .js)
            
        Returns:
            Name of the connected server
            
        Raises:
            ValueError: If the script is invalid
        """
        logger.info(f"Connecting to server with script: {server_script_path}")
        
        # Validate script file extension and get command
        command, server_params = self._create_script_server_params(server_script_path)
        
        # Generate server name from script path
        server_name = os.path.basename(server_script_path).split('.')[0]
        
        # Connect and create server connection
        session = await self._create_server_session(server_params)
        server = ServerInstance(server_name, session)
        await server.initialize()
        
        # Store server connection
        self.servers[server_name] = server
        
        logger.info(f"Connected to server {server_name} with tools: {server.get_tool_names()}")
        return server_name

    @server_connection
    async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
        """
        Connect to an MCP server defined in configuration
        
        Args:
            server_name: Name of the server in the config file
            
        Returns:
            Dictionary with connection status and details
            
        Raises:
            ValueError: If the server configuration is invalid
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
        
        try:
            # Get server configuration
            server_config = self.config_manager.get_server_config(server_name)
            
            # Create server parameters
            server_params = self._create_config_server_params(server_name, server_config)
            
            # Connect and create server connection
            session = await self._create_server_session(server_params)
            server = ServerInstance(server_name, session)
            await server.initialize()
            
            # Store server connection
            self.servers[server_name] = server
            
            # Get tool names for response
            tool_names = server.get_tool_names()
            logger.info(f"Connected to server {server_name} with tools: {tool_names}")
            
            return {
                "status": "connected",
                "server": server_name,
                "tools": tool_names,
                "tool_count": len(tool_names)
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to configured server {server_name}: {error_msg}", exc_info=True)
            
            return {
                "status": "error",
                "server": server_name,
                "error": error_msg
            }

    @task(name="create_server_session")
    async def _create_server_session(self, server_params: StdioServerParameters) -> ClientSession:
        """
        Create and initialize a server session
        
        Args:
            server_params: Parameters for the server
            
        Returns:
            Initialized client session
        """
        # Track the server session creation
        if hasattr(server_params, 'command') and hasattr(server_params, 'args'):
            Traceloop.set_association_properties({
                "command": server_params.command,
                "args": " ".join(server_params.args) if server_params.args else ""
            })
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session
        
    @task(name="create_script_server_params")
    def _create_script_server_params(self, script_path: str) -> tuple[str, StdioServerParameters]:
        """
        Create server parameters for a script path
        
        Args:
            script_path: Path to the server script
            
        Returns:
            Tuple of (command, server_params)
            
        Raises:
            ValueError: If the script has an invalid extension
        """
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
            ValueError: If the configuration is invalid
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
        
        # Log API key status if this is the Brave Search server
        if server_name == "brave-search" and 'BRAVE_API_KEY' in processed_env:
            key_preview = processed_env['BRAVE_API_KEY'][:4] + "..." if processed_env['BRAVE_API_KEY'] else "None"
            logger.info(f"[API KEY STATUS] Brave API key is set (starts with: {key_preview})")
        
        # Merge with current environment
        env = {**os.environ, **processed_env}
        
        return StdioServerParameters(
            command=command_path,
            args=args,
            env=env
        )