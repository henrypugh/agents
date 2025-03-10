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

from .server_config import ServerConfigManager
from .server_connection import ServerConnection

logger = logging.getLogger("ServerManager")

class ServerManager:
    """Manages connections to MCP servers"""
    
    def __init__(self):
        """Initialize the server manager"""
        self.servers = {}  # Dictionary to store server connections
        self.exit_stack = AsyncExitStack()
        self.config_manager = ServerConfigManager()
        logger.info("ServerManager initialized")
        
    def get_server(self, server_name: str) -> Optional[ServerConnection]:
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
        
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up server connections")
        await self.exit_stack.aclose()
    
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
        server = ServerConnection(server_name, session)
        await server.initialize()
        
        # Store server connection
        self.servers[server_name] = server
        
        logger.info(f"Connected to server {server_name} with tools: {server.get_tool_names()}")
        return server_name

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
            server = ServerConnection(server_name, session)
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

    async def _create_server_session(self, server_params: StdioServerParameters) -> ClientSession:
        """
        Create and initialize a server session
        
        Args:
            server_params: Parameters for the server
            
        Returns:
            Initialized client session
        """
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session
        
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