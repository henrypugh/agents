"""
Server connector tools for the MCP server.

This module contains tools for dynamically connecting to external MCP servers.
"""

import json
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("ServerConnector")

# Global storage for server connections - simplified structure
_server_connections = {}

def register_server_connector_tools(mcp: FastMCP) -> None:
    """
    Register all server connector tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.tool()
    async def connect_to_server(server_name: str, ctx: Context) -> Dict[str, Any]:
        """
        Connect to an external MCP server on demand
        
        This tool dynamically connects to another MCP server when needed, making its tools
        available to the LLM. Use this when you need specialised functionality.
        
        Parameters:
        -----------
        server_name : str
            Name of the server to connect to (e.g., "brave-search")
            
        Returns:
        --------
        Dict
            Information about the connected server and available tools
        """
        # Check if server is already connected
        if server_name in _server_connections:
            try:
                session = _server_connections[server_name].get('session')
                if session:
                    # Test the connection is still valid
                    tools_resp = await session.list_tools()
                    tool_names = [t.name for t in tools_resp.tools]
                    ctx.info(f"Server {server_name} already connected with {len(tool_names)} tools")
                    return {
                        "status": "already_connected",
                        "server": server_name,
                        "tools": tool_names
                    }
            except Exception as e:
                ctx.info(f"Existing connection to {server_name} is invalid: {e}. Will reconnect.")
                # Connection is no longer valid, remove it
                if server_name in _server_connections:
                    await _safe_cleanup_connection(server_name)
        
        # Load server configurations
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "server_config.json")
        try:
            with open(config_path, 'r') as f:
                server_config_json = json.load(f)
                
            # Handle different config formats
            if "mcpServers" in server_config_json:
                servers_config = server_config_json["mcpServers"]
            else:
                servers_config = server_config_json
        except (json.JSONDecodeError, FileNotFoundError) as e:
            ctx.info(f"Failed to load server configuration: {e}")
            return {
                "status": "error",
                "error": f"Failed to load server configuration: {str(e)}"
            }
            
        # Check if requested server exists in config
        if server_name not in servers_config:
            ctx.info(f"Server '{server_name}' not found in configuration")
            return {
                "status": "error",
                "error": f"Server '{server_name}' not found in configuration"
            }
            
        server_config = servers_config[server_name]
        
        # Validate required fields
        if 'command' not in server_config:
            ctx.info(f"Server '{server_name}' configuration missing 'command' field")
            return {
                "status": "error",
                "error": f"Server '{server_name}' configuration missing 'command' field"
            }
            
        try:
            # Import here to avoid circular dependencies
            from decouple import config as config_decouple
            
            # Create new connection
            ctx.info(f"Creating new connection to {server_name}")
            
            # Set up server parameters
            command = server_config['command']
            args = server_config.get('args', [])
            env = server_config.get('env', {}).copy() if server_config.get('env') else {}
            
            # Process environment variables - resolve ${VAR} syntax
            processed_env = {}
            for key, value in env.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var_name = value[2:-1]
                    env_value = os.getenv(env_var_name)
                    if env_value:
                        processed_env[key] = env_value
                        ctx.info(f"Resolved env var {env_var_name} for {key}")
                    else:
                        ctx.info(f"Warning: Environment variable {env_var_name} not found")
                else:
                    processed_env[key] = value
            
            # Special handling for Brave Search
            if server_name == "brave-search" and 'BRAVE_API_KEY' not in processed_env:
                try:
                    api_key = config_decouple('BRAVE_API_KEY', default=None)
                    if api_key:
                        processed_env['BRAVE_API_KEY'] = api_key
                        ctx.info("Added Brave API key from environment")
                    else:
                        return {
                            "status": "error",
                            "error": "BRAVE_API_KEY not found in environment"
                        }
                except Exception as e:
                    ctx.info(f"Error loading API key: {e}")
                    return {
                        "status": "error",
                        "error": f"Error loading API key: {str(e)}"
                    }
            
            # Merge with current environment
            merged_env = {**os.environ, **processed_env}
            
            # Prepare server command - ensure we have full path
            import shutil
            command_path = shutil.which(command)
            if not command_path:
                ctx.info(f"Command '{command}' not found in PATH")
                return {
                    "status": "error",
                    "error": f"Command '{command}' not found in PATH"
                }
            
            ctx.info(f"Starting server process: {command_path} {' '.join(args)}")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=command_path,
                args=args,
                env=merged_env
            )
            
            # Use new stack for each connection
            stack = AsyncExitStack()
            
            # Connect to server with timeouts and retries
            try:
                stdio_transport = await asyncio.wait_for(
                    stack.enter_async_context(stdio_client(server_params)),
                    timeout=30.0  # 30 second timeout
                )
                read, write = stdio_transport
                
                session = await asyncio.wait_for(
                    stack.enter_async_context(ClientSession(read, write)),
                    timeout=10.0  # 10 second timeout
                )
                
                # Initialize the connection
                await asyncio.wait_for(
                    session.initialize(),
                    timeout=10.0  # 10 second timeout
                )
                
                ctx.info(f"Successfully initialized session with {server_name}")
                
                # List available tools
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                ctx.info(f"Retrieved {len(tool_names)} tools from {server_name}: {tool_names}")
                
                # Create a simplified connection record
                _server_connections[server_name] = {
                    'session': session,
                    'stack': stack,
                    'tool_names': tool_names,
                    'connected_at': datetime.now().isoformat()
                }
                
                return {
                    "status": "connected",
                    "server": server_name,
                    "tools": tool_names,
                    "toolCount": len(tool_names)
                }
                
            except asyncio.TimeoutError:
                await stack.aclose()
                ctx.info(f"Timeout connecting to server {server_name}")
                return {
                    "status": "error",
                    "error": f"Timeout connecting to server {server_name}"
                }
                
            except Exception as e:
                ctx.info(f"Error connecting to server {server_name}: {e}")
                await stack.aclose()
                return {
                    "status": "error",
                    "error": f"Failed to connect to server: {str(e)}"
                }
                
        except Exception as e:
            ctx.info(f"Unexpected error connecting to {server_name}: {e}")
            return {
                "status": "error",
                "error": f"Unexpected error: {str(e)}"
            }
    
    @mcp.tool()
    async def execute_external_tool(
        server_name: str,
        tool_name: str,
        tool_args_json: str,
        ctx: Context
    ) -> Dict[str, Any]:
        """
        Execute a tool from an external MCP server
        
        Use this after connecting to an external server with connect_to_server.
        
        Parameters:
        -----------
        server_name : str
            Name of the server containing the tool
        tool_name : str
            Name of the tool to execute
        tool_args_json : str
            JSON string representing arguments to pass to the tool
                
        Returns:
        --------
        Dict
            Result of the tool execution
        """
        ctx.info(f"Executing external tool: {tool_name} on server {server_name}")
        
        # Validate server connection
        if server_name not in _server_connections:
            ctx.info(f"Server {server_name} not found in connections")
            return {
                "status": "error",
                "error": f"Server {server_name} is not connected. Use connect_to_server first."
            }
        
        conn_data = _server_connections[server_name]
        if 'session' not in conn_data or conn_data['session'] is None:
            ctx.info(f"Session for server {server_name} is missing or invalid")
            return {
                "status": "error",
                "error": f"Invalid session for server {server_name}. Try reconnecting."
            }
        
        session = conn_data['session']
        
        # Parse JSON with error handling
        try:
            tool_args = json.loads(tool_args_json)
            ctx.info(f"Parsed tool args: {tool_args}")
        except json.JSONDecodeError as e:
            ctx.info(f"JSON parse error: {e}")
            return {
                "status": "error",
                "error": f"Invalid JSON: {str(e)}"
            }
        
        # Execute tool with timeout and error handling
        try:
            # For Brave Search, simplify arguments if needed
            if server_name == "brave-search" and tool_name == "brave_web_search" and "query" in tool_args:
                # Keep it simple
                tool_args = {"query": tool_args["query"]}
                ctx.info(f"Simplified args for brave_web_search: {tool_args}")

            # Execute with timeout
            ctx.info(f"Calling tool on external server: {tool_name}")
            result = await asyncio.wait_for(
                session.call_tool(tool_name, tool_args),
                timeout=30.0  # 30 second timeout
            )
            
            # Process result carefully
            result_text = ""
            if hasattr(result, 'content'):
                # Handle different content types
                if isinstance(result.content, list):
                    for item in result.content:
                        if hasattr(item, 'text'):
                            result_text += item.text
                        else:
                            result_text += str(item)
                elif isinstance(result.content, str):
                    result_text = result.content
                else:
                    result_text = str(result.content)
            else:
                result_text = str(result)
            
            preview = result_text[:100] + "..." if len(result_text) > 100 else result_text
            ctx.info(f"Tool execution successful, result preview: {preview}")
            
            return {
                "status": "success",
                "server": server_name,
                "tool": tool_name,
                "result": result_text
            }
            
        except asyncio.TimeoutError:
            ctx.info(f"Timeout executing tool {tool_name}")
            return {
                "status": "error",
                "error": f"Timeout executing tool {tool_name}"
            }
            
        except Exception as e:
            ctx.info(f"Error executing tool {tool_name}: {e}")
            return {
                "status": "error",
                "error": f"Tool execution failed: {str(e)}"
            }
    
    @mcp.tool()
    async def get_external_server_tools(server_name: str, ctx: Context) -> Dict[str, Any]:
        """
        Get information about tools available on an external MCP server
        
        Use this to check what tools are available from a connected server.
        
        Parameters:
        -----------
        server_name : str
            Name of the server to query
            
        Returns:
        --------
        Dict
            Information about available tools
        """
        # Check if server is connected
        if server_name not in _server_connections:
            return {
                "status": "error",
                "error": f"Server {server_name} is not connected. Use connect_to_server first."
            }
            
        conn_data = _server_connections[server_name]
        if 'session' not in conn_data or conn_data['session'] is None:
            return {
                "status": "error",
                "error": f"Invalid session for server {server_name}. Try reconnecting."
            }
        
        session = conn_data['session']
        
        try:
            # Get fresh tool list
            tools_response = await session.list_tools()
            tools = tools_response.tools
            
            # Extract tool information
            tool_info = []
            for tool in tools:
                tool_info.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema
                })
                
            return {
                "status": "success",
                "server": server_name,
                "tools": tool_info,
                "toolCount": len(tools)
            }
        except Exception as e:
            ctx.info(f"Error getting tools from {server_name}: {e}")
            return {
                "status": "error",
                "error": f"Failed to get tools: {str(e)}"
            }
        
    @mcp.tool()
    async def disconnect_server(server_name: str, ctx: Context) -> Dict[str, Any]:
        """
        Disconnect from an external MCP server
        
        Use this to free up resources when a server is no longer needed.
        
        Parameters:
        -----------
        server_name : str
            Name of the server to disconnect
            
        Returns:
        --------
        Dict
            Status of the disconnection
        """
        if server_name not in _server_connections:
            return {
                "status": "not_connected",
                "message": f"Server {server_name} is not connected"
            }
            
        try:
            # Clean up resources
            await _safe_cleanup_connection(server_name)
            
            ctx.info(f"Successfully disconnected from server {server_name}")
            return {
                "status": "disconnected",
                "server": server_name
            }
            
        except Exception as e:
            ctx.info(f"Error disconnecting from server {server_name}: {e}")
            return {
                "status": "error",
                "error": f"Failed to disconnect: {str(e)}"
            }

    @mcp.tool()
    async def get_available_servers(
        ctx: Context,
        include_details: bool = True  # Add a parameter with default value AFTER ctx
    ) -> Dict[str, Any]:
        """
        Get information about available servers from configuration
        
        Returns:
        --------
        Dict
            Information about configured servers
        """
        # Load server configurations
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "server_config.json")
        try:
            with open(config_path, 'r') as f:
                server_config_json = json.load(f)
                
            # Check for mcpServers structure (Claude Desktop format)
            if "mcpServers" in server_config_json:
                servers_config = server_config_json["mcpServers"]
            else:
                servers_config = server_config_json
                
            # Get connected status
            server_info = {}
            for name, config in servers_config.items():
                server_info[name] = {
                    "connected": name in _server_connections and _server_connections[name].get('session') is not None,
                    "command": config.get("command", ""),
                    "args": config.get("args", [])
                }
                
            return {
                "status": "success",
                "availableServers": server_info,
                "connectedServers": list(_server_connections.keys())
            }
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return {
                "status": "error",
                "error": f"Failed to load server configuration: {str(e)}"
            }

# Helper function for safe cleanup
async def _safe_cleanup_connection(server_name: str) -> None:
    """Safely clean up a server connection"""
    if server_name in _server_connections:
        try:
            if 'stack' in _server_connections[server_name]:
                await _server_connections[server_name]['stack'].aclose()
        except Exception as e:
            logger.warning(f"Error closing stack for {server_name}: {e}")
        finally:
            _server_connections.pop(server_name, None)

# Function to clean up all server connections
async def cleanup_all_connections():
    """Clean up all server connections"""
    logger.info(f"Cleaning up {len(_server_connections)} server connections")
    
    # Make a copy of keys to avoid modification during iteration
    server_names = list(_server_connections.keys())
    
    for server_name in server_names:
        try:
            await _safe_cleanup_connection(server_name)
            logger.info(f"Cleaned up connection to {server_name}")
        except Exception as e:
            logger.error(f"Error cleaning up connection to {server_name}: {e}")