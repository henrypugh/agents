"""
Server connector tools for the MCP server.

This module contains tools for dynamically connecting to external MCP servers.
"""

import json
import os
import logging
import asyncio
import httpx
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack

from mcp.server.fastmcp import FastMCP, Context
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("ServerConnector")

# Global storage for server connections
# Maps server_name -> {session, stack, tools}
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
        available to the LLM. Use this when you need specialised functionality like web search.
        
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
        if server_name in _server_connections and _server_connections[server_name]['session']:
            tools = _server_connections[server_name]['tools']
            ctx.info(f"Server {server_name} already connected with {len(tools)} tools")
            return {
                "status": "already_connected",
                "server": server_name,
                "tools": [t["function"]["name"] for t in tools]
            }
            
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
        except (json.JSONDecodeError, FileNotFoundError) as e:
            return {
                "status": "error",
                "error": f"Failed to load server configuration: {str(e)}"
            }
            
        # Check if requested server exists in config
        if server_name not in servers_config:
            return {
                "status": "error",
                "error": f"Server '{server_name}' not found in configuration"
            }
            
        server_config = servers_config[server_name]
        
        # Validate required fields
        if 'command' not in server_config:
            return {
                "status": "error",
                "error": f"Server '{server_name}' configuration missing 'command' field"
            }
            
        try:
            # Create stack for resource management
            stack = AsyncExitStack()
            
            # Set up server parameters
            args = server_config.get('args', [])
            env = server_config.get('env', {}).copy() if server_config.get('env') else {}
            
            # For Brave Search, handle API key specially
            if server_name == "brave-search" and 'BRAVE_API_KEY' not in env:
                # Try to get from environment
                api_key = os.environ.get('BRAVE_API_KEY')
                if not api_key:
                    return {
                        "status": "error",
                        "error": "BRAVE_API_KEY not found in environment"
                    }
                env['BRAVE_API_KEY'] = api_key
                
            # Merge with current environment
            env = {**os.environ, **env}
            
            ctx.info(f"Connecting to server {server_name}")
            
            # Prepare server command
            command = server_config['command']
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env
            )
            
            # Connect to server
            stdio_transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await stack.enter_async_context(ClientSession(read, write))
            
            # Initialize the connection
            await session.initialize()
            
            # List available tools
            response = await session.list_tools()
            tools = await _convert_tools_to_openai_format(response.tools)
            
            # Store connection for future use
            _server_connections[server_name] = {
                'session': session,
                'stack': stack,
                'tools': tools
            }
            
            tool_names = [t["function"]["name"] for t in tools]
            ctx.info(f"Successfully connected to {server_name} with tools: {tool_names}")
            
            return {
                "status": "connected",
                "server": server_name,
                "tools": tool_names,
                "toolCount": len(tools)
            }
            
        except Exception as e:
            ctx.info(f"Error connecting to server {server_name}: {str(e)}")
            
            # Clean up any partially established connection
            if server_name in _server_connections:
                if 'stack' in _server_connections[server_name]:
                    try:
                        await _server_connections[server_name]['stack'].aclose()
                    except Exception:
                        pass
                del _server_connections[server_name]
                
            return {
                "status": "error",
                "error": f"Failed to connect to server: {str(e)}"
            }
    
    @mcp.tool()
    async def execute_external_tool(
        server_name: str,
        tool_name: str,
        tool_args: Dict[str, Any],
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
        tool_args : Dict[str, Any]
            Arguments to pass to the tool
            
        Returns:
        --------
        Dict
            Result of the tool execution
        """
        # Check if server is connected
        if server_name not in _server_connections or not _server_connections[server_name]['session']:
            return {
                "status": "error",
                "error": f"Server {server_name} is not connected. Use connect_to_server first."
            }
            
        session = _server_connections[server_name]['session']
        
        try:
            # Execute the tool
            ctx.info(f"Executing tool {tool_name} on server {server_name}")
            tool_result = await session.call_tool(tool_name, tool_args)
            
            # Process the result
            result_text = ""
            if hasattr(tool_result, 'content') and tool_result.content:
                # Extract text content
                for content_item in tool_result.content:
                    if hasattr(content_item, 'text'):
                        result_text += content_item.text
            
            return {
                "status": "success",
                "server": server_name,
                "tool": tool_name,
                "result": result_text
            }
            
        except Exception as e:
            ctx.info(f"Error executing tool {tool_name} on server {server_name}: {str(e)}")
            return {
                "status": "error", 
                "error": f"Failed to execute tool: {str(e)}"
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
        if server_name not in _server_connections or not _server_connections[server_name]['session']:
            return {
                "status": "error",
                "error": f"Server {server_name} is not connected. Use connect_to_server first."
            }
            
        tools = _server_connections[server_name]['tools']
        
        # Extract tool information
        tool_info = []
        for tool in tools:
            tool_info.append({
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {})
            })
            
        return {
            "status": "success",
            "server": server_name,
            "tools": tool_info,
            "toolCount": len(tools)
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
            if 'stack' in _server_connections[server_name]:
                await _server_connections[server_name]['stack'].aclose()
                
            # Remove from connections
            del _server_connections[server_name]
            
            ctx.info(f"Successfully disconnected from server {server_name}")
            return {
                "status": "disconnected",
                "server": server_name
            }
            
        except Exception as e:
            ctx.info(f"Error disconnecting from server {server_name}: {str(e)}")
            return {
                "status": "error",
                "error": f"Failed to disconnect: {str(e)}"
            }

    @mcp.tool()
    async def get_available_servers(ctx: Context) -> Dict[str, Any]:
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
                    "connected": name in _server_connections and _server_connections[name]['session'] is not None,
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

# Helper functions
async def _convert_tools_to_openai_format(mcp_tools) -> List[Dict[str, Any]]:
    """Convert MCP tools to OpenAI format for the LLM client"""
    openai_tools = []
    
    for tool in mcp_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema
            }
        }
        openai_tools.append(openai_tool)
        
    return openai_tools

# Function to clean up all server connections
async def cleanup_all_connections():
    """Clean up all server connections"""
    cleanup_tasks = []
    
    for server_name, connection in _server_connections.items():
        if 'stack' in connection:
            cleanup_tasks.append(connection['stack'].aclose())
            
    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
    _server_connections.clear()