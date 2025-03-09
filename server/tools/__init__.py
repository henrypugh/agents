"""
Tools package for MCP server.

This module imports and exports all tool functions to be registered with the MCP server.
"""

from mcp.server.fastmcp import FastMCP

# Import all tool modules
from server.tools.math_tools import register_math_tools
from server.tools.health_tools import register_health_tools
from server.tools.external_data import register_external_data_tools

def register_all_tools(mcp: FastMCP) -> None:
    """
    Register all tool functions with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    register_math_tools(mcp)
    register_health_tools(mcp)
    register_external_data_tools(mcp)