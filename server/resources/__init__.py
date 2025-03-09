"""
Resources package for MCP server.

This module imports and exports all resource functions to be registered with the MCP server.
"""

from mcp.server.fastmcp import FastMCP

# Import all resource modules
from server.resources.greetings import register_greeting_resources

def register_all_resources(mcp: FastMCP) -> None:
    """
    Register all resource functions with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    register_greeting_resources(mcp)