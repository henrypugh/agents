"""
Prompts package for MCP server.

This module imports and exports all prompt templates to be registered with the MCP server.
"""

from mcp.server.fastmcp import FastMCP

# Import all prompt modules (none currently implemented)
# from server.prompts.examples import register_example_prompts

def register_all_prompts(mcp: FastMCP) -> None:
    """
    Register all prompt templates with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    # No prompts currently implemented
    # register_example_prompts(mcp)
    pass