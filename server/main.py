"""
Main entry point for the MCP server.

This module initializes the FastMCP server and registers all tools, resources,
and prompts from their respective modules.
"""

from mcp.server.fastmcp import FastMCP

# Create a single MCP server instance
mcp = FastMCP("MCP Demo Server")

# Import and register all tools, resources, and prompts
from server.tools import register_all_tools
from server.resources import register_all_resources

# Register components with the MCP server
register_all_tools(mcp)
register_all_resources(mcp)

if __name__ == "__main__":
    mcp.run()