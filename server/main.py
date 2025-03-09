"""
Main entry point for the MCP server.

This module initializes the FastMCP server and registers all tools, resources,
and prompts from their respective modules.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from mcp.server.fastmcp import FastMCP

# Import for cleanup function
from tools.server_connector import cleanup_all_connections

# Define lifespan manager for proper cleanup
@asynccontextmanager
async def server_lifespan(_: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle."""
    try:
        # Initialize resources on startup
        yield {}  # No context needed for now
    finally:
        # Clean up on shutdown
        await cleanup_all_connections()

# Create a single MCP server instance with lifespan support
mcp = FastMCP("MCP Demo Server", lifespan=server_lifespan)

# Import and register all tools, resources, and prompts
from tools import register_all_tools
from resources import register_all_resources

# Register components with the MCP server
register_all_tools(mcp)
register_all_resources(mcp)

if __name__ == "__main__":
    mcp.run()