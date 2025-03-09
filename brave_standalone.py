#!/usr/bin/env python
"""
Standalone Brave Search MCP server.

This script creates a dedicated MCP server for Brave Search that can be run
independently from the main server.
"""

import os
import sys
from mcp.server.fastmcp import FastMCP

# Configure logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BraveSearch")

# Create an MCP server named "brave-search"
mcp = FastMCP("brave-search")

# Import our Brave Search implementation
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from server.tools.brave_search import register_brave_search_tools

# Register the tools with our server
register_brave_search_tools(mcp)

if __name__ == "__main__":
    # Check for API key
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        logger.error("Error: BRAVE_API_KEY environment variable is required")
        sys.exit(1)
        
    logger.info("Starting Brave Search MCP server")
    logger.info("Tools registered: brave_web_search, brave_local_search")
    
    # Run the server using stdio transport
    mcp.run(transport='stdio')