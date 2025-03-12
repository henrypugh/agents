# server/tools/__init__.py
from mcp.server.fastmcp import FastMCP

# Import all tool modules
from tools.math_tools import register_math_tools
from tools.health_tools import register_health_tools
from tools.external_data import register_external_data_tools
from tools.server_connector import register_server_connector_tools
from tools.football_tools import register_football_tools
from tools.filesystem_tools import register_filesystem_tools

def register_all_tools(mcp: FastMCP) -> None:
    """Register all tool functions with the MCP server."""
    register_math_tools(mcp)
    register_health_tools(mcp)
    register_external_data_tools(mcp)
    register_server_connector_tools(mcp)
    register_football_tools(mcp)
    register_filesystem_tools(mcp)