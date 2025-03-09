"""
Greeting resources for the MCP server.

This module contains resources for personalized greetings.
"""

from mcp.server.fastmcp import FastMCP

def register_greeting_resources(mcp: FastMCP) -> None:
    """
    Register all greeting resources with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.resource("greeting://{name}")
    def get_greeting(name: str) -> str:
        """
        Get a personalized greeting for a user
        
        Parameters:
        -----------
        name : str
            Name of the person to greet
            
        Returns:
        --------
        str
            Personalized greeting message
        """
        return f"Hello, {name}! Welcome to the MCP Demo Server."