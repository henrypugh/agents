"""
Math-related tools for the MCP server.

This module contains tools for performing mathematical operations.
"""

from mcp.server.fastmcp import FastMCP
from typing import Callable, Any

def register_math_tools(mcp: FastMCP) -> None:
    """
    Register all math tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.tool(category="Math Operations")
    def add(a: int, b: int) -> int:
        """
        Add two numbers together
        
        Parameters:
        -----------
        a : int
            First number to add
        b : int
            Second number to add
            
        Returns:
        --------
        int
            Sum of a and b
        """
        return a + b
    
    # Additional math tools can be added here
    # For example:
    
    @mcp.tool(category="Math Operations")
    def multiply(a: int, b: int) -> int:
        """
        Multiply two numbers
        
        Parameters:
        -----------
        a : int
            First number to multiply
        b : int
            Second number to multiply
            
        Returns:
        --------
        int
            Product of a and b
        """
        return a * b