"""
Health-related tools for the MCP server.

This module contains tools for health calculations and metrics.
"""

from mcp.server.fastmcp import FastMCP
from typing import Dict, Any

def register_health_tools(mcp: FastMCP) -> None:
    """
    Register all health tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.tool(category="Health Calculations")
    def calculate_bmi(weight_kg: float, height_m: float) -> Dict[str, Any]:
        """
        Calculate BMI (Body Mass Index) given weight and height
        
        Parameters:
        -----------
        weight_kg : float
            Weight in kilograms
        height_m : float
            Height in meters
            
        Returns:
        --------
        Dict
            Dictionary containing BMI value and category
        """
        bmi = weight_kg / (height_m ** 2)
        
        # Add BMI category for better context
        category = ""
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
            
        return {
            "bmi": round(bmi, 1),
            "category": category
        }