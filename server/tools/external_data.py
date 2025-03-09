"""
External data tools for the MCP server.

This module contains tools for fetching and processing external data from APIs.
"""

from mcp.server.fastmcp import FastMCP
import httpx
from typing import Dict, Any
from utils.api_helpers import process_weather_data
import os
import json
import httpx
from typing import Dict, Any, List
from mcp.server.fastmcp import FastMCP
from decouple import config

def register_external_data_tools(mcp: FastMCP) -> None:
    """
    Register all external data tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """

    @mcp.tool()
    async def fetch_weather(latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Fetch current weather information for a location using coordinates
        
        Parameters:
        -----------
        latitude : float
            Geographic latitude
        longitude : float
            Geographic longitude
            
        Returns:
        --------
        Dict
            Processed weather data including temperature, humidity, and wind speed
        """
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={latitude}&longitude={longitude}&current_weather=true&"
            f"hourly=temperature_2m,relative_humidity_2m,wind_speed_10m"
        )
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            data = response.json()
            
            # Use the helper function to process the data
            return process_weather_data(data)