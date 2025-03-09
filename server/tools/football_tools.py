"""
External data tools for the MCP server.

This module contains tools for fetching and processing external data from APIs.
"""

from mcp.server.fastmcp import FastMCP
import httpx
from typing import Dict, Any
from utils.api_helpers import process_weather_data, process_fixture_data
import requests
from decouple import config

def register_football_tools(mcp: FastMCP) -> None:
    """
    Register all football tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """

    @mcp.tool()
    async def fetch_next_three_fixtures(league_id: int, season: int, team_id: int) -> Dict[str, Any]:
        """
        Fetch the next three fixtures for a given team in a given league
        
        Parameters:
        -----------
        league_id : int
            The ID of the league
        season : int
            The season
        team_id : int
            The ID of the team
            
        Returns:
        --------
        Dict
            Processed fixture data including teams, goals, and scores
        """
        def call_api(endpoint: str, params: dict = {}) -> dict:
            base_url = 'https://v3.football.api-sports.io'
            headers = {
                'x-rapidapi-host': "v3.football.api-sports.io",
                'x-rapidapi-key': config('API_FOOTBALL_KEY')
            }

            # Construct the full URL including the endpoint and query parameters
            url = f"{base_url}/{endpoint}"
            

            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                return data
            except requests.RequestException as e:
                return None
            
        data = call_api('fixtures', {'league': league_id, 'season': season, 'team':team_id, 'next': 3})

        data = process_fixture_data(data)
        return data

    
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