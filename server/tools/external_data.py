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
    async def brave_web_search(query: str) -> Dict[str, Any]:
        """Perform a web search using Brave Search API"""
        # Get API key from environment variables
        api_key = config("BRAVE_API_KEY")
        if not api_key:
            return {
                "error": "BRAVE_API_KEY not found in environment",
                "message": "Please set the BRAVE_API_KEY environment variable"
            }
            
        try:
            # Prepare the API request
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json", 
                "X-Subscription-Token": api_key
            }
            params = {
                "q": query,
                "count": 10,  # Number of results to return
                "offset": 0   # Starting position
                # Remove these parameters as they might be causing the 422 error
                # "search_lang": "en",
                # "ui_lang": "en",
                # "safesearch": "moderate"
            }
            
            # Make the API request
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                
                data = response.json()
                
                # Format the results for easier consumption
                results = []
                if "web" in data and "results" in data["web"]:
                    for item in data["web"]["results"]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "is_family_friendly": item.get("is_family_friendly", True),
                            "age": item.get("age", "")
                        })
                
                # Format news results if available
                news_results = []
                if "news" in data and "results" in data["news"]:
                    for item in data["news"]["results"]:
                        news_results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "source": item.get("source", ""),
                            "age": item.get("age", "")
                        })
                
                # Build a rich response object with all available data
                response_data = {
                    "query": query,
                    "results": results,
                    "news": news_results,
                    "total_results": data.get("web", {}).get("total", 0),
                    "query_refined": data.get("query", {}).get("refined", query)
                }
                
                return response_data
                
        except httpx.HTTPStatusError as e:
            return {
                "error": f"HTTP error: {e.response.status_code}",
                "message": str(e)
            }
        except httpx.RequestError as e:
            return {
                "error": "Request error",
                "message": str(e)
            }
        except json.JSONDecodeError:
            return {
                "error": "JSON decode error",
                "message": "Could not parse API response"
            }
        except Exception as e:
            return {
                "error": "Unknown error",
                "message": str(e)
            }
    
    @mcp.tool()
    async def brave_local_search(query: str, location: str) -> Dict[str, Any]:
        """
        Search for local businesses and services using Brave Search
        
        Parameters:
        -----------
        query : str
            The search query (e.g., "restaurants", "plumbers")
        location : str
            Location for the search (e.g., "New York, NY")
            
        Returns:
        --------
        Dict
            Local search results
        """
        # Get API key from environment variables
        api_key = os.environ.get("BRAVE_API_KEY")
        if not api_key:
            return {
                "error": "BRAVE_API_KEY not found in environment",
                "message": "Please set the BRAVE_API_KEY environment variable"
            }
            
        try:
            # Prepare the API request for local search
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json", 
                "X-Subscription-Token": api_key
            }
            
            # Combine query and location
            combined_query = f"{query} in {location}"
            
            params = {
                "q": combined_query,
                "count": 10,
                "search_lang": "en",
                "ui_lang": "en"
            }
            
            # Make the API request
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Format the results for easier consumption
                results = []
                if "web" in data and "results" in data["web"]:
                    for item in data["web"]["results"]:
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("url", ""),
                            "description": item.get("description", ""),
                            "is_family_friendly": item.get("is_family_friendly", True)
                        })
                
                return {
                    "query": combined_query,
                    "location": location,
                    "results": results,
                    "total_results": data.get("web", {}).get("total", 0)
                }
                
        except Exception as e:
            return {
                "error": "Error in local search",
                "message": str(e)
            }
    
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