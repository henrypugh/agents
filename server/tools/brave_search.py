"""
Brave Search tools for the MCP server.

This module contains tools for performing web searches using the Brave Search API.
"""

import os
import httpx
import urllib.parse
from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP
import logging

logger = logging.getLogger("BraveSearch")

# Brave API base URL
BRAVE_API_BASE = "https://api.search.brave.com/res/v1"

async def make_brave_request(query: str, api_key: str, count: int = 10, offset: int = 0) -> Dict[str, Any]:
    """
    Make a request to the Brave API with proper error handling.
    
    Parameters:
    -----------
    query : str
        The search query
    api_key : str
        The Brave API key
    count : int
        Number of results to return
    offset : int
        Result offset for pagination
        
    Returns:
    --------
    Dict[str, Any]
        The JSON response or an error dict
    """
    if not api_key:
        return {"error": "Missing API key", "message": "BRAVE_API_KEY environment variable is required"}

    # URL-encode the query
    encoded_query = urllib.parse.quote(query)
    
    url = f"{BRAVE_API_BASE}/web/search"
    
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }
    
    params = {
        "q": query,  # Use the original query string
        "count": count,
        "offset": offset
    }
    
    logger.info(f"Making Brave Search request for query: {query}")
    logger.debug(f"Full request URL: {url}")
    logger.debug(f"Headers: {headers} (API key redacted)")
    logger.debug(f"Params: {params}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, 
                headers=headers, 
                params=params,
                timeout=30.0
            )
            
            # Log response status
            logger.info(f"Brave Search API response status: {response.status_code}")
            
            # Check if we got a successful response
            if response.status_code == 200:
                return response.json()
            else:
                error_text = f"HTTP error: {response.status_code}"
                logger.error(f"{error_text} - Response: {response.text}")
                return {
                    "error": error_text,
                    "message": response.text,
                    "status_code": response.status_code
                }
                
    except httpx.RequestError as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": "Request error", "message": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": "Unexpected error", "message": str(e)}

def format_web_results(data: Dict[str, Any]) -> str:
    """
    Format web search results into readable text.
    
    Parameters:
    -----------
    data : Dict[str, Any]
        The JSON response from the Brave API
        
    Returns:
    --------
    str
        Formatted search results as text
    """
    # Check for errors first
    if "error" in data:
        return f"Error fetching search results: {data['error']}\nDetails: {data.get('message', 'No details provided')}"
        
    # Check if we have results
    if not data.get("web", {}).get("results"):
        return "No results found."

    # Format the results
    results = []
    for index, result in enumerate(data["web"]["results"], 1):
        results.append(
            f"Result {index}:\nTitle: {result.get('title', '')}\n"
            f"Description: {result.get('description', '')}\n"
            f"URL: {result.get('url', '')}"
        )

    return "\n\n".join(results)

def register_brave_search_tools(mcp: FastMCP) -> None:
    """
    Register all Brave Search tools with the MCP server.
    
    Parameters:
    -----------
    mcp : FastMCP
        The MCP server instance
    """
    
    @mcp.tool()
    async def brave_web_search(query: str, count: int = 10, offset: int = 0) -> str:
        """
        Performs a web search using the Brave Search API.
        
        Parameters:
        -----------
        query : str
            Search query (max 400 chars, 50 words)
        count : int
            Number of results (1-20, default 10)
        offset : int
            Pagination offset (max 9, default 0)
            
        Returns:
        --------
        str
            Formatted search results
        """
        # Get API key from environment with debug logging
        api_key = os.environ.get("BRAVE_API_KEY")
        logger.debug(f"BRAVE_API_KEY is {'set' if api_key else 'NOT set'}")
        
        if not api_key:
            return "Error: BRAVE_API_KEY environment variable is not set"
            
        # Ensure count and offset are within bounds
        count = min(max(1, count), 20)
        offset = min(max(0, offset), 9)

        # Make the request
        data = await make_brave_request(query, api_key, count, offset)
        
        # Format and return the results
        return format_web_results(data)
    
    @mcp.tool()
    async def brave_local_search(query: str, location: str, count: int = 10) -> str:
        """
        Search for local businesses and services using Brave Search.
        
        Parameters:
        -----------
        query : str
            The search query (e.g., "restaurants", "plumbers")
        location : str
            Location for the search (e.g., "New York, NY")
        count : int
            Number of results (1-20, default 10)
            
        Returns:
        --------
        str
            Formatted local search results
        """
        # Get API key from environment
        api_key = os.environ.get("BRAVE_API_KEY")
        logger.debug(f"BRAVE_API_KEY is {'set' if api_key else 'NOT set'}")
        
        if not api_key:
            return "Error: BRAVE_API_KEY environment variable is not set"
            
        # Ensure count is within bounds
        count = min(max(1, count), 20)
        
        # Combine query and location
        combined_query = f"{query} in {location}"
        
        # Make the request
        data = await make_brave_request(combined_query, api_key, count, 0)
        
        # Format and return the results
        return format_web_results(data)