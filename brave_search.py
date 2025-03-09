# brave_search.py
import asyncio
import os
import sys

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.utils.logger_setup import setup_logging
from src.client.mcp_client import MCPClient

# Setup logging
logger = setup_logging()

# Load environment variables
load_dotenv()

async def main() -> None:
    """Connect to Brave Search server and start chat loop"""
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-flash-1.5-8b")
    logger.info(f"Starting MCP client with Brave Search using model: {model}")
    
    client = MCPClient(model=model)
    try:
        await client.connect_to_configured_server("brave-search")
        
        # Display available tools information to the user
        print("\nBrave Search MCP Server Connected!")
        print("Available search tools:")
        print("  brave_web_search - Execute web searches with pagination and filtering")
        print("  brave_local_search - Search for local businesses and services")
        
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())