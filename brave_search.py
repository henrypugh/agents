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
    """Start Brave Search client and begin chat session"""
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.0-flash-001")
    logger.info(f"Starting MCP client with Brave Search using model: {model}")
    
    client = MCPClient(model=model)
    try:
        # Connect to the main server, which now includes the Brave Search tools
        await client.connect_to_server("server/main.py")
        
        # Display available search tools information to the user
        print("\nBrave Search MCP Tools Available!")
        print("Available search tools:")
        print("  brave_web_search - Execute web searches with pagination")
        print("  brave_local_search - Search for local businesses in a specific location")
        
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    # Make sure BRAVE_API_KEY is set
    if not os.getenv("BRAVE_API_KEY"):
        print("Error: BRAVE_API_KEY environment variable is required")
        print("Please set it in your .env file or export it in your shell")
        sys.exit(1)
        
    asyncio.run(main())