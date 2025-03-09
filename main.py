import asyncio
import sys
import os

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.utils.logger_setup import setup_logging
from src.client.mcp_client import MCPClient

# Setup logging
logger = setup_logging()

# Load environment variables
load_dotenv()  # Load environment variables from .env

async def main() -> None:
    """Main entry point for the application"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <server_script_path>")
        sys.exit(1)
        
    server_script = sys.argv[1]
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-flash-1.5-8b")
    
    logger.info(f"Starting MCP client with model: {model}")
    logger.info(f"Using server script: {server_script}")
    
    client = MCPClient(model=model)
    try:
        await client.connect_to_server(server_script)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())