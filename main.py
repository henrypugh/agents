# In main.py
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
        print("Usage:")
        print("  python main.py <server_script_path>            # Connect to local server script")
        print("  python main.py --server <server_name>          # Connect to configured server")
        sys.exit(1)
        
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.0-flash-001")
    
    client = MCPClient(model=model)
    try:
        if sys.argv[1] == "--server":
            if len(sys.argv) < 3:
                print("Error: Missing server name")
                print("Usage: python main.py --server <server_name>")
                sys.exit(1)
            
            server_name = sys.argv[2]
            logger.info(f"Connecting to configured server: {server_name}")
            await client.connect_to_configured_server(server_name)
        else:
            server_script = sys.argv[1]
            logger.info(f"Using server script: {server_script}")
            await client.connect_to_server(server_script)
        
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())