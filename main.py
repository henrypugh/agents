# main.py
import asyncio
import sys
import os
import argparse

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
    """Main entry point for the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCP Client with dynamic server connection")
    parser.add_argument("server_scripts", nargs="*", help="Optional server script paths to connect to at startup")
    parser.add_argument("--server", action="append", dest="configured_servers", 
                        help="Named servers from config to connect to at startup (can be used multiple times)")
    args = parser.parse_args()
    
    # Get LLM model from environment or use default
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.0-flash-001")
    
    # Initialize client
    client = MCPClient(model=model)
    
    try:
        # Pre-connect to requested servers if any
        pre_connected = []
        
        # Connect to script-based servers
        for script in args.server_scripts or []:
            logger.info(f"Pre-connecting to server script: {script}")
            server_name = await client.connect_to_server(script)
            pre_connected.append(server_name)
        
        # Connect to configured servers
        if args.configured_servers:
            for server_name in args.configured_servers:
                logger.info(f"Pre-connecting to configured server: {server_name}")
                result = await client.connect_to_configured_server(server_name)
                if result["status"] in ["connected", "already_connected"]:
                    pre_connected.append(server_name)
        
        # Show pre-connection status if any servers were connected
        if pre_connected:
            print(f"Pre-connected to servers: {', '.join(pre_connected)}")
        
        # Start chat loop - the LLM can still connect to additional servers as needed
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())