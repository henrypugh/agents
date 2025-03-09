# main.py
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
    model = os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.0-flash-001")
    
    client = MCPClient(model=model)
    try:
        # Parse arguments to collect all servers (both script paths and configured servers)
        server_scripts = []
        configured_servers = []
        
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--server":
                if i + 1 < len(sys.argv):
                    configured_servers.append(sys.argv[i + 1])
                    i += 2
                else:
                    print("Error: --server requires a server name")
                    sys.exit(1)
            else:
                server_scripts.append(sys.argv[i])
                i += 1
        
        if not server_scripts and not configured_servers:
            print("Usage:")
            print("  python main.py <server_script_path> [--server <server_name>]")
            print("  Examples:")
            print("    python main.py server/main.py                     # Connect to local server only")
            print("    python main.py --server brave-search              # Connect to configured server only") 
            print("    python main.py server/main.py --server brave-search # Connect to both")
            sys.exit(1)
        
        # Connect to all script-based servers
        for script in server_scripts:
            logger.info(f"Connecting to server script: {script}")
            await client.connect_to_server(script)
        
        # Connect to all configured servers
        for server_name in configured_servers:
            logger.info(f"Connecting to configured server: {server_name}")
            await client.connect_to_configured_server(server_name)
        
        # Start chat loop
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())