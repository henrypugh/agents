"""
Runner script for the SimpleAgent.

This script initializes and runs the SimpleAgent for code analysis tasks.
"""



import asyncio
import logging
import argparse
import sys
import os

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.utils.logger_setup import setup_logging
from src.client.mcp_client import MCPClient
from src.client.tool_processor import ToolProcessor
from simple_agent import SimpleAgent

# Load environment variables
load_dotenv()

from traceloop.sdk import Traceloop

Traceloop.init(
  disable_batch=True, 
  api_key=os.getenv("TRACEL_API_KEY")
)

# Setup logging
logger = setup_logging()

async def main():
    """Main entry point for the SimpleAgent runner"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run SimpleAgent for code analysis tasks")
    
    parser.add_argument(
        "task",
        nargs="?",
        default="Review my codebase structure and suggest improvements for organization and modularity",
        help="Task for the agent to perform"
    )
    
    parser.add_argument(
        "--model",
        default=os.getenv("DEFAULT_LLM_MODEL", "google/gemini-2.0-flash-001"),
        help="LLM model to use"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM generation"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for LLM responses"
    )
    
    args = parser.parse_args()
    
    # Get the task from arguments
    task = args.task
    
    # Initialize MCP client
    client = MCPClient(model=args.model)
    
    try:
        # Start main server (which has filesystem tools)
        logger.info("Connecting to main server...")
        result = await client.connect_to_configured_server("main-server")
        
        if not result or result.get("status") not in ["connected", "already_connected"]:
            logger.error("Failed to connect to main server")
            return
            
        logger.info("Successfully connected to main server")
        
        # Create agent configuration
        agent_config = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        }
        
        # Create tool processor
        tool_processor = ToolProcessor(client.server_manager)
        
        # Create SimpleAgent
        agent = SimpleAgent(
            client.llm_client,
            client.server_manager,
            tool_processor,
            agent_config
        )
        
        # Print task banner
        print("\n" + "=" * 80)
        print(f"TASK: {task}")
        print("=" * 80 + "\n")
        
        # Execute task
        print("Working on task, please wait...\n")
        result = await agent.execute_task(task)
        
        # Print results
        print("\n" + "=" * 80)
        print("AGENT RECOMMENDATIONS")
        print("=" * 80 + "\n")
        print(result)
        
    except Exception as e:
        logger.error(f"Error running SimpleAgent: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
    finally:
        # Clean up resources
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())