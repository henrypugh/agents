"""
Runner script for the SimpleAgent.

This script initializes and runs the SimpleAgent for code analysis tasks.
"""

import asyncio
import logging
import argparse
import sys
import os
import time
import uuid
import hashlib

# Add the project root directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dotenv import load_dotenv
from src.utils.logger_setup import setup_logging
from src.client.mcp_client import MCPClient
from src.client.tool_processor import ToolProcessor
from simple_agent import SimpleAgent

# Setup tracing
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task, agent

# Load environment variables
load_dotenv()

# Initialize Traceloop
Traceloop.init(
  disable_batch=True, 
  api_key=os.getenv("TRACEL_API_KEY")
)

# Setup logging
logger = setup_logging()

@workflow(name="agent_runner_workflow")
async def main():
    """Main entry point for the SimpleAgent runner"""
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Set global association properties for this session
    Traceloop.set_association_properties({
        "session_id": session_id,
        "session_type": "agent_runner",
        "start_time": start_time
    })
    
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
    
    # Track command line arguments
    Traceloop.set_association_properties({
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "default_task": args.task == parser.get_default("task")
    })
    
    # Get the task from arguments
    task = args.task
    
    # Generate a task ID
    task_id = hashlib.md5(task.encode()).hexdigest()[:12]
    
    # Track task information
    Traceloop.set_association_properties({
        "task_id": task_id,
        "task_length": len(task),
        "task_preview": task[:50] + "..." if len(task) > 50 else task
    })
    
    # Initialize MCP client
    client = MCPClient(model=args.model)
    
    try:
        # Start main server (which has filesystem tools)
        logger.info("Connecting to main server...")
        
        # Track server connection attempt
        Traceloop.set_association_properties({
            "connection_stage": "connecting",
            "server_name": "main-server"
        })
        
        result = await client.connect_to_configured_server("main-server")
        
        if not result or result.get("status") not in ["connected", "already_connected"]:
            logger.error("Failed to connect to main server")
            
            # Track connection failure
            Traceloop.set_association_properties({
                "connection_stage": "failed",
                "connection_error": "server_connection_failed"
            })
            
            return
            
        # Track successful connection
        Traceloop.set_association_properties({
            "connection_stage": "connected",
            "tool_count": len(result.get("tools", []))
        })
        
        logger.info("Successfully connected to main server")
        
        # Create agent configuration
        agent_config = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        }
        
        # Track agent configuration
        Traceloop.set_association_properties({
            "agent_config": agent_config
        })
        
        # Create tool processor
        tool_processor = ToolProcessor(client.server_manager)
        
        # Create SimpleAgent
        logger.info("Initializing SimpleAgent...")
        
        # Track agent initialization
        Traceloop.set_association_properties({
            "initialization_stage": "creating_agent"
        })
        
        agent = SimpleAgent(
            client.llm_client,
            client.server_manager,
            tool_processor,
            agent_config
        )
        
        # Track agent creation
        Traceloop.set_association_properties({
            "initialization_stage": "agent_created"
        })
        
        # Print task banner
        print("\n" + "=" * 80)
        print(f"TASK: {task}")
        print("=" * 80 + "\n")
        
        # Execute task
        logger.info("Starting task execution...")
        print("Working on task, please wait...\n")
        
        # Track task execution start
        Traceloop.set_association_properties({
            "execution_stage": "started",
            "execution_start_time": time.time()
        })
        
        result = await agent.execute_task(task)
        
        # Track task execution completion
        Traceloop.set_association_properties({
            "execution_stage": "completed",
            "execution_end_time": time.time(),
            "execution_duration": time.time() - start_time,
            "result_length": len(result)
        })
        
        # Print results
        print("\n" + "=" * 80)
        print("AGENT RECOMMENDATIONS")
        print("=" * 80 + "\n")
        print(result)
        
        # Record successful execution for annotations
        Traceloop.user_feedback.create(
            "agent_task_feedback",
            task_id,
            {"execution_status": "success"}
        )
        
    except Exception as e:
        logger.error(f"Error running SimpleAgent: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        
        # Track error
        Traceloop.set_association_properties({
            "execution_stage": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "execution_end_time": time.time(),
            "execution_duration": time.time() - start_time
        })
        
        # Record error for annotations
        Traceloop.user_feedback.create(
            "agent_task_feedback",
            task_id,
            {"execution_status": "error", "error_type": type(e).__name__}
        )
        
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        
        # Track cleanup
        Traceloop.set_association_properties({
            "cleanup_stage": "started"
        })
        
        await client.cleanup()
        
        # Track session end
        session_duration = time.time() - start_time
        Traceloop.set_association_properties({
            "cleanup_stage": "completed",
            "session_duration": session_duration,
            "session_end_time": time.time()
        })
        
        logger.info(f"Agent runner completed in {session_duration:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())