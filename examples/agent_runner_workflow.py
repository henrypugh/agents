"""
Agent Runner with Workflow System.

This script demonstrates how to use the new event-driven workflow system
to run a code analysis task. It replaces the SimpleAgent approach with
the new workflow-based approach.
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
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from dotenv import load_dotenv
from src.utils.logger_setup import setup_logging
from src.client.agent import Agent
from src.utils.decorators import workflow, server_connection
from src.utils.schemas import ConnectResponseStatus

# Import the workflow components
from src.utils.workflow import (
    BaseEvent,
    StartEvent,
    StopEvent,
    ErrorEvent,
    WorkflowContext,
    draw_all_possible_flows,
    draw_most_recent_execution
)

# Import our code analysis workflow
from examples.workflows.code_analysis_workflow import CodeAnalysisWorkflow

# Setup tracing
from traceloop.sdk import Traceloop

# Load environment variables
load_dotenv()

# Initialize Traceloop
Traceloop.init(
  disable_batch=True, 
  api_key=os.getenv("TRACELOOP_API_KEY")
)

# Setup logging
logger = setup_logging()

@workflow()
async def main():
    """Main entry point for the workflow-based agent runner"""
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Set global association properties for this session
    Traceloop.set_association_properties({
        "session_id": session_id,
        "session_type": "agent_runner_workflow",
        "start_time": start_time
    })
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CodeAnalysisWorkflow for code analysis tasks")
    
    parser.add_argument(
        "task",
        nargs="?",
        default="Review my codebase structure and suggest improvements for organization and modularity",
        help="Task for the workflow to perform"
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
    
    # Initialize Agent
    client = Agent(model=args.model)
    
    try:
        # Start main server (which has filesystem tools)
        logger.info("Connecting to main server...")
        
        # Track server connection attempt
        Traceloop.set_association_properties({
            "connection_stage": "connecting",
            "server_name": "main-server"
        })
        
        await connect_to_main_server(client)
        
        # Create workflow configuration
        workflow_config = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens
        }
        
        # Track workflow configuration
        Traceloop.set_association_properties({
            "workflow_config": workflow_config
        })
        
        # Register the workflow
        client.register_workflow("code_analysis", CodeAnalysisWorkflow)
        
        # Generate workflow visualization
        visualization_path = "code_analysis_workflow.html"
        draw_all_possible_flows(CodeAnalysisWorkflow, filename=visualization_path)
        print(f"Workflow visualization saved to {visualization_path}")
        
        # Print task banner
        print("\n" + "=" * 80)
        print(f"TASK: {task}")
        print("=" * 80 + "\n")
        
        # Execute workflow
        logger.info("Starting workflow execution...")
        print("Working on task, please wait...\n")
        
        # Track workflow execution start
        Traceloop.set_association_properties({
            "execution_stage": "started",
            "execution_start_time": time.time()
        })
        
        # Get a workflow handler to stream events
        workflow_handler = client.workflows["code_analysis"](client).run(task=task)
        
        # Stream events while the workflow runs
        progress_count = 0
        print("Progress updates:")
        async for event in workflow_handler.stream_events():
            if isinstance(event, StopEvent):
                print("\nWorkflow completed with result")
            elif isinstance(event, ErrorEvent):
                print(f"\nError: {event.error_message}")
            else:
                # For other events, show basic progress info
                progress_count += 1
                event_type = event.__class__.__name__
                
                # Determine a suitable message based on event type
                message = f"Step {progress_count}: "
                if event_type == "ExecuteStepEvent":
                    message += f"Executing '{event.description}'"
                elif event_type == "StepResultEvent":
                    message += f"Completed step (status: {event.status})"
                elif event_type == "PlanningEvent":
                    message += "Planning analysis approach"
                elif event_type == "GenerateRecommendationsEvent":
                    message += "Generating recommendations"
                else:
                    message += f"Processing {event_type}"
                    
                print(f"{message}")
        
        # Get the final result
        result = await workflow_handler
        
        # Generate execution visualization
        workflow_instance = client.workflows["code_analysis"](client)
        execution_viz_path = "code_analysis_execution.html"
        draw_most_recent_execution(workflow_instance, filename=execution_viz_path)
        print(f"\nExecution visualization saved to {execution_viz_path}")
        
        # Track workflow execution completion
        Traceloop.set_association_properties({
            "execution_stage": "completed",
            "execution_end_time": time.time(),
            "execution_duration": time.time() - start_time,
            "result_length": len(result) if result else 0
        })
        
        # Print results
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80 + "\n")
        print(result)
        
    except Exception as e:
        logger.error(f"Error running workflow: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        
        # Track error
        Traceloop.set_association_properties({
            "execution_stage": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "execution_end_time": time.time(),
            "execution_duration": time.time() - start_time
        })
        
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

@server_connection()
async def connect_to_main_server(client):
    """Connect to the main server"""
    try:
        result = await client.connect_to_configured_server("main-server")
        
        # Check response using Pydantic model attributes
        if not hasattr(result, 'status') or result.status not in [ConnectResponseStatus.CONNECTED, ConnectResponseStatus.ALREADY_CONNECTED]:
            logger.error("Failed to connect to main server")
            
            # Track connection failure
            Traceloop.set_association_properties({
                "connection_stage": "failed",
                "connection_error": "server_connection_failed"
            })
            
            raise RuntimeError("Failed to connect to main server")
            
        # Track successful connection
        Traceloop.set_association_properties({
            "connection_stage": "connected",
            "tool_count": getattr(result, 'tool_count', 0) or len(getattr(result, 'tools', []) or [])
        })
        
        logger.info("Successfully connected to main server")
        return result
        
    except Exception as e:
        logger.error(f"Error connecting to main server: {str(e)}")
        
        # Track connection error
        Traceloop.set_association_properties({
            "connection_stage": "error",
            "error_type": type(e).__name__,
            "error_message": str(e)
        })
        
        raise


if __name__ == "__main__":
    asyncio.run(main())