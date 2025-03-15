"""
Tool Processor module for handling tool execution and processing.

This module provides Pydantic-validated tool execution and processing,
with structured input and output handling.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, Protocol, Type, TypeVar, Union, cast
import uuid
import asyncio

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop
from pydantic import TypeAdapter

from src.utils.schemas import (
    ToolCall, 
    ToolResult, 
    ToolResultStatus,
    Message,
    MessageHistory
)
from src.utils.decorators import tool_execution

logger = logging.getLogger(__name__)

# Default timeout for tool execution
DEFAULT_TOOL_TIMEOUT = 30.0

class ResponseProcessorProtocol(Protocol):
    """Protocol for response processor follow-up method"""
    async def get_follow_up_response(
        self,
        messages: List[Message],
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        ...

class ServerRegistryProtocol(Protocol):
    """Protocol for server registry"""
    def get_server(self, server_name: str) -> Any:
        ...

class ToolExecutionConfig:
    """Configuration for tool execution"""
    def __init__(
        self,
        timeout: float = DEFAULT_TOOL_TIMEOUT,
        trace_arguments: bool = True,
        validate_results: bool = True
    ):
        self.timeout = timeout
        self.trace_arguments = trace_arguments
        self.validate_results = validate_results

class ToolExecutor:
    """
    Processes and executes tool calls from LLMs with Pydantic validation.
    
    This class handles:
    - Finding the appropriate server for each tool
    - Executing tools on servers
    - Processing results with proper validation
    - Managing error handling and result extraction
    """
    
    def __init__(
        self, 
        server_manager: ServerRegistryProtocol,
        config: Optional[ToolExecutionConfig] = None
    ):
        """
        Initialize the tool processor
        
        Args:
            server_manager: Server manager instance
            config: Optional configuration for tool execution
        """
        self.server_manager = server_manager
        self.config = config or ToolExecutionConfig()
        
        # Generate a unique ID for this processor instance
        processor_id = str(uuid.uuid4())[:8]
        
        # Create TypeAdapter for validating tool results
        self.tool_result_validator = TypeAdapter(ToolResult)
        
        # Set global association properties for this processor instance
        Traceloop.set_association_properties({
            "processor_id": processor_id
        })
        
        logger.info("ToolExecutor initialized")
    
    @task(name="find_server_for_tool")
    def find_server_for_tool(self, tool_name: str, tools: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find which server handles the specified tool
        
        Args:
            tool_name: Name of the tool to find
            tools: Available tools in OpenAI format
            
        Returns:
            Server name or None if not found
        """
        Traceloop.set_association_properties({
            "tool_name": tool_name,
            "available_tools_count": len(tools)
        })
        
        # Look for server in tool metadata
        for tool in tools:
            if tool["function"]["name"] == tool_name:
                if "metadata" in tool["function"]:
                    # Check if it's an internal tool
                    if tool["function"]["metadata"].get("internal"):
                        return "internal"
                    
                    # Otherwise check for server metadata
                    if "server" in tool["function"]["metadata"]:
                        return tool["function"]["metadata"]["server"]
        
        return None
    
    @tool_execution()
    async def execute_tool(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        server_name: str,
        timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a tool on a server with timeout
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            server_name: Name of the server to execute on
            timeout: Optional custom timeout in seconds
            
        Returns:
            Result of tool execution
            
        Raises:
            ValueError: If server not found
            TimeoutError: If execution times out
            Exception: Other execution errors
        """
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")
        
        # Use provided timeout or default from config
        execution_timeout = timeout or self.config.timeout
        
        # Trace arguments if enabled
        if self.config.trace_arguments:
            # Truncate large arguments for tracing
            traced_args = {}
            for key, value in tool_args.items():
                if isinstance(value, str) and len(value) > 100:
                    traced_args[key] = f"{value[:100]}... (truncated, {len(value)} chars)"
                else:
                    traced_args[key] = value
            
            Traceloop.set_association_properties({
                "tool_arguments": traced_args
            })
        
        # Execute with timeout
        logger.info(f"Executing tool '{tool_name}' on server '{server_name}' with timeout {execution_timeout}s")
        start_time = time.time()
        
        try:
            return await asyncio.wait_for(
                server.execute_tool(tool_name, tool_args),
                timeout=execution_timeout
            )
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.warning(f"Tool execution timed out after {execution_time:.2f}s: {tool_name}")
            
            # Create timeout result and trace it
            Traceloop.set_association_properties({
                "execution_status": "timeout",
                "execution_time": execution_time
            })
            
            raise TimeoutError(f"Tool execution timed out after {execution_time:.2f}s")
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing tool '{tool_name}': {e}")
            
            # Trace error details
            Traceloop.set_association_properties({
                "execution_status": "error",
                "execution_time": execution_time,
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise
    
    @task(name="extract_result_text")
    def extract_result_text(self, result: Any) -> str:
        """
        Extract text content from a tool result
        
        Args:
            result: Raw tool result
            
        Returns:
            Extracted text
        """
        Traceloop.set_association_properties({
            "result_type": type(result).__name__
        })
        
        try:
            # Handle various result formats
            if not hasattr(result, "content"):
                return str(result)
                
            if isinstance(result.content, list):
                result_text = ""
                for item in result.content:
                    if hasattr(item, "text"):
                        result_text += item.text
                    else:
                        result_text += str(item)
                return result_text
                
            elif isinstance(result.content, str):
                return result.content
                
            else:
                return str(result.content)
                
        except Exception as e:
            logger.error(f"Error extracting result text: {e}")
            return f"Error extracting result: {str(e)}"
    
    @tool_execution()
    async def process_external_tool(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        get_follow_up_response: Callable[[List[Message], List[Dict[str, Any]], List[str]], Awaitable[None]]
    ) -> Optional[ToolResult]:
        """
        Process an external tool call with Pydantic validation
        
        Args:
            tool_call: Validated tool call model
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
            get_follow_up_response: Function to get follow-up response
            
        Returns:
            Tool result model or None on error
        """
        # Find server for this tool
        server_name = self.find_server_for_tool(tool_call.tool_name, tools)
        
        if not server_name or server_name not in ["internal"] and not hasattr(self.server_manager, "servers") or (
            hasattr(self.server_manager, "servers") and server_name not in self.server_manager.servers):
            error_msg = f"Error: Can't determine which server handles tool '{tool_call.tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            return ToolResult.error(tool_call.id, error_msg)
        
        # Notify the user about tool execution
        final_text.append(f"[Calling tool {tool_call.tool_name} from {server_name} server]")
        
        try:
            # Track execution metrics
            start_time = time.time()
            
            # Execute the tool with error handling and timeout
            try:
                result = await self.execute_tool(tool_call.tool_name, tool_call.arguments, server_name)
                execution_time = time.time() - start_time
                
                # Process result into text
                result_text = self.extract_result_text(result)
                if result_text:
                    # Truncate very long results for display
                    if len(result_text) > 2000:
                        display_text = f"{result_text[:2000]}... (truncated, {len(result_text)} characters total)"
                        final_text.append(f"Tool result: {display_text}")
                    else:
                        final_text.append(f"Tool result: {result_text}")
                
                # Track successful execution
                Traceloop.set_association_properties({
                    "execution_status": "success",
                    "execution_time": execution_time,
                    "result_length": len(result_text) if result_text else 0
                })
                
                # Update message history using the utility class
                MessageHistory.add_tool_interaction(messages, tool_call, result_text)
                
                # Get follow-up response
                await get_follow_up_response(messages, tools, final_text)
                
                # Return successful result
                return ToolResult.success(tool_call.id, result_text, execution_time)
                
            except TimeoutError as te:
                execution_time = time.time() - start_time
                error_msg = f"Tool execution timed out after {execution_time:.2f} seconds"
                logger.error(error_msg)
                final_text.append(error_msg)
                
                # Update message history with error
                error_result = f"Error: {error_msg}"
                MessageHistory.add_tool_interaction(messages, tool_call, error_result)
                
                # Get follow-up response to handle the timeout
                await get_follow_up_response(messages, tools, final_text)
                
                return ToolResult.timeout(tool_call.id, execution_time)
                
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            return ToolResult.error(tool_call.id, error_msg)
            
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            
            # Update message history with error
            error_result = f"Error: {error_msg}"
            MessageHistory.add_tool_interaction(messages, tool_call, error_result)
            
            # Get follow-up response to handle the error
            await get_follow_up_response(messages, tools, final_text)
            
            return ToolResult.error(tool_call.id, error_msg)