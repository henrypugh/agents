"""
Tool Processor module for handling tool execution and processing.

This module provides Pydantic-validated tool execution and processing,
with structured input and output handling.
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable, Awaitable, Protocol, Type, TypeVar, Union
import uuid

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.schemas import (
    ToolCall, 
    ToolResult, 
    ToolResultStatus,
    Message
)
from src.utils.decorators import tool_execution

logger = logging.getLogger(__name__)

class ResponseProcessorProtocol(Protocol):
    """Protocol for response processor follow-up method"""
    async def get_follow_up_response(
        self,
        messages: List[Message],
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        ...

class ToolExecutor:
    """
    Processes and executes tool calls from LLMs with Pydantic validation.
    
    This class handles:
    - Finding the appropriate server for each tool
    - Executing tools on servers
    - Processing results with proper validation
    - Managing error handling and result extraction
    """
    
    def __init__(self, server_manager):
        """
        Initialize the tool processor
        
        Args:
            server_manager: Server manager instance
        """
        self.server_manager = server_manager
        
        # Generate a unique ID for this processor instance
        processor_id = str(uuid.uuid4())[:8]
        
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
        server_name: str
    ) -> Any:
        """
        Execute a tool on a server
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            server_name: Name of the server to execute on
            
        Returns:
            Result of tool execution
            
        Raises:
            ValueError: If server not found
        """
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")
        
        logger.info(f"Executing tool '{tool_name}' on server '{server_name}'")
        return await server.execute_tool(tool_name, tool_args)
    
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
        
        if not server_name or server_name not in self.server_manager.servers:
            error_msg = f"Error: Can't determine which server handles tool '{tool_call.tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            return ToolResult.error(tool_call.id, error_msg)
        
        final_text.append(f"[Calling tool {tool_call.tool_name} from {server_name} server]")
        
        try:
            # Execution timing
            start_time = time.time()
            
            # Execute the tool
            result = await self.execute_tool(tool_call.tool_name, tool_call.arguments, server_name)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Process result into text
            result_text = self.extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await get_follow_up_response(messages, tools, final_text)
            
            # Return successful result
            return ToolResult.success(tool_call.id, result_text, execution_time)
            
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            return ToolResult.error(tool_call.id, error_msg)
            
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            return ToolResult.error(tool_call.id, error_msg)
    
    def _update_message_history(
        self,
        messages: List[Message],
        tool_call: ToolCall,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result
        
        Args:
            messages: Conversation history as Message models
            tool_call: Tool call model
            result_text: Result of tool execution
        """
        try:
            # Create assistant message with tool call
            assistant_message = Message.assistant(tool_calls=[tool_call])
            
            # Create tool response message
            tool_message = Message.tool(tool_call_id=tool_call.id, content=result_text)
            
            # Add messages to history
            messages.append(assistant_message)
            messages.append(tool_message)
            
        except Exception as e:
            logger.error(f"Error updating message history: {e}")
            
            # Fall back to dictionary-based message creation
            try:
                # Create messages as dictionaries
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.arguments)
                            }
                        }
                    ]
                }
                
                tool_response_message = {
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": result_text
                }
                
                messages.append(tool_call_message)
                messages.append(tool_response_message)
                
            except Exception as e2:
                logger.error(f"Error in fallback message history update: {e2}")