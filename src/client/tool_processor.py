"""
Tool Processor module for handling tool execution and processing.
"""

import logging
import json
from typing import Dict, List, Optional, Any, Callable
import uuid

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.schemas import ToolCall, ToolResult
from src.utils.decorators import tool_execution

logger = logging.getLogger("ToolExecutor")

class ToolExecutor:
    """Processes and executes tool calls from LLMs"""
    
    def __init__(self, server_manager):
        """Initialize the tool processor"""
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
        """Find which server handles the specified tool"""
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
        """Execute a tool on a server"""
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")
        
        logger.info(f"Executing tool '{tool_name}' on server '{server_name}'")
        return await server.execute_tool(tool_name, tool_args)
    
    @task(name="extract_result_text")
    def extract_result_text(self, result: Any) -> str:
        """Extract text content from a tool result"""
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
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        get_follow_up_response: Callable
    ) -> None:
        """Process an external tool call"""
        # Get tool details
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args_raw = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        # Find server for this tool
        server_name = self.find_server_for_tool(tool_name, tools)
        
        if not server_name or server_name not in self.server_manager.servers:
            error_msg = f"Error: Can't determine which server handles tool '{tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            return
        
        final_text.append(f"[Calling tool {tool_name} from {server_name} server]")
        
        try:
            # Parse arguments
            tool_args = json.loads(tool_args_raw) if isinstance(tool_args_raw, str) else tool_args_raw
            
            # Execute the tool
            result = await self.execute_tool(tool_name, tool_args, server_name)
            
            # Process result into text
            result_text = self.extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await get_follow_up_response(messages, tools, final_text)
            
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
    
    async def process_tool_call_as_model(
        self,
        tool_call: ToolCall,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        get_follow_up_response: Callable
    ) -> Optional[ToolResult]:
        """Process a tool call with Pydantic models"""
        try:
            # Find server for this tool
            server_name = self.find_server_for_tool(tool_call.tool_name, tools)
            
            if not server_name or server_name not in self.server_manager.servers:
                return ToolResult(
                    tool_id=tool_call.id,
                    status="error",
                    error=f"Can't determine which server handles tool '{tool_call.tool_name}'"
                )
            
            final_text.append(f"[Calling tool {tool_call.tool_name} from {server_name} server]")
            
            # Execute the tool
            import time
            start_time = time.time()
            result = await self.execute_tool(tool_call.tool_name, tool_call.arguments, server_name)
            execution_time = time.time() - start_time
            
            # Process result into text
            result_text = self.extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await get_follow_up_response(messages, tools, final_text)
            
            return ToolResult(
                tool_id=tool_call.id,
                status="success",
                result=result_text,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error executing tool: {e}")
            final_text.append(f"Error executing tool: {str(e)}")
            
            return ToolResult(
                tool_id=tool_call.id,
                status="error",
                error=str(e)
            )
    
    def _update_message_history(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """Update message history with tool call and result"""
        # Extract tool information
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Handle ToolCall model or other formats
            tool_name = getattr(tool_call, 'tool_name', getattr(tool_call, 'name', 'unknown'))
            tool_args = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        # Add assistant message with tool call
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
                    }
                }
            ]
        }
        messages.append(tool_call_message)
        
        # Add tool response message
        tool_response_message = {
            "role": "tool", 
            "tool_call_id": tool_id,
            "content": result_text
        }
        messages.append(tool_response_message)