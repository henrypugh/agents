"""
Tool Processor module for handling tool execution and processing.
"""

import logging
import json
from typing import Dict, List, Optional, Any
import hashlib
import uuid

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.decorators import tool_execution

logger = logging.getLogger("ToolExecutor")

class ToolExecutor:
    """Processes and executes tool calls from LLMs"""
    
    def __init__(self, server_manager):
        """
        Initialize the tool processor
        
        Args:
            server_manager: Server manager instance
        """
        self.server_manager = server_manager
        
        # Generate a unique ID for this processor instance
        processor_id = str(uuid.uuid4())
        
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
            tool_name: Name of the tool
            tools: List of available tools
            
        Returns:
            Name of the server or None if not found
        """
        # Set association properties for this lookup
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
                        # Track found internal tool
                        Traceloop.set_association_properties({
                            "server_found": True,
                            "server_name": "internal",
                            "is_internal": True
                        })
                        return "internal"
                    
                    # Otherwise check for server metadata
                    if "server" in tool["function"]["metadata"]:
                        server_name = tool["function"]["metadata"]["server"]
                        # Track found server
                        Traceloop.set_association_properties({
                            "server_found": True,
                            "server_name": server_name,
                            "is_internal": False
                        })
                        return server_name
        
        # Track no server found
        Traceloop.set_association_properties({
            "server_found": False
        })
        
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
            tool_name: Name of the tool
            tool_args: Tool arguments
            server_name: Name of the server
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If the server is not found
        """
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")
        
        logger.info(f"Executing tool '{tool_name}' on server '{server_name}'")
        
        try:
            # Execute the tool
            result = await server.execute_tool(tool_name, tool_args)
            return result
            
        except Exception as e:
            logger.error(f"Error executing tool '{tool_name}' on server '{server_name}': {str(e)}", exc_info=True)
            raise
    
    @task(name="process_external_tool")
    async def process_external_tool(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        get_follow_up_response
    ) -> None:
        """
        Process an external tool call
        
        Args:
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
            get_follow_up_response: Function to get follow-up response
        """
        # Get tool details from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args_raw = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        # Find server for this tool
        server_name = self.find_server_for_tool(tool_name, tools)
        
        if not server_name or server_name not in self.server_manager.servers:
            error_msg = f"Error: Can't determine which server handles tool '{tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "server_not_found",
                "server_name": server_name or "unknown"
            })
            return
        
        # Track the server that will handle this tool
        Traceloop.set_association_properties({
            "server_name": server_name
        })
        
        # Execute the tool
        await self._execute_and_process_tool(
            server_name, 
            tool_call, 
            messages, 
            tools, 
            final_text,
            get_follow_up_response
        )
    
    @tool_execution()
    async def _execute_and_process_tool(
        self,
        server_name: str,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        get_follow_up_response
    ) -> None:
        """
        Execute a tool and process the result
        
        Args:
            server_name: Name of the server
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
            get_follow_up_response: Function to get follow-up response
        """
        # Get tool details from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args_raw = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        final_text.append(f"[Calling tool {tool_name} from {server_name} server with args {tool_args_raw}]")
        
        try:
            # Parse arguments
            tool_args = json.loads(tool_args_raw)
            
            # Execute the tool
            result = await self.execute_tool(tool_name, tool_args, server_name)
            
            # Process result into text
            result_text = self.extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history - we need to adapt this for the new API
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await get_follow_up_response(messages, tools, final_text)
            
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "json_decode",
                "error_message": str(e)
            })
            
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg, exc_info=True)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "execution_failed",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
    
    @task(name="extract_result_text")
    def extract_result_text(self, result: Any) -> str:
        """
        Extract text content from a tool result
        
        Args:
            result: Result from tool execution
            
        Returns:
            Extracted text content
        """
        # Track the extraction process
        Traceloop.set_association_properties({
            "result_type": type(result).__name__
        })
        
        try:
            if not hasattr(result, "content"):
                result_text = str(result)
                
                # Track direct string conversion
                Traceloop.set_association_properties({
                    "extraction_method": "direct_string",
                    "extraction_status": "success",
                    "text_length": len(result_text)
                })
                
                return result_text
                
            if isinstance(result.content, list):
                result_text = ""
                for item in result.content:
                    if hasattr(item, "text"):
                        result_text += item.text
                    else:
                        result_text += str(item)
                
                # Track list extraction
                Traceloop.set_association_properties({
                    "extraction_method": "list_content",
                    "extraction_status": "success",
                    "list_length": len(result.content),
                    "text_length": len(result_text)
                })
                
                return result_text
                
            elif isinstance(result.content, str):
                # Track string extraction
                Traceloop.set_association_properties({
                    "extraction_method": "string_content",
                    "extraction_status": "success",
                    "text_length": len(result.content)
                })
                
                return result.content
                
            else:
                result_text = str(result.content)
                
                # Track object extraction
                Traceloop.set_association_properties({
                    "extraction_method": "object_content",
                    "extraction_status": "success",
                    "object_type": type(result.content).__name__,
                    "text_length": len(result_text)
                })
                
                return result_text
                
        except Exception as e:
            # Track extraction error
            Traceloop.set_association_properties({
                "extraction_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            # Fall back to safe string conversion
            logger.error(f"Error extracting result text: {str(e)}", exc_info=True)
            return f"Error extracting result: {str(e)}"
    
    def _update_message_history(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result
        
        Args:
            messages: Conversation history to update
            tool_call: Tool call from LLM
            result_text: Result of tool execution
        """
        # Extract tool information from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
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
                        "arguments": tool_args
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
        
        # Log updated message history for debugging
        logger.debug(f"Updated message history with tool call and result. Now have {len(messages)} messages.")