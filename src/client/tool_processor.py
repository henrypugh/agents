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
from .server_registry import ServerRegistry

logger = logging.getLogger("ToolExecutor")

class ToolExecutor:
    """Processes and executes tool calls from LLMs"""
    
    def __init__(self, server_manager: ServerRegistry):
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