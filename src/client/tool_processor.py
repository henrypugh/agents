"""
Tool Processor module for handling tool execution and processing.
"""

import logging
import json
from typing import Dict, List, Optional, Any

from traceloop.sdk.decorators import tool
from traceloop.sdk import Traceloop
from .server_manager import ServerManager

logger = logging.getLogger("ToolProcessor")

class ToolProcessor:
    """Processes and executes tool calls from LLMs"""
    
    def __init__(self, server_manager: ServerManager):
        """
        Initialize the tool processor
        
        Args:
            server_manager: Server manager instance
        """
        self.server_manager = server_manager
        logger.info("ToolProcessor initialized")
    
    def find_server_for_tool(self, tool_name: str, tools: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find which server handles the specified tool
        
        Args:
            tool_name: Name of the tool
            tools: List of available tools
            
        Returns:
            Name of the server or None if not found
        """
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
    
    @tool(name="execute_tool")
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
        # Associate this tool execution with the tool name and server for tracing
        Traceloop.set_association_properties({
            "tool_name": tool_name,
            "server_name": server_name,
            "tool_args": json.dumps(tool_args)[:200] if tool_args else "{}"
        })
        
        server = self.server_manager.get_server(server_name)
        if not server:
            raise ValueError(f"Server '{server_name}' not found")
        
        logger.info(f"Executing tool '{tool_name}' on server '{server_name}'")
        return await server.execute_tool(tool_name, tool_args)
    
    def extract_result_text(self, result: Any) -> str:
        """
        Extract text content from a tool result
        
        Args:
            result: Result from tool execution
            
        Returns:
            Extracted text content
        """
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