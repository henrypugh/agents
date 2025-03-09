import json
import logging
from typing import Dict, List, Any

from mcp import ClientSession

logger = logging.getLogger("ToolManager")

class ToolManager:
    """Manages MCP tool operations and formatting"""
    
    @staticmethod
    async def get_available_tools(session: ClientSession, server_name: str = None) -> List[Dict[str, Any]]:
        """Get available tools from the MCP session in OpenAI format
        
        Args:
            session: The MCP client session
            server_name: Optional name of the server to tag tools with
            
        Returns:
            List of tools in OpenAI format with server metadata
        """
        response = await session.list_tools()
        
        tools = []
        for tool in response.tools:
            tool_dict = { 
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or f"Tool: {tool.name}",
                    "parameters": tool.inputSchema
                }
            }
            
            # Add server information if provided
            if server_name:
                # Add metadata to store server origin
                if "metadata" not in tool_dict["function"]:
                    tool_dict["function"]["metadata"] = {}
                tool_dict["function"]["metadata"]["server"] = server_name
                
                # Enhance description with server info to help LLM route correctly
                description = tool_dict["function"]["description"]
                tool_dict["function"]["description"] = f"[From {server_name} server] {description}"
            
            tools.append(tool_dict)
            
        logger.info(f"Found {len(tools)} tools from {server_name or 'unknown'} server")
        return tools
    
    @staticmethod
    async def execute_tool_call(session: ClientSession, tool_call: Any) -> Dict[str, Any]:
        """Execute a tool call and format the result
        
        Args:
            session: The MCP client session to use
            tool_call: The tool call from the LLM
            
        Returns:
            Formatted result with metadata
        """
        tool_name = tool_call.function.name
        tool_args_raw = tool_call.function.arguments
        
        try:
            tool_args = json.loads(tool_args_raw)
            logger.info(f"Calling tool: {tool_name} with parsed args: {tool_args}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool arguments: {e}")
            logger.info(f"Attempting to use raw arguments string")
            tool_args = tool_args_raw
        
        # Execute tool call
        logger.info(f"Executing tool call to {tool_name}")
        result = await session.call_tool(tool_name, tool_args)
        logger.info(f"Tool call result received")
        logger.debug(f"Tool result content: {result.content if hasattr(result, 'content') else result}")
        
        return {
            "call": tool_name,
            "result": result,
            "tool_call_id": tool_call.id,
            "result_text": ToolManager.extract_result_text(result)
        }
    
    @staticmethod
    def extract_result_text(result: Any) -> str:
        """Extract text content from a tool result
        
        Args:
            result: The result from a tool call
            
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