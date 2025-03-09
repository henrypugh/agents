import json
import logging
from typing import Dict, List, Any

from mcp import ClientSession

logger = logging.getLogger("ToolManager")

class ToolManager:
    """Manages MCP tool operations and formatting"""
    
    @staticmethod
    async def get_available_tools(session: ClientSession) -> List[Dict[str, Any]]:
        """Get available tools from the MCP session in OpenAI format"""
        response = await session.list_tools()
        return [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]
    
    @staticmethod
    async def execute_tool_call(session: ClientSession, tool_call: Any) -> Dict[str, Any]:
        """Execute a tool call and format the result"""
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
        logger.debug(f"Tool result content: {result.content}")
        
        return {
            "call": tool_name,
            "result": result,
            "tool_call_id": tool_call.id,
            "result_text": ToolManager.extract_result_text(result)
        }
    
    @staticmethod
    def extract_result_text(result: Any) -> str:
        """Extract text content from a tool result"""
        if not hasattr(result, "content"):
            return str(result)
            
        if isinstance(result.content, list):
            result_text = ""
            for item in result.content:
                if hasattr(item, "text"):
                    result_text += item.text
            return result_text
        elif isinstance(result.content, str):
            return result.content
        else:
            return str(result.content)