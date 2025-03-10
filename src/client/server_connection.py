from typing import Dict, List, Any
from datetime import datetime

from mcp import ClientSession


class ServerConnection:
    """Manages the connection to a single MCP server"""
    
    def __init__(self, server_name: str, session: ClientSession):
        self.server_name = server_name
        self.session = session
        self.tools = []
        self.connected_at = datetime.now()
        
    async def initialize(self) -> None:
        """Initialize the connection and fetch tools"""
        await self.refresh_tools()
        
    async def refresh_tools(self) -> List[Any]:
        """
        Refresh the list of available tools
        
        Returns:
            List of available tools
        """
        response = await self.session.list_tools()
        self.tools = response.tools
        return self.tools
        
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on this server
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        return await self.session.call_tool(tool_name, arguments)
        
    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [tool.name for tool in self.tools]
        
    def get_openai_format_tools(self) -> List[Dict[str, Any]]:
        """
        Get tools in OpenAI format with server metadata
        
        Returns:
            List of tools formatted for OpenAI API
        """
        formatted_tools = []
        
        for tool in self.tools:
            tool_dict = { 
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": f"[From {self.server_name} server] {tool.description or f'Tool: {tool.name}'}",
                    "parameters": tool.inputSchema,
                    "metadata": {
                        "server": self.server_name
                    }
                }
            }
            formatted_tools.append(tool_dict)
            
        return formatted_tools

