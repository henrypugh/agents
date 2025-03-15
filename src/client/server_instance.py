from typing import Dict, List, Any
from datetime import datetime

from mcp import ClientSession
from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.decorators import server_connection, tool_execution

class ServerInstance:
    """Manages the connection to a single MCP server"""
    
    def __init__(self, server_name: str, session: ClientSession):
        self.server_name = server_name
        self.session = session
        self.tools = []
        self.connected_at = datetime.now()
        
    @server_connection
    async def initialize(self) -> None:
        """Initialize the connection and fetch tools"""
        await self.refresh_tools()
        
    @task(name="refresh_tools")
    async def refresh_tools(self) -> List[Any]:
        """
        Refresh the list of available tools
        
        Returns:
            List of available tools
        """
        response = await self.session.list_tools()
        self.tools = response.tools
        
        # Track the tools discovered for this server
        tool_names = [tool.name for tool in self.tools]
        Traceloop.set_association_properties({
            "tool_count": len(tool_names),
            "tools": ",".join(tool_names)
        })
        
        return self.tools
        
    @tool_execution
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on this server
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        try:
            result = await self.session.call_tool(tool_name, arguments)
            return result
            
        except Exception as e:
            # Re-raise the exception for proper error handling
            raise
        
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