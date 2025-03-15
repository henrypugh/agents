"""
Server instance module for managing connections to MCP servers.
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import AsyncExitStack
import logging

from mcp import ClientSession
from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.utils.schemas import ServerToolInfo, ServerInfo
from src.utils.decorators import server_connection, tool_execution, resource_cleanup

logger = logging.getLogger("ServerInstance")

class ServerInstance:
    """Manages a connection to a single MCP server"""
    
    def __init__(self, server_name: str, session: ClientSession):
        self.server_name = server_name
        self.session = session
        self.tools = []
        self.connected_at = datetime.now()
        self.stack = AsyncExitStack()  # Each instance manages its own resources
        
    @server_connection
    async def initialize(self) -> None:
        """Initialize the connection and fetch tools"""
        await self.refresh_tools()
        
    @task(name="refresh_tools")
    async def refresh_tools(self) -> List[Any]:
        """Refresh the list of available tools"""
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
        """Execute a tool on this server"""
        return await self.session.call_tool(tool_name, arguments)
        
    def get_tool_names(self) -> List[str]:
        """Get list of tool names"""
        return [tool.name for tool in self.tools]
        
    def get_openai_format_tools(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format with server metadata"""
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
    
    def get_tools_as_models(self) -> List[ServerToolInfo]:
        """Get tools as ServerToolInfo Pydantic models"""
        try:
            return [
                ServerToolInfo(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.inputSchema,
                    server_name=self.server_name
                )
                for tool in self.tools
            ]
        except Exception as e:
            logger.error(f"Error converting tools to models: {e}")
            return []
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server connection information"""
        return {
            "name": self.server_name,
            "tools": self.get_tool_names(),
            "connected_at": self.connected_at.isoformat(),
            "tool_count": len(self.tools)
        }
    
    def get_server_info_as_model(self) -> ServerInfo:
        """Get server connection information as ServerInfo model"""
        try:
            return ServerInfo(
                name=self.server_name,
                connected=True,
                tools=self.get_tool_names(),
                connected_at=self.connected_at,
                status="connected"
            )
        except Exception as e:
            logger.error(f"Error creating ServerInfo: {e}")
            return ServerInfo(
                name=self.server_name,
                connected=True,
                status="connected",
                tools=[]
            )
        
    @resource_cleanup
    async def cleanup(self) -> None:
        """Clean up resources specific to this server instance"""
        logger.info(f"Cleaning up resources for server '{self.server_name}'")
        
        # Track the cleanup operation
        Traceloop.set_association_properties({
            "server_name": self.server_name,
            "cleanup_operation": "server_instance"
        })
        
        await self.stack.aclose()
        logger.info(f"Successfully cleaned up resources for server '{self.server_name}'")