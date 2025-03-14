from typing import Dict, List, Any
from datetime import datetime

from mcp import ClientSession
from traceloop.sdk.decorators import workflow, task, tool
from traceloop.sdk import Traceloop

class ServerInstance:
    """Manages the connection to a single MCP server"""
    
    def __init__(self, server_name: str, session: ClientSession):
        self.server_name = server_name
        self.session = session
        self.tools = []
        self.connected_at = datetime.now()
        
    @workflow(name="initialize_server")
    async def initialize(self) -> None:
        """Initialize the connection and fetch tools"""
        # Set association properties for the initialization
        Traceloop.set_association_properties({
            "server_name": self.server_name,
            "connection_timestamp": self.connected_at.isoformat()
        })
        
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
        
    @tool(name="execute_server_tool")
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool on this server
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        # Track the tool execution
        Traceloop.set_association_properties({
            "server_name": self.server_name,
            "tool_name": tool_name,
            "arguments": str(arguments)
        })
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            
            # Track successful execution
            if hasattr(result, 'content'):
                content_type = type(result.content).__name__
                content_sample = str(result.content)[:100] + "..." if len(str(result.content)) > 100 else str(result.content)
            else:
                content_type = type(result).__name__
                content_sample = str(result)[:100] + "..." if len(str(result)) > 100 else str(result)
                
            Traceloop.set_association_properties({
                "execution_status": "success",
                "result_type": content_type,
                "result_sample": content_sample
            })
            
            return result
            
        except Exception as e:
            # Track failed execution
            Traceloop.set_association_properties({
                "execution_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)[:200]  # Truncate long error messages
            })
            
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