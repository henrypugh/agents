"""
Server Management module for handling server-related tools and operations.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor
from src.utils.schemas import ToolCall, Message, ServerInfo

logger = logging.getLogger("ServerManagement")

class ServerManagementHandler:
    """Handles server management operations"""
    
    def __init__(
        self,
        server_manager: ServerRegistry,
        tool_processor: ToolExecutor
    ):
        """
        Initialize the server management handler
        
        Args:
            server_manager: Server manager instance
            tool_processor: Tool processor instance
        """
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        logger.info("ServerManagementHandler initialized")
    
    def create_server_management_tools(self) -> List[Dict[str, Any]]:
        """
        Create tools for server management that will be available to the LLM
        
        Returns:
            List of server management tools in OpenAI format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_available_servers",
                    "description": "List all available MCP servers that can be connected to",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "metadata": {
                        "internal": True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "connect_to_server",
                    "description": "Connect to an MCP server to access its tools",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "server_name": {
                                "type": "string",
                                "description": "Name of the server to connect to"
                            }
                        },
                        "required": ["server_name"]
                    },
                    "metadata": {
                        "internal": True
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_connected_servers",
                    "description": "List all currently connected MCP servers and their available tools",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    },
                    "metadata": {
                        "internal": True
                    }
                }
            }
        ]
    
    @task(name="handle_server_management_tool")
    async def handle_server_management_tool(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        response_processor
    ) -> None:
        """
        Handle server management tool calls
        
        Args:
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
            response_processor: Response processor for follow-up handling
        """
        # Extract tool information from potentially different structures
        tool_name, tool_args_raw, tool_id = self._extract_tool_info(tool_call)
        logger.info(f"Handling server management tool: {tool_name}")
        
        # Track the management tool being used
        Traceloop.set_association_properties({
            "management_tool": tool_name
        })
        
        try:
            # Parse arguments if any
            tool_args = {}
            if tool_args_raw:
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError as e:
                    final_text.append(f"Error parsing arguments: {str(e)}")
                    
                    # Track the error
                    Traceloop.set_association_properties({
                        "error": "json_decode",
                        "error_message": str(e)
                    })
                    return
            
            # Handle each tool type
            if tool_name == "list_available_servers":
                await self._handle_list_available_servers(tool_call, messages, final_text)
            elif tool_name == "connect_to_server":
                await self._handle_connect_to_server(tool_call, tool_args, messages, final_text, response_processor)
            elif tool_name == "list_connected_servers":
                await self._handle_list_connected_servers(tool_call, messages, final_text)
            else:
                final_text.append(f"Unknown server management tool: {tool_name}")
                
                # Track the error
                Traceloop.set_association_properties({
                    "error": "unknown_management_tool"
                })
        
        except Exception as e:
            logger.error(f"Error handling server management tool: {str(e)}", exc_info=True)
            final_text.append(f"Error: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "management_tool_execution",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
    
    @staticmethod
    def _extract_tool_info(tool_call: Any) -> tuple:
        """Extract tool name, arguments, and ID from tool call object"""
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        
        return tool_name, tool_args, tool_id
    
    @staticmethod
    def as_tool_call_model(tool_call: Any) -> Optional[ToolCall]:
        """Convert raw tool call to ToolCall model"""
        try:
            if hasattr(tool_call, 'function'):
                return ToolCall(
                    id=getattr(tool_call, 'id', 'unknown_id'),
                    tool_name=tool_call.function.name,
                    arguments=tool_call.function.arguments
                )
            else:
                return ToolCall(
                    id=getattr(tool_call, 'id', 'unknown_id'),
                    tool_name=getattr(tool_call, 'name', 'unknown'),
                    arguments=getattr(tool_call, 'arguments', '{}')
                )
        except Exception as e:
            logger.error(f"Error converting to ToolCall model: {e}")
            return None

    @task(name="list_available_servers")
    async def _handle_list_available_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_available_servers tool"""
        # Extract tool ID
        _, _, tool_id = self._extract_tool_info(tool_call)
            
        available_servers = await self.server_manager.get_available_servers()
        
        # Track available servers
        Traceloop.set_association_properties({
            "available_server_count": len(available_servers),
            "available_servers": ",".join(available_servers.keys())
        })
        
        # Format for JSON response
        result = {
            "available_servers": {},
            "count": len(available_servers)
        }
        
        for server_name, server_info in available_servers.items():
            result["available_servers"][server_name] = server_info
        
        # Format for display
        if available_servers:
            server_list = []
            for server_name, info in available_servers.items():
                server_type = info.get("type", "unknown")
                source = info.get("source", "unknown")
                server_list.append(f"{server_name} ({server_type} from {source})")
            
            result_text = f"Available servers ({len(available_servers)}):\n" + "\n".join(server_list)
        else:
            result_text = "No available servers found"
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))
    
    async def list_available_servers_as_models(self) -> List[ServerInfo]:
        """Get available servers as ServerInfo models"""
        try:
            server_models = []
            available_servers = await self.server_manager.get_available_servers()
            
            for server_name, server_info in available_servers.items():
                server_models.append(ServerInfo(
                    name=server_name,
                    connected=False,
                    status="disconnected",
                    tools=[]
                ))
            return server_models
        except Exception as e:
            logger.error(f"Error getting server models: {e}")
            return []
            
    @task(name="connect_to_server")
    async def _handle_connect_to_server(
        self,
        tool_call: Any,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        final_text: List[str],
        response_processor
    ) -> None:
        """Handle connect_to_server tool"""
        if "server_name" not in args:
            error_msg = "Missing required argument: server_name"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({"error": error_msg}))
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "missing_server_name"
            })
            return
            
        server_name = args["server_name"]
        
        # Track the server being connected to
        Traceloop.set_association_properties({
            "target_server": server_name
        })
        
        # Get available servers
        available_servers = await self.server_manager.get_available_servers()
        
        if server_name not in available_servers:
            error_msg = f"Server '{server_name}' not found. Available servers: {', '.join(available_servers.keys())}"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({"error": error_msg}))
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "server_not_found",
                "available_servers": ",".join(available_servers.keys())
            })
            return
        
        # Connect to the server
        try:
            connection_result = await self.server_manager.connect_to_configured_server(server_name)
            result_text = f"Successfully connected to server: {server_name}"
            final_text.append(f"[Server management] {result_text}")
            
            # Track successful connection
            Traceloop.set_association_properties({
                "connection_status": "success",
                "tool_count": len(connection_result.get("tools", []))
            })
            
            # Update message history
            self._update_message_history(messages, tool_call, json.dumps({
                "success": True,
                "server_name": server_name,
                "message": result_text
            }))
            
            # Get follow-up response with updated tools
            updated_tools = self.server_manager.collect_all_tools() + self.create_server_management_tools()
            await response_processor.get_follow_up_response(messages, updated_tools, final_text)
            
        except Exception as e:
            error_msg = f"Failed to connect to server '{server_name}': {str(e)}"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({
                "success": False,
                "error": error_msg
            }))
            
            # Track the error
            Traceloop.set_association_properties({
                "connection_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

    @task(name="list_connected_servers")
    async def _handle_list_connected_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_connected_servers tool"""
        connected_servers = self.server_manager.get_connected_servers()
        
        # Track connected servers
        Traceloop.set_association_properties({
            "connected_server_count": len(connected_servers),
            "connected_servers": ",".join(connected_servers.keys())
        })
         
        # Format for JSON response
        result = {
            "connected_servers": {},
            "count": len(connected_servers)
        }
        
        for server_name, server_info in connected_servers.items():
            result["connected_servers"][server_name] = {
                "tools": [tool["function"]["name"] for tool in server_info.get("tools", [])]
            }
        
        # Format for display
        if connected_servers:
            server_list = []
            for server_name, info in connected_servers.items():
                tools = [tool["function"]["name"] for tool in info.get("tools", [])]
                tool_count = len(tools)
                server_list.append(f"{server_name} ({tool_count} tools available)")
            
            result_text = f"Connected servers ({len(connected_servers)}):\n" + "\n".join(server_list)
        else:
            result_text = "No servers currently connected"
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))
    
    def get_connected_servers_as_models(self) -> List[ServerInfo]:
        """Get connected servers as ServerInfo models"""
        try:
            server_models = []
            connected_servers = self.server_manager.get_connected_servers()
            
            for server_name, server_info in connected_servers.items():
                tool_names = [tool["function"]["name"] for tool in server_info.get("tools", [])]
                server_models.append(ServerInfo(
                    name=server_name,
                    connected=True,
                    status="connected",
                    tools=tool_names
                ))
            return server_models
        except Exception as e:
            logger.error(f"Error getting connected server models: {e}")
            return []
    
    def _update_message_history(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """Helper method to update message history"""
        # Extract tool information
        tool_name, tool_args, tool_id = self._extract_tool_info(tool_call)
            
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
    
    def _update_message_history_with_models(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """Update message history using Pydantic models"""
        try:
            # Extract tool information and create models
            tool_call_model = self.as_tool_call_model(tool_call)
            if not tool_call_model:
                # Fall back to standard update if model creation fails
                self._update_message_history(messages, tool_call, result_text)
                return
                
            # Create assistant message with tool call
            assistant_message = Message(
                role="assistant",
                content=None,
                tool_calls=[tool_call_model]
            )
            
            # Create tool response message
            tool_response = Message(
                role="tool",
                tool_call_id=tool_call_model.id,
                content=result_text
            )
            
            # Add to messages in OpenAI format
            messages.append(assistant_message.to_openai_format())
            messages.append(tool_response.to_openai_format())
            
        except Exception as e:
            logger.error(f"Error updating message history with models: {e}")
            # Fall back to standard update
            self._update_message_history(messages, tool_call, result_text)