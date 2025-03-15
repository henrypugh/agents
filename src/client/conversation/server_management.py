"""
Server Management module for handling server-related tools and operations.

This module provides Pydantic-integrated tools and handlers for server management,
including server discovery, connection, and tool access.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable
from enum import Enum

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor
from src.utils.schemas import (
    ToolCall,
    Message,
    ServerInfo,
    ServerListResponse,
    ConnectResponse,
    MessageRole,
    ConnectResponseStatus
)

logger = logging.getLogger(__name__)

class ResponseProcessorProtocol(Protocol):
    """Protocol defining the response processor follow-up method"""
    async def get_follow_up_response(
        self,
        messages: List[Message],
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        ...

class ServerManagementHandler:
    """
    Handles server management operations with Pydantic validation.
    
    This class handles:
    - Server discovery and connection
    - Creation of server management tools
    - Processing of server management tool calls
    """
    
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
        tool_call: ToolCall,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        response_processor: ResponseProcessorProtocol
    ) -> None:
        """
        Handle server management tool calls
        
        Args:
            tool_call: Validated tool call model
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
            response_processor: Response processor for follow-up handling
        """
        logger.info(f"Handling server management tool: {tool_call.tool_name}")
        
        # Track the management tool being used
        Traceloop.set_association_properties({
            "management_tool": tool_call.tool_name
        })
        
        try:
            # Handle each tool type
            if tool_call.tool_name == "list_available_servers":
                await self._handle_list_available_servers(tool_call, messages, final_text)
            elif tool_call.tool_name == "connect_to_server":
                await self._handle_connect_to_server(tool_call, messages, final_text, response_processor, tools)
            elif tool_call.tool_name == "list_connected_servers":
                await self._handle_list_connected_servers(tool_call, messages, final_text)
            else:
                final_text.append(f"Unknown server management tool: {tool_call.tool_name}")
                
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

    @task(name="list_available_servers")
    async def _handle_list_available_servers(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        final_text: List[str]
    ) -> None:
        """
        Handle list_available_servers tool
        
        Args:
            tool_call: Validated tool call
            messages: Conversation history
            final_text: List to append text to
        """
        # Get available servers
        available_servers = await self.server_manager.get_available_servers()
        
        # Track available servers
        Traceloop.set_association_properties({
            "available_server_count": len(available_servers),
            "available_servers": ",".join(available_servers.keys())
        })
        
        # Create response model
        server_info_dict = {}
        for server_name, server_info in available_servers.items():
            is_connected = server_info.get("connected", False)
            server_type = server_info.get("type", "unknown")
            server_source = server_info.get("source", "unknown")
            
            server_info_dict[server_name] = ServerInfo(
                name=server_name,
                connected=is_connected,
                status="connected" if is_connected else "disconnected",
                tools=[]
            )
            
        response = ServerListResponse(
            available_servers=server_info_dict,
            count=len(available_servers)
        )
        
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
        self._update_message_history(messages, tool_call, response.model_dump_json())
    
    @task(name="connect_to_server")
    async def _handle_connect_to_server(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        final_text: List[str],
        response_processor: ResponseProcessorProtocol,
        tools: List[Dict[str, Any]]
    ) -> None:
        """
        Handle connect_to_server tool
        
        Args:
            tool_call: Validated tool call
            messages: Conversation history
            final_text: List to append text to
            response_processor: Response processor for follow-up handling
            tools: Available tools
        """
        # Validate arguments
        if "server_name" not in tool_call.arguments:
            error_response = ConnectResponse(
                status=ConnectResponseStatus.ERROR,
                server="unknown",
                error="Missing required argument: server_name"
            )
            
            final_text.append(f"[Server management] Error: {error_response.error}")
            self._update_message_history(messages, tool_call, error_response.model_dump_json())
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "missing_server_name"
            })
            return
            
        server_name = tool_call.arguments["server_name"]
        
        # Track the server being connected to
        Traceloop.set_association_properties({
            "target_server": server_name
        })
        
        # Get available servers
        available_servers = await self.server_manager.get_available_servers()
        
        if server_name not in available_servers:
            error_response = ConnectResponse(
                status=ConnectResponseStatus.ERROR,
                server=server_name,
                error=f"Server '{server_name}' not found. Available servers: {', '.join(available_servers.keys())}"
            )
            
            final_text.append(f"[Server management] Error: {error_response.error}")
            self._update_message_history(messages, tool_call, error_response.model_dump_json())
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "server_not_found",
                "available_servers": ",".join(available_servers.keys())
            })
            return
        
        # Connect to the server
        try:
            connection_result = await self.server_manager.connect_to_configured_server(server_name)
            
            if connection_result.get("status") == "already_connected":
                response = ConnectResponse(
                    status=ConnectResponseStatus.ALREADY_CONNECTED,
                    server=server_name,
                    tools=connection_result.get("tools", []),
                    tool_count=connection_result.get("tool_count", 0)
                )
                result_text = f"Server {server_name} is already connected"
            else:
                response = ConnectResponse(
                    status=ConnectResponseStatus.CONNECTED,
                    server=server_name,
                    tools=connection_result.get("tools", []),
                    tool_count=connection_result.get("tool_count", 0)
                )
                result_text = f"Successfully connected to server: {server_name}"
                
            final_text.append(f"[Server management] {result_text}")
            
            # Track successful connection
            Traceloop.set_association_properties({
                "connection_status": "success",
                "tool_count": response.tool_count or 0
            })
            
            # Update message history
            self._update_message_history(messages, tool_call, response.model_dump_json())
            
            # Get follow-up response with updated tools
            updated_tools = self.server_manager.collect_all_tools() + self.create_server_management_tools()
            await response_processor.get_follow_up_response(messages, updated_tools, final_text)
            
        except Exception as e:
            error_response = ConnectResponse(
                status=ConnectResponseStatus.ERROR,
                server=server_name,
                error=f"Failed to connect to server: {str(e)}"
            )
            
            final_text.append(f"[Server management] Error: {error_response.error}")
            self._update_message_history(messages, tool_call, error_response.model_dump_json())
            
            # Track the error
            Traceloop.set_association_properties({
                "connection_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

    @task(name="list_connected_servers")
    async def _handle_list_connected_servers(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        final_text: List[str]
    ) -> None:
        """
        Handle list_connected_servers tool
        
        Args:
            tool_call: Validated tool call
            messages: Conversation history
            final_text: List to append text to
        """
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
    
    def _update_message_history(
        self,
        messages: List[Message],
        tool_call: ToolCall,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result
        
        Args:
            messages: Conversation history
            tool_call: Tool call model
            result_text: Result of tool execution
        """
        try:
            # Create assistant message with tool call
            assistant_message = Message.assistant(tool_calls=[tool_call])
            
            # Create tool response message
            tool_message = Message.tool(tool_call_id=tool_call.id, content=result_text)
            
            # Add messages to history
            messages.append(assistant_message)
            messages.append(tool_message)
            
        except Exception as e:
            logger.error(f"Error updating message history: {e}")
            # Fall back to creating dictionaries directly
            tool_call_message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.tool_name,
                            "arguments": json.dumps(tool_call.arguments)
                        }
                    }
                ]
            }
            
            tool_response_message = {
                "role": "tool", 
                "tool_call_id": tool_call.id,
                "content": result_text
            }
            
            messages.append(tool_call_message)
            messages.append(tool_response_message)