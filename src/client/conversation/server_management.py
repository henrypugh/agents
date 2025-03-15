"""
Server Management module for handling server-related tools and operations.

This module provides Pydantic-integrated tools and handlers for server management,
including server discovery, connection, and tool access.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable, Set, TypeVar, Union
from enum import Enum, auto
from pydantic import TypeAdapter, ValidationError

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
    ConnectResponseStatus,
    OpenAIAdapter,
    MessageHistory
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

class ServerManagementConfig:
    """Configuration for server management"""
    def __init__(
        self,
        trace_operations: bool = True,
        secure_mode: bool = False,
        validate_responses: bool = True,
        max_connect_attempts: int = 2
    ):
        self.trace_operations = trace_operations
        self.secure_mode = secure_mode  # If True, restricts some server operations
        self.validate_responses = validate_responses
        self.max_connect_attempts = max_connect_attempts


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
        tool_processor: ToolExecutor,
        config: Optional[ServerManagementConfig] = None
    ):
        """
        Initialize the server management handler
        
        Args:
            server_manager: Server manager instance
            tool_processor: Tool processor instance
            config: Optional configuration
        """
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        self.config = config or ServerManagementConfig()
        
        # Create type adapters for validation
        self.server_list_validator = TypeAdapter(ServerListResponse)
        self.connect_response_validator = TypeAdapter(ConnectResponse)
        
        logger.info("ServerManagementHandler initialized")
    
    def create_server_management_tools(self) -> List[Dict[str, Any]]:
        """
        Create tools for server management that will be available to the LLM
        
        Returns:
            List of server management tools in OpenAI format
        """
        # Base tools always available
        tools = [
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
        
        # Add connect tool unless secure mode is enabled
        if not self.config.secure_mode:
            tools.append({
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
            })
            
        return tools
    
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
        
        # Track the operation start time
        operation_start = time.time()
        
        # Track the management tool being used
        if self.config.trace_operations:
            Traceloop.set_association_properties({
                "management_tool": tool_call.tool_name,
                "operation_start": operation_start
            })
        
        try:
            # Handle each tool type
            if tool_call.tool_name == "list_available_servers":
                await self._handle_list_available_servers(tool_call, messages, final_text)
            elif tool_call.tool_name == "connect_to_server":
                if self.config.secure_mode:
                    final_text.append(f"[Server management] Error: connect_to_server is disabled in secure mode")
                    self._update_message_history(messages, tool_call, json.dumps({
                        "status": "error",
                        "error": "Server connections are restricted in secure mode"
                    }))
                else:
                    await self._handle_connect_to_server(tool_call, messages, final_text, response_processor, tools)
            elif tool_call.tool_name == "list_connected_servers":
                await self._handle_list_connected_servers(tool_call, messages, final_text)
            else:
                final_text.append(f"Unknown server management tool: {tool_call.tool_name}")
                
                # Track the error
                if self.config.trace_operations:
                    Traceloop.set_association_properties({
                        "error": "unknown_management_tool"
                    })
                    
            # Track operation duration
            if self.config.trace_operations:
                operation_duration = time.time() - operation_start
                Traceloop.set_association_properties({
                    "operation_duration": operation_duration,
                    "operation_complete": True
                })
                
        except Exception as e:
            logger.error(f"Error handling server management tool: {str(e)}", exc_info=True)
            final_text.append(f"Error: {str(e)}")
            
            # Track the error
            if self.config.trace_operations:
                operation_duration = time.time() - operation_start
                Traceloop.set_association_properties({
                    "error": "management_tool_execution",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "operation_duration": operation_duration
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
        if self.config.trace_operations:
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
        
        # Add to final text with consistent formatting
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
            if self.config.trace_operations:
                Traceloop.set_association_properties({
                    "error": "missing_server_name"
                })
            return
            
        server_name = tool_call.arguments["server_name"]
        
        # Track the server being connected to
        if self.config.trace_operations:
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
            if self.config.trace_operations:
                Traceloop.set_association_properties({
                    "error": "server_not_found",
                    "available_servers": ",".join(available_servers.keys())
                })
            return
        
        # Connect to the server with retry
        for attempt in range(self.config.max_connect_attempts):
            try:
                # Track attempt
                if self.config.trace_operations and attempt > 0:
                    Traceloop.set_association_properties({
                        "connect_attempt": attempt + 1
                    })
                    
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
                if self.config.trace_operations:
                    Traceloop.set_association_properties({
                        "connection_status": "success",
                        "tool_count": response.tool_count or 0
                    })
                
                # Update message history
                self._update_message_history(messages, tool_call, response.model_dump_json())
                
                # Get follow-up response with updated tools
                updated_tools = self.server_manager.collect_all_tools() + self.create_server_management_tools()
                await response_processor.get_follow_up_response(messages, updated_tools, final_text)
                
                # Successful connection, exit retry loop
                break
                
            except Exception as e:
                logger.error(f"Error connecting to server (attempt {attempt+1}): {e}")
                
                # Track error
                if self.config.trace_operations:
                    Traceloop.set_association_properties({
                        "connect_attempt": attempt + 1,
                        "connection_error": str(e),
                        "error_type": type(e).__name__
                    })
                
                # Last attempt - report error
                if attempt == self.config.max_connect_attempts - 1:
                    error_response = ConnectResponse(
                        status=ConnectResponseStatus.ERROR,
                        server=server_name,
                        error=f"Failed to connect to server: {str(e)}"
                    )
                    
                    final_text.append(f"[Server management] Error: {error_response.error}")
                    self._update_message_history(messages, tool_call, error_response.model_dump_json())

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
        if self.config.trace_operations:
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
        MessageHistory.add_tool_interaction(messages, tool_call, result_text)
        
    def set_secure_mode(self, enable: bool) -> None:
        """
        Enable or disable secure mode
        
        Args:
            enable: Whether to enable secure mode
        """
        self.config.secure_mode = enable
        logger.info(f"Secure mode {'enabled' if enable else 'disabled'}")