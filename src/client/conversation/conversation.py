"""
Main conversation interface for handling LLM interactions.

This module provides a Pydantic-integrated conversation manager
that coordinates message processing, response handling, and tool execution.
"""

import logging
from typing import Dict, List, Any, Optional, TypeVar, Type, Protocol, Union
from pydantic import TypeAdapter

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task

from src.utils.decorators import message_processing
from src.utils.schemas import (
    Message, 
    MessageRole, 
    ToolCall, 
    LLMResponse,
    ServerInfo,
    ServerToolInfo,
    OpenAIAdapter
)
from .message_processor import MessageProcessor, MessageProcessorConfig
from .response_processor import ResponseProcessor
from .server_management import ServerManagementHandler
from src.client.llm_service import LLMService
from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor

logger = logging.getLogger(__name__)

class ConversationConfig:
    """Configuration for the conversation manager"""
    def __init__(
        self,
        trace_tools: bool = True,
        include_server_management: bool = True,
        validate_responses: bool = True
    ):
        self.trace_tools = trace_tools
        self.include_server_management = include_server_management
        self.validate_responses = validate_responses


class Conversation:
    """
    Manages conversations with LLMs and tool execution using Pydantic models.
    
    This class orchestrates:
    - Message processing and history management
    - Tool discovery and execution
    - LLM response handling
    - Server management
    """
    
    def __init__(
        self, 
        llm_client: LLMService, 
        server_manager: ServerRegistry,
        tool_processor: ToolExecutor,
        message_processor_config: Optional[MessageProcessorConfig] = None,
        config: Optional[ConversationConfig] = None
    ):
        """
        Initialize the conversation manager
        
        Args:
            llm_client: LLM client instance
            server_manager: Server manager instance
            tool_processor: Tool processor instance
            message_processor_config: Optional configuration for message processor
            config: Optional configuration for conversation
        """
        self.llm_client = llm_client
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        self.config = config or ConversationConfig()
        
        # Initialize component handlers
        self.server_handler = ServerManagementHandler(server_manager, tool_processor)
        self.message_processor = MessageProcessor(
            llm_client, 
            config=message_processor_config
        )
        self.response_processor = ResponseProcessor(
            llm_client,
            tool_processor,
            self.server_handler
        )
        
        # Get server management tools
        self.server_management_tools = self.server_handler.create_server_management_tools() if self.config.include_server_management else []
        
        # Create type adapters for validation
        self.server_info_validator = TypeAdapter(ServerInfo)
        self.server_tool_info_validator = TypeAdapter(ServerToolInfo)
        
        logger.info("Conversation initialized")
    
    @message_processing()
    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM and available tools
        
        Args:
            query: User query text
            
        Returns:
            Generated response
        """
        logger.info(f"Processing query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Create validated user message
        user_message = Message.user(query)
        
        # Initialize conversation with validated user message
        messages = [user_message]
        
        # Collect tools from all connected servers and add server management tools
        all_tools = self._collect_tools()
        
        # Track conversation metrics
        Traceloop.set_association_properties({
            "query_length": len(query),
            "tool_count": len(all_tools),
            "server_management_tools": len(self.server_management_tools) if self.config.include_server_management else 0,
            "connected_servers": len(self.server_manager.servers),
        })
        
        # Run the conversation
        return await self.message_processor.run_conversation(
            messages, 
            all_tools,
            self.response_processor
        )
    
    @message_processing()
    async def process_query_with_history(
        self, 
        query: str, 
        history: List[Message]
    ) -> str:
        """
        Process a user query with existing conversation history
        
        Args:
            query: User query text
            history: Existing conversation history
            
        Returns:
            Generated response
        """
        logger.info(f"Processing query with history: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        # Create validated user message
        user_message = Message.user(query)
        
        # Add user message to history
        messages = history + [user_message]
        
        # Collect tools from all connected servers and add server management tools
        all_tools = self._collect_tools()
        
        # Track conversation metrics
        Traceloop.set_association_properties({
            "query_length": len(query),
            "history_length": len(history),
            "tool_count": len(all_tools),
            "server_management_tools": len(self.server_management_tools) if self.config.include_server_management else 0,
            "connected_servers": len(self.server_manager.servers),
        })
        
        # Run the conversation
        return await self.message_processor.run_conversation(
            messages, 
            all_tools,
            self.response_processor
        )
    
    @task(name="collect_tools")
    def _collect_tools(self) -> List[Dict[str, Any]]:
        """
        Collect tools from all sources and prepare for LLM
        
        Returns:
            Combined list of tools in OpenAI format
        """
        # Collect tools from all connected servers
        server_tools = self.server_manager.collect_all_tools()
        
        # Track tool collection
        if self.config.trace_tools:
            tools_by_server = {}
            for server_name, server in self.server_manager.servers.items():
                tools_by_server[server_name] = server.get_tool_names()
                
            Traceloop.set_association_properties({
                "tools_by_server": tools_by_server,
                "server_tools_count": len(server_tools),
                "management_tools_count": len(self.server_management_tools)
            })
        
        # Combine with server management tools if enabled
        return server_tools + self.server_management_tools
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools including server management tools
        
        Returns:
            List of tools in OpenAI format
        """
        return self._collect_tools()
    
    def get_available_tools_as_models(self) -> List[ServerToolInfo]:
        """
        Get all available tools as Pydantic models
        
        Returns:
            List of tool info models
        """
        return self.server_manager.collect_all_tools_as_models()
    
    @task(name="get_available_servers")
    async def get_available_servers(self) -> List[ServerInfo]:
        """
        Get list of available servers as Pydantic models
        
        Returns:
            List of server info models
        """
        return await self.server_manager.get_available_servers_as_models()
    
    @task(name="get_connected_servers")
    def get_connected_servers(self) -> List[ServerInfo]:
        """
        Get list of connected servers as Pydantic models
        
        Returns:
            List of server info models
        """
        return self.server_manager.get_connected_servers_as_models()
    
    @staticmethod
    def create_system_prompt(content: str) -> Message:
        """
        Create a system prompt message
        
        Args:
            content: System prompt content
            
        Returns:
            System message
        """
        return Message.system(content)
        
    @task(name="add_system_prompt")
    def add_system_prompt(self, messages: List[Message], system_prompt: str) -> List[Message]:
        """
        Add or update system prompt in message history
        
        Args:
            messages: Existing message history
            system_prompt: System prompt content
            
        Returns:
            Updated message history
        """
        # Check if there's already a system message
        system_index = next((i for i, m in enumerate(messages) if m.role == MessageRole.SYSTEM), None)
        
        if system_index is not None:
            # Replace existing system message
            system_message = Message.system(system_prompt)
            messages = messages.copy()  # Create a copy to avoid modifying the original
            messages[system_index] = system_message
        else:
            # Add system message at the beginning
            system_message = Message.system(system_prompt)
            messages = [system_message] + messages
            
        return messages
        
    def configure_server_management(self, enable: bool) -> None:
        """
        Enable or disable server management tools
        
        Args:
            enable: Whether to enable server management tools
        """
        self.config.include_server_management = enable
        
        # Refresh server management tools
        if enable:
            self.server_management_tools = self.server_handler.create_server_management_tools()
        else:
            self.server_management_tools = []
            
        logger.info(f"Server management tools {'enabled' if enable else 'disabled'}")