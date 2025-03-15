"""
Main conversation interface for handling LLM interactions.

This module provides a Pydantic-integrated conversation manager
that coordinates message processing, response handling, and tool execution.
"""

import logging
from typing import Dict, List, Any, Optional

from traceloop.sdk import Traceloop

from src.utils.decorators import message_processing
from src.utils.schemas import (
    Message, 
    MessageRole, 
    ToolCall, 
    LLMResponse,
    ServerInfo,
    ServerToolInfo
)
from .message_processor import MessageProcessor
from .response_processor import ResponseProcessor
from .server_management import ServerManagementHandler
from src.client.llm_service import LLMService
from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor

logger = logging.getLogger(__name__)

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
        tool_processor: ToolExecutor
    ):
        """
        Initialize the conversation manager
        
        Args:
            llm_client: LLM client instance
            server_manager: Server manager instance
            tool_processor: Tool processor instance
        """
        self.llm_client = llm_client
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        
        # Initialize component handlers
        self.server_handler = ServerManagementHandler(server_manager, tool_processor)
        self.message_processor = MessageProcessor(llm_client)
        self.response_processor = ResponseProcessor(
            llm_client,
            tool_processor,
            self.server_handler
        )
        
        # Get server management tools
        self.server_management_tools = self.server_handler.create_server_management_tools()
        
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
        logger.info(f"Processing query: {query}")
        
        # Create validated user message
        user_message = Message.user(query)
        
        # Initialize conversation with validated user message
        messages = [user_message]
        
        # Collect tools from all connected servers and add server management tools
        all_tools = self.server_manager.collect_all_tools() + self.server_management_tools
        
        # Track conversation metrics
        Traceloop.set_association_properties({
            "query_length": len(query),
            "tool_count": len(all_tools)
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
        logger.info(f"Processing query with history: {query}")
        
        # Create validated user message
        user_message = Message.user(query)
        
        # Add user message to history
        messages = history + [user_message]
        
        # Collect tools from all connected servers and add server management tools
        all_tools = self.server_manager.collect_all_tools() + self.server_management_tools
        
        # Track conversation metrics
        Traceloop.set_association_properties({
            "query_length": len(query),
            "history_length": len(history),
            "tool_count": len(all_tools)
        })
        
        # Run the conversation
        return await self.message_processor.run_conversation(
            messages, 
            all_tools,
            self.response_processor
        )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools including server management tools
        
        Returns:
            List of tools in OpenAI format
        """
        return self.server_manager.collect_all_tools() + self.server_management_tools
    
    def get_available_tools_as_models(self) -> List[ServerToolInfo]:
        """
        Get all available tools as Pydantic models
        
        Returns:
            List of tool info models
        """
        return self.server_manager.collect_all_tools_as_models()
    
    async def get_available_servers(self) -> List[ServerInfo]:
        """
        Get list of available servers as Pydantic models
        
        Returns:
            List of server info models
        """
        return await self.server_manager.get_available_servers_as_models()
    
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