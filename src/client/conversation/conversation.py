"""
Main conversation interface for handling LLM interactions.
"""

import logging
from typing import Dict, List, Any

from traceloop.sdk import Traceloop

from src.utils.decorators import message_processing
from .message_processor import MessageProcessor
from .response_processor import ResponseProcessor
from .server_management import ServerManagementHandler
from src.client.llm_service import LLMService
from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor

logger = logging.getLogger("Conversation")

class Conversation:
    """Manages conversations with LLMs and tool execution"""
    
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
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
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