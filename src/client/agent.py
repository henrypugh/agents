"""
Agent module for orchestrating MCP interactions and LLM processing.

This module provides a Pydantic-integrated Agent to coordinate conversation,
server connections, and tool execution.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Sequence
import uuid

from traceloop.sdk.decorators import workflow, task
from traceloop.sdk import Traceloop

from src.utils.decorators import message_processing, server_connection, resource_cleanup
from src.utils.schemas import (
    AgentConfig, 
    ServerInfo, 
    Message, 
    MessageRole,
    ConnectResponse,
    ConnectResponseStatus,
    LLMResponse
)
from src.client.server_registry import ServerRegistry
from src.client.conversation import Conversation
from src.client.tool_processor import ToolExecutor
from src.client.llm_service import LLMService

logger = logging.getLogger(__name__)

class Agent:
    """
    Agent for interacting with MCP servers and LLMs using Pydantic validation.
    
    This class orchestrates:
    - Server connection discovery and management
    - Conversation processing
    - Tool execution coordination
    - Resource lifecycle management
    """
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the MCP client agent
        
        Args:
            model: LLM model identifier to use
        """
        # Generate a unique ID for this agent instance
        agent_id = str(uuid.uuid4())
        
        # Initialize components
        self.llm_client = LLMService(model)
        self.server_manager = ServerRegistry()
        self.tool_processor = ToolExecutor(self.server_manager)
        self.conversation_manager = Conversation(
            self.llm_client,
            self.server_manager,
            self.tool_processor
        )
        
        # Create configuration
        self.config = AgentConfig(model=model)
        
        # Set global association properties for this agent instance
        Traceloop.set_association_properties({
            "agent_id": agent_id,
            "model": model
        })
        
        logger.info(f"Agent initialized with model: {model}")
    
    def get_config(self) -> AgentConfig:
        """
        Get agent configuration as a Pydantic model
        
        Returns:
            Agent configuration model
        """
        return self.config
    
    def update_config(self, **kwargs) -> AgentConfig:
        """
        Update agent configuration
        
        Args:
            **kwargs: Configuration parameters to update
            
        Returns:
            Updated agent configuration
        """
        # Create a model with the current config
        current_dict = self.config.model_dump()
        
        # Update with new values
        current_dict.update(kwargs)
        
        # Create and validate new config
        self.config = AgentConfig.model_validate(current_dict)
        
        # Update LLM model if it was changed
        if 'model' in kwargs:
            self.llm_client.model = self.config.model
            
        return self.config
        
    @server_connection()
    async def connect_to_server(self, server_script_path: str) -> str:
        """
        Connect to an MCP server using a script path
        
        Args:
            server_script_path: Path to server script
            
        Returns:
            Server name
            
        Raises:
            ValueError: If script path is invalid
        """
        logger.info(f"Connecting to server using script path: {server_script_path}")
        return await self.server_manager.connect_to_server(server_script_path)
            
    @server_connection()
    async def connect_to_configured_server(self, server_name: str) -> ConnectResponse:
        """
        Connect to an MCP server defined in configuration
        
        Args:
            server_name: Name of the server to connect to
            
        Returns:
            Connection response model
        """
        logger.info(f"Connecting to configured server: {server_name}")
        result = await self.server_manager.connect_to_configured_server(server_name)
        
        # Convert dictionary result to Pydantic model
        try:
            if result.get("status") == "already_connected":
                status = ConnectResponseStatus.ALREADY_CONNECTED
            elif result.get("status") == "connected":
                status = ConnectResponseStatus.CONNECTED
            else:
                status = ConnectResponseStatus.ERROR
                
            return ConnectResponse(
                status=status,
                server=result.get("server", server_name),
                tools=result.get("tools"),
                error=result.get("error"),
                tool_count=result.get("tool_count")
            )
        except Exception as e:
            logger.error(f"Error creating ConnectResponse: {e}")
            # Return error response on failure
            return ConnectResponse(
                status=ConnectResponseStatus.ERROR,
                server=server_name,
                error=str(e)
            )

    @message_processing()
    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM and available tools
        
        Args:
            query: User query text
            
        Returns:
            Generated response text
        """
        logger.info(f"Processing user query: {query[:50]}{'...' if len(query) > 50 else ''}")
        return await self.conversation_manager.process_query(query)
    
    @message_processing()
    async def process_query_with_history(
        self, 
        query: str, 
        history: List[Message]
    ) -> str:
        """
        Process a query in the context of previous conversation history
        
        Args:
            query: User query text
            history: Previous conversation history
            
        Returns:
            Generated response text
        """
        logger.info(f"Processing query with history: {query[:50]}{'...' if len(query) > 50 else ''}")
        return await self.conversation_manager.process_query_with_history(query, history)
    
    @message_processing()
    async def process_system_prompt(
        self,
        system_prompt: str,
        user_query: str
    ) -> str:
        """
        Process a query with a system prompt to guide the response
        
        Args:
            system_prompt: System instructions
            user_query: User query text
            
        Returns:
            Generated response text
        """
        logger.info(f"Processing system prompt and query")
        
        # Create system and user messages
        system_message = Message.system(system_prompt)
        user_message = Message.user(user_query)
        
        # Create history with system prompt
        history = [system_message]
        
        # Use existing method for processing
        return await self.conversation_manager.process_query_with_history(user_query, history)
    
    @workflow(name="chat_loop")
    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop
        
        This method starts an interactive chat session, allowing the user
        to communicate with the agent via the console.
        """
        # Generate a session ID for tracing
        session_id = str(uuid.uuid4())
        
        # Set association properties for this chat session
        Traceloop.set_association_properties({
            "session_id": session_id,
            "session_type": "interactive"
        })
        
        logger.info("Starting chat loop")
        print("\nMCP Client Started with dynamic server connection!")
        
        # Show available servers
        available_servers = await self.server_manager.get_available_configured_servers()
        if available_servers:
            print(f"Available servers: {', '.join(available_servers)}")
            print("The assistant can connect to these servers when needed.")
        else:
            print("No configured servers found.")
            
        print("\nType your queries or 'quit' to exit.")
        
        # Initialize conversation history
        conversation_history: List[Message] = []
        interaction_count = 0
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    logger.info("Exiting chat loop")
                    break
                
                # Track this interaction
                interaction_count += 1
                Traceloop.set_association_properties({
                    "interaction_number": interaction_count,
                    "query_length": len(query)
                })
                
                # Create user message
                user_message = Message.user(query)
                
                # Add to history
                if conversation_history:
                    # Process with existing history
                    conversation_history.append(user_message)
                    response = await self.conversation_manager.process_query_with_history(
                        query, conversation_history
                    )
                else:
                    # Start new conversation
                    conversation_history = [user_message]
                    response = await self.process_query(query)
                
                # Display response
                print("\n" + response)
                
                # Create and add assistant message to history
                assistant_message = Message.assistant(content=response)
                conversation_history.append(assistant_message)
                
                # Show current connections status after each query
                connected_servers = list(self.server_manager.servers.keys())
                if connected_servers:
                    print(f"\n[Currently connected to: {', '.join(connected_servers)}]")
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"\nError: {str(e)}")
    
    @task(name="get_connected_servers")
    def get_connected_servers(self) -> List[ServerInfo]:
        """
        Get information about connected servers
        
        Returns:
            List of ServerInfo models
        """
        return self.server_manager.get_connected_servers_as_models()
        
    @task(name="get_primary_server")
    def get_primary_server(self) -> Optional[str]:
        """
        Get the name of the primary server if any
        
        Returns:
            Server name or None
        """
        return self.server_manager.get_primary_server()
        
    @task(name="get_available_configured_servers")
    async def get_available_configured_servers(self) -> List[str]:
        """
        Get list of available configured servers
        
        Returns:
            List of server names
        """
        return await self.server_manager.get_available_configured_servers()
    
    async def get_available_servers(self) -> List[ServerInfo]:
        """
        Get available servers as Pydantic models
        
        Returns:
            List of ServerInfo models
        """
        return await self.server_manager.get_available_servers_as_models()
    
    def create_system_message(self, content: str) -> Message:
        """
        Create a system message
        
        Args:
            content: System message content
            
        Returns:
            System message model
        """
        return Message.system(content)
    
    def create_user_message(self, content: str) -> Message:
        """
        Create a user message
        
        Args:
            content: User message content
            
        Returns:
            User message model
        """
        return Message.user(content)
    
    @resource_cleanup
    async def cleanup(self) -> None:
        """
        Clean up resources
        
        This method ensures all server connections and resources
        are properly closed when the agent is no longer needed.
        """
        logger.info("Cleaning up agent resources")
        await self.server_manager.cleanup()