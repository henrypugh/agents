"""
Agent module for orchestrating MCP interactions and LLM processing.

This module provides the central coordinating component that handles connections 
to multiple MCP servers and interacts with LLMs.
"""

import logging
from typing import Dict, List, Optional, Any
import hashlib
import uuid

from traceloop.sdk.decorators import workflow, task
from traceloop.sdk import Traceloop

from src.utils.decorators import message_processing, server_connection, resource_cleanup
from .server_registry import ServerRegistry
from src.client.conversation import Conversation
from .tool_processor import ToolExecutor
from .llm_service import LLMService

logger = logging.getLogger("Agent")

class Agent:
    """Client for interacting with MCP servers and LLMs"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the MCP client
        
        Args:
            model: LLM model to use
        """
        # Generate a unique ID for this client instance
        client_id = str(uuid.uuid4())
        
        # Initialize components
        self.llm_client = LLMService(model)
        self.server_manager = ServerRegistry()
        self.tool_processor = ToolExecutor(self.server_manager)
        self.conversation_manager = Conversation(
            self.llm_client,
            self.server_manager,
            self.tool_processor
        )
        
        # Set global association properties for this client instance
        Traceloop.set_association_properties({
            "client_id": client_id,
            "model": model
        })
        
        logger.info("Agent initialized with model: %s", model)
        
    @server_connection()
    async def connect_to_server(self, server_script_path: str) -> str:
        """
        Connect to an MCP server using a script path
        
        Args:
            server_script_path: Path to the server script (.py or .js)
            
        Returns:
            Name of the connected server
            
        Raises:
            ValueError: If the script is invalid
        """
        logger.info(f"Connecting to server using script path: {server_script_path}")
        
        try:
            # Delegate to server manager
            server_name = await self.server_manager.connect_to_server(server_script_path)
            
            logger.info(f"Successfully connected to server: {server_name}")
            return server_name
            
        except Exception as e:
            logger.error(f"Failed to connect to server: {str(e)}")
            raise
            
    @server_connection()
    async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
        """
        Connect to an MCP server defined in configuration
        
        Args:
            server_name: Name of the server in the config file
            
        Returns:
            Dictionary with connection status and details
            
        Raises:
            ValueError: If the server configuration is invalid
        """
        logger.info(f"Connecting to configured server: {server_name}")
        
        try:
            # Delegate to server manager
            result = await self.server_manager.connect_to_configured_server(server_name)
            
            logger.info(f"Connection to {server_name} status: {result.get('status', 'unknown')}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to connect to configured server {server_name}: {str(e)}")
            raise

    @message_processing()
    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM and available tools
        
        Args:
            query: User query text
            
        Returns:
            Generated response
        """
        logger.info(f"Processing user query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        try:
            # Delegate to conversation manager
            response = await self.conversation_manager.process_query(query)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise
    
    @workflow(name="chat_loop")
    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop
        
        The LLM will have access to server management tools and can connect
        to servers as needed during the conversation.
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
            
            # Track available servers
            Traceloop.set_association_properties({
                "available_servers_count": len(available_servers),
                "available_servers": ",".join(available_servers)
            })
        else:
            print("No configured servers found.")
            
            # Track no servers available
            Traceloop.set_association_properties({
                "available_servers_count": 0
            })
            
        print("\nType your queries or 'quit' to exit.")
        
        interaction_count = 0
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    logger.info("Exiting chat loop")
                    
                    # Track session end
                    Traceloop.set_association_properties({
                        "session_end_reason": "user_exit",
                        "total_interactions": interaction_count
                    })
                    
                    break
                
                # Track this interaction
                interaction_count += 1
                Traceloop.set_association_properties({
                    "interaction_number": interaction_count,
                    "query_length": len(query)
                })
                    
                response = await self.process_query(query)
                print("\n" + response)
                
                # Show current connections status after each query
                connected_servers = self.server_manager.get_connected_servers()
                if connected_servers:
                    print(f"\n[Currently connected to: {', '.join(connected_servers.keys())}]")
                    
                    # Track connected servers
                    Traceloop.set_association_properties({
                        "connected_servers_count": len(connected_servers),
                        "connected_servers": ",".join(connected_servers.keys())
                    })
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")
                
                # Track error
                Traceloop.set_association_properties({
                    "error": "chat_loop_error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
    
    @task(name="get_connected_servers")
    def get_connected_servers(self) -> Dict[str, List[str]]:
        """
        Get information about connected servers and their tools
        
        Returns:
            Dictionary mapping server names to lists of tool names
        """
        return self.server_manager.get_connected_servers()
        
    @task(name="get_primary_server")
    def get_primary_server(self) -> Optional[str]:
        """
        Get the name of the primary server if any
        
        Returns:
            Name of the primary server or None if no servers are connected
        """
        return self.server_manager.get_primary_server()
        
    @task(name="get_available_configured_servers")
    async def get_available_configured_servers(self) -> List[str]:
        """
        Get list of available configured servers
        
        Returns:
            List of server names from configuration
        """
        return await self.server_manager.get_available_configured_servers()
    
    @resource_cleanup()
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.server_manager.cleanup()