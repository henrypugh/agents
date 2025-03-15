"""
Agent module for orchestrating MCP interactions and LLM processing.
"""

import logging
from typing import Dict, List, Any, Optional
import uuid

from traceloop.sdk.decorators import workflow, task
from traceloop.sdk import Traceloop

from src.utils.decorators import message_processing, server_connection, resource_cleanup
from src.utils.schemas import AgentConfig, ServerInfo
from src.client.server_registry import ServerRegistry
from src.client.conversation import Conversation
from src.client.tool_processor import ToolExecutor
from src.client.llm_service import LLMService

logger = logging.getLogger("Agent")

class Agent:
    """Client for interacting with MCP servers and LLMs"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """Initialize the MCP client"""
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
    
    def get_config(self) -> AgentConfig:
        """Get agent configuration as a Pydantic model"""
        try:
            return AgentConfig(model=self.llm_client.model)
        except Exception as e:
            logger.error(f"Error creating agent config: {e}")
            return AgentConfig()
        
    @server_connection()
    async def connect_to_server(self, server_script_path: str) -> str:
        """Connect to an MCP server using a script path"""
        logger.info(f"Connecting to server using script path: {server_script_path}")
        return await self.server_manager.connect_to_server(server_script_path)
            
    @server_connection()
    async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
        """Connect to an MCP server defined in configuration"""
        logger.info(f"Connecting to configured server: {server_name}")
        return await self.server_manager.connect_to_configured_server(server_name)

    @message_processing()
    async def process_query(self, query: str) -> str:
        """Process a user query using the LLM and available tools"""
        logger.info(f"Processing user query: {query[:50]}{'...' if len(query) > 50 else ''}")
        return await self.conversation_manager.process_query(query)
    
    @message_processing()
    async def process_query_with_models(self, query: str) -> str:
        """Process a query using Pydantic models for validation"""
        logger.info(f"Processing query with models: {query[:50]}{'...' if len(query) > 50 else ''}")
        try:
            return await self.conversation_manager.process_query_with_models(query)
        except Exception as e:
            logger.error(f"Error in model-based processing: {e}")
            # Fall back to standard processing
            return await self.process_query(query)
    
    @workflow(name="chat_loop")
    async def chat_loop(self) -> None:
        """Run an interactive chat loop"""
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
                    
                response = await self.process_query(query)
                print("\n" + response)
                
                # Show current connections status after each query
                connected_servers = list(self.server_manager.servers.keys())
                if connected_servers:
                    print(f"\n[Currently connected to: {', '.join(connected_servers)}]")
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"\nError: {str(e)}")
    
    @task(name="get_connected_servers")
    def get_connected_servers(self) -> Dict[str, List[str]]:
        """Get information about connected servers and their tools"""
        connected_servers = {}
        for name, server in self.server_manager.servers.items():
            connected_servers[name] = server.get_tool_names()
        return connected_servers
    
    def get_connected_servers_as_models(self) -> List[ServerInfo]:
        """Get connected servers as ServerInfo models"""
        return self.server_manager.get_connected_servers_as_models()
        
    @task(name="get_primary_server")
    def get_primary_server(self) -> Optional[str]:
        """Get the name of the primary server if any"""
        return self.server_manager.get_primary_server()
        
    @task(name="get_available_configured_servers")
    async def get_available_configured_servers(self) -> List[str]:
        """Get list of available configured servers"""
        return await self.server_manager.get_available_configured_servers()
    
    async def get_available_servers_as_models(self) -> List[ServerInfo]:
        """Get available servers as ServerInfo models"""
        return await self.server_manager.get_available_servers_as_models()
    
    @resource_cleanup()
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.server_manager.cleanup()