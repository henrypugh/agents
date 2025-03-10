"""
Refactored MCP Client implementation with improved structure and organization.
This module handles connections to multiple MCP servers and interacts with an LLM.
"""

import logging
from typing import Dict, List, Optional, Any

from .server_manager import ServerManager
from .conversation_manager import ConversationManager
from .tool_processor import ToolProcessor
from .llm_client import LLMClient

logger = logging.getLogger("MCPClient")

class MCPClient:
    """Client for interacting with MCP servers and LLMs"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the MCP client
        
        Args:
            model: LLM model to use
        """
        # Initialize components
        self.llm_client = LLMClient(model)
        self.server_manager = ServerManager()
        self.tool_processor = ToolProcessor(self.server_manager)
        self.conversation_manager = ConversationManager(
            self.llm_client,
            self.server_manager,
            self.tool_processor
        )
        logger.info("MCPClient initialized with model: %s", model)
        
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
        return await self.server_manager.connect_to_server(server_script_path)

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
        return await self.server_manager.connect_to_configured_server(server_name)

    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM and available tools
        
        The LLM will have access to server management tools and can connect
        to additional servers as needed during the conversation.
        
        Args:
            query: User query text
            
        Returns:
            Generated response
        """
        return await self.conversation_manager.process_query(query)
    
    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop
        
        The LLM will have access to server management tools and can connect
        to servers as needed during the conversation.
        """
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
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    logger.info("Exiting chat loop")
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                
                # Show current connections status after each query
                connected_servers = self.server_manager.get_connected_servers()
                if connected_servers:
                    print(f"\n[Currently connected to: {', '.join(connected_servers.keys())}]")
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")
    
    def get_connected_servers(self) -> Dict[str, List[str]]:
        """
        Get information about connected servers and their tools
        
        Returns:
            Dictionary mapping server names to lists of tool names
        """
        return self.server_manager.get_connected_servers()
        
    def get_primary_server(self) -> Optional[str]:
        """
        Get the name of the primary server if any
        
        Returns:
            Name of the primary server or None if no servers are connected
        """
        return self.server_manager.get_primary_server()
        
    async def get_available_configured_servers(self) -> List[str]:
        """
        Get list of available configured servers
        
        Returns:
            List of server names from configuration
        """
        return await self.server_manager.get_available_configured_servers()
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.server_manager.cleanup()