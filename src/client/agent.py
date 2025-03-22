"""
Agent module for orchestrating MCP interactions and LLM processing.

This module provides a Pydantic-integrated Agent to coordinate conversation,
server connections, and tool execution.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Sequence, Type, TypeVar, Generic, Protocol, cast
import uuid
import time
from enum import Enum, auto

from traceloop.sdk.decorators import workflow, task
from traceloop.sdk import Traceloop
from pydantic import TypeAdapter, ValidationError

from src.utils.decorators import message_processing, server_connection, resource_cleanup
from src.utils.schemas import (
    AgentConfig, 
    ServerInfo, 
    Message, 
    MessageRole,
    ConnectResponse,
    ConnectResponseStatus,
    LLMResponse,
    OpenAIAdapter,
    MessageHistory
)

import os 
from src.client.server_registry import ServerRegistry, ServerRegistryConfig, ConnectionSettings, ServerConnectionError
from src.client.conversation import Conversation
from src.client.conversation.message_processor import MessageProcessorConfig
from src.client.tool_processor import ToolExecutor, ToolExecutionConfig
from src.client.llm_service import LLMService, RetrySettings


"""
Agent module for orchestrating MCP interactions and LLM processing.

This module provides a Pydantic-integrated Agent to coordinate conversation,
server connections, tool execution, and workflows.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union, Sequence, Type, TypeVar, Generic, Protocol, cast
import uuid
import time
from enum import Enum, auto

from traceloop.sdk.decorators import workflow, task
from traceloop.sdk import Traceloop
from pydantic import TypeAdapter, ValidationError

from src.utils.decorators import message_processing, server_connection, resource_cleanup
from src.utils.workflow import Workflow, WorkflowResult




logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing modes for the agent"""
    STANDARD = auto()
    STREAMING = auto()
    ASYNC = auto()


class AgentSettings:
    """Settings for the Agent"""
    def __init__(
        self,
        agent_id: Optional[str] = None,
        processing_mode: ProcessingMode = ProcessingMode.STANDARD,
        trace_queries: bool = True,
        max_history_length: int = 100,
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.processing_mode = processing_mode
        self.trace_queries = trace_queries
        self.max_history_length = max_history_length


class Agent:
    """
    Agent for interacting with MCP servers and LLMs using Pydantic validation.
    
    This class orchestrates:
    - Server connection discovery and management
    - Conversation processing
    - Tool execution coordination
    - Resource lifecycle management
    """
    
    def __init__(
            self, 
            model: str = "google/gemini-2.0-flash-001",
            agent_settings: Optional[AgentSettings] = None,
            server_config: Optional[ServerRegistryConfig] = None,
            llm_retry_settings: Optional[RetrySettings] = None,
            tool_config: Optional[ToolExecutionConfig] = None
        ):
            """
            Initialize the MCP client agent
            
            Args:
                model: LLM model identifier to use
                agent_settings: Optional settings for the agent
                server_config: Optional configuration for server registry
                llm_retry_settings: Optional retry settings for LLM calls
                tool_config: Optional configuration for tool execution
            """
            # Use provided settings or create defaults
            self.settings = agent_settings or AgentSettings()
            
            # Initialize components with appropriate configurations
            self.llm_client = LLMService(model, retry_settings=llm_retry_settings)
            self.server_manager = ServerRegistry(config=server_config)
            self.tool_processor = ToolExecutor(self.server_manager, config=tool_config)
            
            # Create message processor configuration
            message_processor_config = MessageProcessorConfig(
                max_history_length=self.settings.max_history_length,
                validate_messages=True,
                trace_content=self.settings.trace_queries
            )
            
            # Initialize conversation manager with configuration
            self.conversation_manager = Conversation(
                self.llm_client,
                self.server_manager,
                self.tool_processor,
                message_processor_config
            )
            
            # Create configuration model with validation
            self.config = AgentConfig(model=model)
            
            # Initialize TypeAdapter for ServerInfo validation
            self.server_info_validator = TypeAdapter(ServerInfo)
            self.connect_response_validator = TypeAdapter(ConnectResponse)
            
            # Initialize workflow registry
            self.workflows: Dict[str, Type[Workflow]] = {}
            
            # Set global association properties for this agent instance
            Traceloop.set_association_properties({
                "agent_id": self.settings.agent_id,
                "model": model,
                "max_history_length": self.settings.max_history_length,
                "processing_mode": self.settings.processing_mode.name
            })
            
            logger.info(f"Agent initialized with model: {model}, id: {self.settings.agent_id}")

    def register_workflow(self, workflow_id: str, workflow_cls: Type[Workflow]) -> None:
        """
        Register a workflow class with the agent.
        
        Args:
            workflow_id: ID to use for the workflow
            workflow_cls: Workflow class to register
        """
        self.workflows[workflow_id] = workflow_cls
        logger.info(f"Registered workflow: {workflow_id}")
    
    @task(name="run_workflow")
    async def run_workflow(self, workflow_id: str, **input_data) -> Any:
        """
        Run a registered workflow.
        
        Args:
            workflow_id: ID of the workflow to run
            **input_data: Input data for the workflow
            
        Returns:
            The workflow result
            
        Raises:
            ValueError: If the workflow is not registered
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not registered")
            
        # Track workflow execution
        Traceloop.set_association_properties({
            "workflow_id": workflow_id,
            "workflow_input_keys": list(input_data.keys())
        })
        
        workflow_cls = self.workflows[workflow_id]
        workflow = workflow_cls(self)
        
        # Run the workflow and await the result
        handler = workflow.run(**input_data)
        result = await handler
        
        return result
    
    
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
        # Track config update
        Traceloop.set_association_properties({
            "config_update": {k: v for k, v in kwargs.items() if k != "api_key"}
        })
        
        # Create a model with the current config
        current_dict = self.config.model_dump()
        
        # Update with new values
        current_dict.update(kwargs)
        
        try:
            # Create and validate new config
            self.config = AgentConfig.model_validate(current_dict)
            
            # Update LLM model if it was changed
            if 'model' in kwargs:
                self.llm_client.model = self.config.model
                
            return self.config
        except ValidationError as e:
            logger.error(f"Invalid configuration: {e}")
            # Return original config on error
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
            ServerConnectionError: If script path is invalid or connection fails
        """
        logger.info(f"Connecting to server using script path: {server_script_path}")
        
        # Track connection attempt
        Traceloop.set_association_properties({
            "connection_type": "script",
            "script_path": server_script_path
        })
        
        # Verify script path exists
        if not os.path.exists(server_script_path):
            error = f"Server script not found: {server_script_path}"
            logger.error(error)
            raise ServerConnectionError("unknown", error)
        
        try:
            server_name = await self.server_manager.connect_to_server(server_script_path)
            
            # Track successful connection
            Traceloop.set_association_properties({
                "connection_status": "success",
                "server_name": server_name
            })
            
            return server_name
        except Exception as e:
            # Track connection failure
            Traceloop.set_association_properties({
                "connection_status": "error",
                "error": str(e)
            })
            raise
            
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
        
        # Track connection attempt
        Traceloop.set_association_properties({
            "connection_type": "configured",
            "server_name": server_name
        })
        
        try:
            # Connect to the server
            result = await self.server_manager.connect_to_configured_server(server_name)
            
            # Convert dictionary result to Pydantic model
            try:
                if result.get("status") == "already_connected":
                    status = ConnectResponseStatus.ALREADY_CONNECTED
                elif result.get("status") == "connected":
                    status = ConnectResponseStatus.CONNECTED
                else:
                    status = ConnectResponseStatus.ERROR
                    
                response = ConnectResponse(
                    status=status,
                    server=result.get("server", server_name),
                    tools=result.get("tools"),
                    error=result.get("error"),
                    tool_count=result.get("tool_count")
                )
                
                # Track successful connection
                Traceloop.set_association_properties({
                    "connection_status": status,
                    "tool_count": response.tool_count or 0
                })
                
                return response
            except ValidationError as e:
                logger.error(f"Error creating ConnectResponse: {e}")
                # Return error response on validation failure
                return ConnectResponse(
                    status=ConnectResponseStatus.ERROR,
                    server=server_name,
                    error=f"Validation error: {str(e)}"
                )
        except Exception as e:
            logger.error(f"Error connecting to server {server_name}: {e}")
            
            # Track connection failure
            Traceloop.set_association_properties({
                "connection_status": "error",
                "error": str(e)
            })
            
            # Return error response
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
        # Track query processing
        if self.settings.trace_queries:
            query_preview = query[:50] + "..." if len(query) > 50 else query
            Traceloop.set_association_properties({
                "query_preview": query_preview,
                "query_length": len(query)
            })
            
        logger.info(f"Processing user query: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        try:
            response = await self.conversation_manager.process_query(query)
            
            # Track response metrics
            Traceloop.set_association_properties({
                "response_length": len(response),
                "processing_status": "success"
            })
            
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Track error
            Traceloop.set_association_properties({
                "processing_status": "error",
                "error": str(e)
            })
            
            # Return error message
            return f"Error processing your query: {str(e)}"
    
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
        # Track query processing with history
        if self.settings.trace_queries:
            query_preview = query[:50] + "..." if len(query) > 50 else query
            Traceloop.set_association_properties({
                "query_preview": query_preview,
                "query_length": len(query),
                "history_length": len(history)
            })
            
        logger.info(f"Processing query with history: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        try:
            # Validate history if not empty
            if history and self.settings.max_history_length:
                # Trim history if needed
                if len(history) > self.settings.max_history_length:
                    logger.info(f"Trimming history from {len(history)} to {self.settings.max_history_length} messages")
                    # Keep system messages and most recent messages
                    system_messages = [m for m in history if m.role == MessageRole.SYSTEM]
                    non_system = [m for m in history if m.role != MessageRole.SYSTEM]
                    recent_messages = non_system[-(self.settings.max_history_length - len(system_messages)):]
                    history = system_messages + recent_messages
            
            response = await self.conversation_manager.process_query_with_history(query, history)
            
            # Track response metrics
            Traceloop.set_association_properties({
                "response_length": len(response),
                "processing_status": "success"
            })
            
            return response
        except Exception as e:
            logger.error(f"Error processing query with history: {e}")
            
            # Track error
            Traceloop.set_association_properties({
                "processing_status": "error",
                "error": str(e)
            })
            
            # Return error message
            return f"Error processing your query: {str(e)}"
    
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
        
        # Track processing with system prompt
        if self.settings.trace_queries:
            Traceloop.set_association_properties({
                "has_system_prompt": True,
                "system_prompt_length": len(system_prompt),
                "query_length": len(user_query)
            })
        
        try:
            # Create system and user messages
            system_message = Message.system(system_prompt)
            user_message = Message.user(user_query)
            
            # Create history with system prompt
            history = [system_message]
            
            # Use existing method for processing
            response = await self.conversation_manager.process_query_with_history(user_query, history)
            
            # Track response metrics
            Traceloop.set_association_properties({
                "response_length": len(response),
                "processing_status": "success"
            })
            
            return response
        except Exception as e:
            logger.error(f"Error processing system prompt and query: {e}")
            
            # Track error
            Traceloop.set_association_properties({
                "processing_status": "error",
                "error": str(e)
            })
            
            # Return error message
            return f"Error processing your query: {str(e)}"
    
    @workflow(name="chat_loop")
    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop
        
        This method starts an interactive chat session, allowing the user
        to communicate with the agent via the console.
        """
        # Generate a session ID for tracing
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Set association properties for this chat session
        Traceloop.set_association_properties({
            "session_id": session_id,
            "session_type": "interactive",
            "start_time": start_time
        })
        
        logger.info("Starting chat loop")
        print("\nMCP Client Started with dynamic server connection!")
        
        try:
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
                    
                    # Skip empty queries
                    if not query:
                        continue
                    
                    # Track this interaction
                    interaction_count += 1
                    interaction_start = time.time()
                    
                    Traceloop.set_association_properties({
                        "interaction_number": interaction_count,
                        "query_length": len(query),
                        "interaction_start": interaction_start
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
                    
                    # Track interaction completion
                    interaction_duration = time.time() - interaction_start
                    Traceloop.set_association_properties({
                        "interaction_duration": interaction_duration,
                        "response_length": len(response)
                    })
                    
                    # Display response
                    print("\n" + response)
                    
                    # Create and add assistant message to history
                    assistant_message = Message.assistant(content=response)
                    conversation_history.append(assistant_message)
                    
                    # Trim history if it gets too long
                    if len(conversation_history) > self.settings.max_history_length:
                        # Keep system messages and most recent messages
                        system_messages = [m for m in conversation_history if m.role == MessageRole.SYSTEM]
                        non_system = [m for m in conversation_history if m.role != MessageRole.SYSTEM]
                        history_limit = self.settings.max_history_length - len(system_messages)
                        recent_messages = non_system[-history_limit:]
                        conversation_history = system_messages + recent_messages
                    
                    # Show current connections status after each query
                    connected_servers = list(self.server_manager.servers.keys())
                    if connected_servers:
                        print(f"\n[Currently connected to: {', '.join(connected_servers)}]")
                        
                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'quit' to exit.")
                    
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}", exc_info=True)
                    print(f"\nError: {str(e)}")
                    
                    # Track error
                    Traceloop.set_association_properties({
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                    
        except KeyboardInterrupt:
            print("\nExiting chat loop")
            
        finally:
            # Track session completion
            session_duration = time.time() - start_time
            Traceloop.set_association_properties({
                "session_duration": session_duration,
                "total_interactions": interaction_count,
                "end_time": time.time()
            })
            
            logger.info(f"Chat loop completed after {session_duration:.2f}s with {interaction_count} interactions")
    
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
        cleanup_start = time.time()
        
        # Track cleanup start
        Traceloop.set_association_properties({
            "cleanup_start": cleanup_start
        })
        
        try:
            await self.server_manager.cleanup()
            
            # Track successful cleanup
            cleanup_duration = time.time() - cleanup_start
            Traceloop.set_association_properties({
                "cleanup_duration": cleanup_duration,
                "cleanup_status": "success"
            })
            
            logger.info(f"Cleanup completed in {cleanup_duration:.2f}s")
            
        except Exception as e:
            # Track cleanup error
            cleanup_duration = time.time() - cleanup_start
            Traceloop.set_association_properties({
                "cleanup_duration": cleanup_duration,
                "cleanup_status": "error",
                "cleanup_error": str(e)
            })
            
            logger.error(f"Error during cleanup: {e}")