"""
Refactored MCP Client implementation with improved structure and organization.
This module handles connections to multiple MCP servers and interacts with an LLM.
"""

import logging
import json
import os
import shutil
from typing import Dict, List, Optional, Any, Tuple
from contextlib import AsyncExitStack
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tool_manager import ToolManager  # Keep this import for backward compatibility
from .llm_client import LLMClient

logger = logging.getLogger("MCPClient")

from .server_config import ServerConfigManager
from .server_connection import ServerConnection


class MCPClient:
    """Client for interacting with MCP servers and LLMs"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the MCP client
        
        Args:
            model: LLM model to use
        """
        self.servers = {}  # Dictionary to store server connections
        self.exit_stack = AsyncExitStack()
        self.llm_client = LLMClient(model)
        self.config_manager = ServerConfigManager()
        # Define LLM tools for server management
        self.server_management_tools = self._create_server_management_tools()
        logger.info("MCPClient initialized")
        
    def _create_server_management_tools(self) -> List[Dict[str, Any]]:
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
        logger.info(f"Connecting to server with script: {server_script_path}")
        
        # Validate script file extension and get command
        command, server_params = self._create_script_server_params(server_script_path)
        
        # Generate server name from script path
        server_name = os.path.basename(server_script_path).split('.')[0]
        
        # Connect and create server connection
        session = await self._create_server_session(server_params)
        server = ServerConnection(server_name, session)
        await server.initialize()
        
        # Store server connection
        self.servers[server_name] = server
        
        logger.info(f"Connected to server {server_name} with tools: {server.get_tool_names()}")
        return server_name

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
        
        # Check if already connected
        if server_name in self.servers:
            logger.info(f"Server {server_name} is already connected")
            tools = self.servers[server_name].get_tool_names()
            return {
                "status": "already_connected",
                "server": server_name,
                "tools": tools,
                "tool_count": len(tools)
            }
        
        try:
            # Get server configuration
            server_config = self.config_manager.get_server_config(server_name)
            
            # Create server parameters
            server_params = self._create_config_server_params(server_name, server_config)
            
            # Connect and create server connection
            session = await self._create_server_session(server_params)
            server = ServerConnection(server_name, session)
            await server.initialize()
            
            # Store server connection
            self.servers[server_name] = server
            
            # Get tool names for response
            tool_names = server.get_tool_names()
            logger.info(f"Connected to server {server_name} with tools: {tool_names}")
            
            return {
                "status": "connected",
                "server": server_name,
                "tools": tool_names,
                "tool_count": len(tool_names)
            }
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to configured server {server_name}: {error_msg}", exc_info=True)
            return {
                "status": "error",
                "server": server_name,
                "error": error_msg
            }

    async def _create_server_session(self, server_params: StdioServerParameters) -> ClientSession:
        """
        Create and initialize a server session
        
        Args:
            server_params: Parameters for the server
            
        Returns:
            Initialized client session
        """
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session
        
    def _create_script_server_params(self, script_path: str) -> Tuple[str, StdioServerParameters]:
        """
        Create server parameters for a script path
        
        Args:
            script_path: Path to the server script
            
        Returns:
            Tuple of (command, server_params)
            
        Raises:
            ValueError: If the script has an invalid extension
        """
        # Validate script file extension
        is_python = script_path.endswith('.py')
        is_js = script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        # Set up server parameters
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None
        )
        
        return command, server_params
        
    def _create_config_server_params(self, server_name: str, server_config: Dict[str, Any]) -> StdioServerParameters:
        """
        Create server parameters from configuration
        
        Args:
            server_name: Name of the server
            server_config: Server configuration dictionary
            
        Returns:
            Server parameters
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Validate required fields
        if 'command' not in server_config:
            raise ValueError(f"Server '{server_name}' configuration missing 'command' field")
            
        # Find command in PATH
        command = server_config['command']
        command_path = shutil.which(command)
        if not command_path:
            raise ValueError(f"Command '{command}' not found in PATH")
            
        # Set up server parameters
        args = server_config.get('args', [])
        env = server_config.get('env', {}).copy() if server_config.get('env') else {}
        
        # Process environment variables
        processed_env = self.config_manager.process_environment_variables(env)
        
        # Log API key status if this is the Brave Search server
        if server_name == "brave-search" and 'BRAVE_API_KEY' in processed_env:
            key_preview = processed_env['BRAVE_API_KEY'][:4] + "..." if processed_env['BRAVE_API_KEY'] else "None"
            logger.info(f"[API KEY STATUS] Brave API key is set (starts with: {key_preview})")
        
        # Merge with current environment
        env = {**os.environ, **processed_env}
        
        return StdioServerParameters(
            command=command_path,
            args=args,
            env=env
        )

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
        logger.info(f"Processing query: {query}")
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Collect tools from all connected servers and add server management tools
        all_tools = self._collect_all_tools() + self.server_management_tools
        
        # Start conversation
        return await self._run_conversation(messages, all_tools)
    
    def _collect_all_tools(self) -> List[Dict[str, Any]]:
        """
        Collect tools from all connected servers
        
        Returns:
            List of all available tools in OpenAI format
        """
        all_tools = []
        for server in self.servers.values():
            all_tools.extend(server.get_openai_format_tools())
        
        num_servers = len(self.servers)
        num_tools = len(all_tools)
        if num_servers > 0:
            logger.info(f"Collected {num_tools} tools from {num_servers} servers")
        else:
            logger.info("No servers connected yet. Only server management tools will be available.")
            
        return all_tools
        
    async def _run_conversation(
        self, 
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]]
    ) -> str:
        """
        Run a conversation with the LLM using tools
        
        Args:
            messages: Conversation history
            tools: Available tools
            
        Returns:
            Generated response
        """
        # Get initial response from LLM
        response = await self.llm_client.get_completion(messages, tools)
        
        # Process response and handle any tool calls
        final_text = []
        await self._process_response(response, messages, tools, final_text)
        
        # Join all parts with newlines, ensuring there's no empty content
        result = "\n".join(part for part in final_text if part and part.strip())
        return result if result else "No response content received from the LLM."
        
    async def _process_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process an LLM response and handle tool calls
        
        Args:
            response: LLM response object
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        try:
            # Extract message from the response
            choices = response.choices
            if not choices or len(choices) == 0:
                final_text.append("No response generated by the LLM.")
                return
                
            message = choices[0].message
            
            # Add text content to output if present
            if hasattr(message, 'content') and message.content:
                content = message.content.strip()
                if content:  # Only add non-empty content
                    final_text.append(content)
            
            # Check for and process tool calls
            has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
            logger.info(f"LLM tool usage decision: {'Used tools' if has_tool_calls else 'Did NOT use any tools'}")
            
            if has_tool_calls:
                # Process each tool call
                for tool_call in message.tool_calls:
                    await self._process_tool_call(tool_call, messages, tools, final_text)
        
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
            final_text.append(f"Error occurred while processing: {str(e)}")
    
    async def _process_tool_call(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process a single tool call
        
        Args:
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        logger.debug(f"Processing tool call: {tool_call}")
        
        # Get tool details
        tool_name = tool_call.function.name
        
        # Check if this is an internal server management tool
        if tool_name in ["list_available_servers", "connect_to_server", "list_connected_servers"]:
            await self._handle_server_management_tool(tool_call, messages, tools, final_text)
            return
            
        # Otherwise, process as a regular server tool
        server_name = self._find_server_for_tool(tool_name, tools)
        
        if not server_name or server_name not in self.servers:
            error_msg = f"Error: Can't determine which server handles tool '{tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            return
            
        # Get server connection
        server = self.servers[server_name]
        
        # Execute the tool
        await self._execute_and_process_tool(
            server, 
            server_name, 
            tool_call, 
            messages, 
            tools, 
            final_text
        )
    
    def _find_server_for_tool(self, tool_name: str, tools: List[Dict[str, Any]]) -> Optional[str]:
        """
        Find which server handles the specified tool
        
        Args:
            tool_name: Name of the tool
            tools: List of available tools
            
        Returns:
            Name of the server or None if not found
        """
        for tool in tools:
            if tool["function"]["name"] == tool_name:
                if "metadata" in tool["function"]:
                    # Check if it's an internal tool
                    if tool["function"]["metadata"].get("internal"):
                        return "internal"
                    # Otherwise check for server metadata
                    if "server" in tool["function"]["metadata"]:
                        return tool["function"]["metadata"]["server"]
        return None
        
    async def _execute_and_process_tool(
        self,
        server: ServerConnection,
        server_name: str,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Execute a tool and process the result
        
        Args:
            server: Server connection
            server_name: Name of the server
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        tool_name = tool_call.function.name
        tool_args_raw = tool_call.function.arguments
        final_text.append(f"[Calling tool {tool_name} from {server_name} server with args {tool_args_raw}]")
        
        try:
            # Parse arguments
            tool_args = json.loads(tool_args_raw)
            
            # Execute the tool
            result = await server.execute_tool(tool_name, tool_args)
            
            # Process result into text
            result_text = self._extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await self._get_follow_up_response(messages, tools, final_text)
            
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg, exc_info=True)
            final_text.append(error_msg)
            
    def _extract_result_text(self, result: Any) -> str:
        """
        Extract text content from a tool result
        
        Args:
            result: Result from tool execution
            
        Returns:
            Extracted text content
        """
        if not hasattr(result, "content"):
            return str(result)
            
        if isinstance(result.content, list):
            result_text = ""
            for item in result.content:
                if hasattr(item, "text"):
                    result_text += item.text
                else:
                    result_text += str(item)
            return result_text
        elif isinstance(result.content, str):
            return result.content
        else:
            return str(result.content)
    
    def _update_message_history(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result
        
        Args:
            messages: Conversation history to update
            tool_call: Tool call from LLM
            result_text: Result of tool execution
        """
        # Add assistant message with tool call
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
            ]
        }
        messages.append(tool_call_message)
        
        # Add tool response message
        tool_response_message = {
            "role": "tool", 
            "tool_call_id": tool_call.id,
            "content": result_text
        }
        messages.append(tool_response_message)
        
        # Log updated message history for debugging
        logger.debug(f"Updated message history with tool call and result. Now have {len(messages)} messages.")
    
    async def _get_follow_up_response(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Get and process follow-up response from LLM after tool call
        
        Args:
            messages: Conversation history
            available_tools: Available tools
            final_text: List to append text to
        """
        logger.info("Getting follow-up response with tool results")
        try:
            # Log message count for debugging
            logger.debug(f"Sending {len(messages)} messages for follow-up")
            
            # Get follow-up response from LLM
            follow_up_response = await self.llm_client.get_completion(messages, available_tools)
            
            # Process follow-up response
            if follow_up_response.choices and len(follow_up_response.choices) > 0:
                follow_up_message = follow_up_response.choices[0].message
                
                # Check for content
                if hasattr(follow_up_message, 'content') and follow_up_message.content:
                    content = follow_up_message.content.strip()
                    if content:  # Ensure content is not empty
                        logger.debug(f"Got follow-up content: {content[:100]}...")
                        final_text.append(content)
                
                # Check for nested tool calls
                if hasattr(follow_up_message, 'tool_calls') and follow_up_message.tool_calls:
                    logger.info(f"Found {len(follow_up_message.tool_calls)} nested tool calls in follow-up")
                    # We use individual tool processing to avoid recursion issues
                    for tool_call in follow_up_message.tool_calls:
                        await self._process_tool_call(tool_call, messages, available_tools, final_text)
        except Exception as e:
            logger.error(f"Error in follow-up API call: {str(e)}", exc_info=True)
            final_text.append(f"Error in follow-up response: {str(e)}")

    async def chat_loop(self) -> None:
        """
        Run an interactive chat loop
        
        The LLM will have access to server management tools and can connect
        to servers as needed during the conversation.
        """
        logger.info("Starting chat loop")
        print("\nMCP Client Started with dynamic server connection!")
        
        # Show available servers
        available_servers = await self.get_available_configured_servers()
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
                if self.servers:
                    print(f"\n[Currently connected to: {', '.join(self.servers.keys())}]")
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")
    
    def get_connected_servers(self) -> Dict[str, List[str]]:
        """
        Get information about connected servers and their tools
        
        Returns:
            Dictionary mapping server names to lists of tool names
        """
        server_info = {}
        for name, server in self.servers.items():
            server_info[name] = server.get_tool_names()
        return server_info
        
    def get_primary_server(self) -> Optional[str]:
        """
        Get the name of the primary server if any
        
        Returns:
            Name of the primary server or None if no servers are connected
        """
        if not self.servers:
            return None
        return next(iter(self.servers))
        
    async def get_available_configured_servers(self) -> List[str]:
        """
        Get list of available configured servers
        
        Returns:
            List of server names from configuration
        """
        try:
            config = self.config_manager.load_config()
            return list(config.keys())
        except Exception as e:
            logger.error(f"Error loading server configuration: {e}")
            return []
            
    async def _handle_server_management_tool(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Handle server management tool calls
        
        Args:
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        tool_name = tool_call.function.name
        logger.info(f"Handling server management tool: {tool_name}")
        
        try:
            # Parse arguments if any
            tool_args = {}
            if tool_call.function.arguments:
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    final_text.append(f"Error parsing arguments: {str(e)}")
                    return
            
            # Handle each tool type
            if tool_name == "list_available_servers":
                await self._handle_list_available_servers(tool_call, messages, final_text)
            elif tool_name == "connect_to_server":
                await self._handle_connect_to_server(tool_call, tool_args, messages, final_text)
            elif tool_name == "list_connected_servers":
                await self._handle_list_connected_servers(tool_call, messages, final_text)
            else:
                final_text.append(f"Unknown server management tool: {tool_name}")
        
        except Exception as e:
            logger.error(f"Error handling server management tool: {str(e)}", exc_info=True)
            final_text.append(f"Error: {str(e)}")
            
    async def _handle_list_available_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_available_servers tool"""
        available_servers = await self.get_available_servers()
        
        # Format for JSON response
        result = {
            "available_servers": {},
            "count": len(available_servers)
        }
        
        for server_name, server_info in available_servers.items():
            result["available_servers"][server_name] = server_info
        
        # Format for display
        if available_servers:
            server_list = []
            for server_name, info in available_servers.items():
                server_type = info["type"]
                source = info["source"]
                server_list.append(f"{server_name} ({server_type} from {source})")
            
            result_text = f"Available servers ({len(available_servers)}):\n" + "\n".join(server_list)
        else:
            result_text = "No available servers found"
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))

    async def _handle_connect_to_server(
        self,
        tool_call: Any,
        args: Dict[str, Any],
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle connect_to_server tool"""
        if "server_name" not in args:
            error_msg = "Missing required argument: server_name"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({"error": error_msg}))
            return
            
        server_name = args["server_name"]
        
        # Get available servers
        available_servers = await self.get_available_servers()
        
        if server_name not in available_servers:
            error_msg = f"Server '{server_name}' not found. Available servers: {', '.join(available_servers.keys())}"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({"error": error_msg}))
            return
        
        # Already connected?
        if server_name in self.servers:
            tools = self.servers[server_name].get_tool_names()
            result = {
                "status": "already_connected",
                "server": server_name,
                "tools": tools,
                "tool_count": len(tools)
            }
            result_text = f"Already connected to {server_name}. Available tools: {', '.join(tools)}"
            final_text.append(f"[Server management] {result_text}")
            self._update_message_history(messages, tool_call, json.dumps(result))
            return
        
        # Get server info
        server_info = available_servers[server_name]
        server_type = server_info["type"]
        
        try:
            # Connect based on type
            if server_type == "configured":
                result = await self.connect_to_configured_server(server_name)
            else:  # local_script
                script_path = server_info["source"]
                await self.connect_to_server(script_path)
                tools = self.servers[server_name].get_tool_names() if server_name in self.servers else []
                result = {
                    "status": "connected",
                    "server": server_name,
                    "source": script_path,
                    "tools": tools,
                    "tool_count": len(tools)
                }
            
            # Format for display
            if result["status"] == "connected" or result["status"] == "already_connected":
                tools_str = ", ".join(result["tools"])
                result_text = f"Successfully connected to {server_name}. Available tools: {tools_str}"
            else:
                result_text = f"Failed to connect to {server_name}: {result.get('error', 'Unknown error')}"
            
            final_text.append(f"[Server management] {result_text}")
            
            # Update message history
            self._update_message_history(messages, tool_call, json.dumps(result))
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to connect to server {server_name}: {error_msg}", exc_info=True)
            result = {
                "status": "error",
                "server": server_name,
                "error": error_msg
            }
            final_text.append(f"[Server management] Error connecting to {server_name}: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps(result))
    
    async def get_available_servers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available servers (both configured and local)
        
        Returns:
            Dictionary mapping server names to their source information
        """
        available_servers = {}
        
        # Add configured servers from config file
        try:
            config = self.config_manager.load_config()
            for server_name, server_config in config.items():
                available_servers[server_name] = {
                    "type": "configured",
                    "source": "server_config.json",
                    "command": server_config.get("command", ""),
                    "args": server_config.get("args", [])
                }
        except Exception as e:
            logger.error(f"Error loading server configuration: {e}")
        
        # Look for common local server paths
        common_paths = [
            "server/main.py",
            "src/server/main.py",
            "server/server.py",
            "src/server/server.py"
        ]
        
        for path in common_paths:
            if os.path.isfile(path):
                server_name = os.path.basename(os.path.dirname(path))
                available_servers[server_name] = {
                    "type": "local_script",
                    "source": path,
                    "command": "python",
                    "args": [path]
                }
        
        return available_servers

    async def _handle_list_connected_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_connected_servers tool"""
        connected_servers = self.get_connected_servers()
        
        # Build detailed result
        result = {
            "connected_servers": list(connected_servers.keys()),
            "count": len(connected_servers),
            "details": {}
        }
        
        for server_name, tools in connected_servers.items():
            result["details"][server_name] = {
                "tools": tools,
                "tool_count": len(tools)
            }
        
        # Format for display
        if connected_servers:
            server_details = []
            for server_name, tools in connected_servers.items():
                server_details.append(f"{server_name} ({len(tools)} tools)")
            result_text = f"Connected servers: {', '.join(server_details)}"
        else:
            result_text = "No servers are currently connected"
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.exit_stack.aclose()