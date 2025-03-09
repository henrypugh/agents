import logging
import json
import os
import shutil
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack
from decouple import config

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .tool_manager import ToolManager
from .llm_client import LLMClient

logger = logging.getLogger("MCPClient")

class MCPClient:
    """Client for interacting with MCP servers"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.llm_client = LLMClient(model)
        logger.info("MCPClient initialized")

    async def connect_to_server(self, server_script_path: str) -> None:
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.info(f"Connecting to server with script: {server_script_path}")
        
        # Validate script file extension
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        # Set up server parameters
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # Connect to server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to server with tools: {[tool.name for tool in tools]}")

    async def connect_to_configured_server(self, server_name: str, config_path: str = "server_config.json") -> None:
        """Connect to an MCP server defined in configuration
        
        Args:
            server_name: Name of the server in the config file
            config_path: Path to the config file
        """
        logger.info(f"Connecting to configured server: {server_name}")
        
        # Load server configuration
        try:
            with open(config_path, 'r') as f:
                server_config_json = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Failed to load server configuration: {e}")
        
        # Check for mcpServers structure (Claude Desktop format)
        if "mcpServers" in server_config_json:
            servers_config = server_config_json["mcpServers"]
        else:
            servers_config = server_config_json
            
        if server_name not in servers_config:
            raise ValueError(f"Server '{server_name}' not found in configuration")
            
        server_config = servers_config[server_name]
        
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
        
        # Handle environment variables with ${VAR} pattern
        processed_env = {}
        for key, value in env.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var_name = value[2:-1]
                env_value = os.getenv(env_var_name)
                if env_value:
                    processed_env[key] = env_value
                else:
                    logger.warning(f"Environment variable {env_var_name} not found")
            else:
                processed_env[key] = value
        
        # Direct setting for Brave Search API key if needed
        if server_name == "brave-search" and 'BRAVE_API_KEY' not in processed_env:
            try:
                api_key = config('BRAVE_API_KEY', default=None)
                if api_key:
                    processed_env['BRAVE_API_KEY'] = api_key
                    logger.info("Added Brave API key from environment")
                else:
                    raise ValueError("BRAVE_API_KEY not found in environment")
            except Exception as e:
                raise ValueError(f"Error loading Brave API key: {str(e)}. Please set BRAVE_API_KEY in your .env file.")
        
        # Merge with current environment
        env = {**os.environ, **processed_env}
        
        # Create server parameters
        server_params = StdioServerParameters(
            command=command_path,
            args=args,
            env=env
        )
        
        # Connect to server
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info(f"Connected to server {server_name} with tools: {[tool.name for tool in tools]}")

    async def process_query(self, query: str) -> str:
        """Process a query using LLM and available tools"""
        if not self.session:
            raise RuntimeError("Not connected to an MCP server. Call connect_to_server first.")
            
        logger.info(f"Processing query: {query}")
        
        # Initialize conversation with user query
        messages = [{"role": "user", "content": query}]
        
        # Get available tools
        available_tools = await ToolManager.get_available_tools(self.session)
        
        # Get initial response from LLM
        response = await self.llm_client.get_completion(messages, available_tools)
        
        # Process response and handle any tool calls
        return await self._process_llm_response(response, messages, available_tools)
        
    async def _process_llm_response(
        self, 
        response: Any, 
        messages: List[Dict[str, Any]], 
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """Process an LLM response and handle any tool calls"""
        final_text = []
        
        try:
            # Extract message from the response
            choices = response.choices
            if not choices or len(choices) == 0:
                return "No response generated by the LLM."
                
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
                    await self._process_single_tool_call(tool_call, messages, available_tools, final_text)
        
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
            final_text.append(f"Error occurred while processing: {str(e)}")

        # Join all parts with newlines, ensuring there's no empty content
        result = "\n".join(part for part in final_text if part and part.strip())
        return result if result else "No response content received from the LLM."

    async def _process_single_tool_call(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Process a single tool call"""
        logger.debug(f"Processing tool call: {tool_call}")
        
        # Execute the tool call
        tool_result = await ToolManager.execute_tool_call(self.session, tool_call)
        
        # Add tool call info to output
        tool_name = tool_call.function.name
        tool_args_raw = tool_call.function.arguments
        final_text.append(f"[Calling tool {tool_name} with args {tool_args_raw}]")
        
        # Add result to output
        result_text = tool_result.get('result_text', '')
        if result_text:
            final_text.append(f"Tool result: {result_text}")
        
        # Update message history with tool call
        self._update_message_history_sync(messages, tool_call, tool_result)
        
        # Get follow-up response from LLM
        await self._get_follow_up_response(messages, available_tools, final_text)
        
    async def _handle_tool_calls(
        self, 
        tool_calls: List[Any], 
        messages: List[Dict[str, Any]], 
        available_tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle tool calls from the LLM response"""
        logger.info(f"Processing {len(tool_calls)} tool calls")
        
        for tool_call in tool_calls:
            await self._process_single_tool_call(tool_call, messages, available_tools, final_text)
            
    def _update_message_history_sync(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        tool_result: Dict[str, Any]
    ) -> None:
        """Update message history with tool call and result (synchronous version)"""
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
            "content": tool_result["result_text"]
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
        """Get and process follow-up response from LLM after tool call"""
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
                        await self._process_single_tool_call(tool_call, messages, available_tools, final_text)
        except Exception as e:
            logger.error(f"Error in follow-up API call: {str(e)}", exc_info=True)
            final_text.append(f"Error in follow-up response: {str(e)}")

    async def chat_loop(self) -> None:
        """Run an interactive chat loop"""
        if not self.session:
            raise RuntimeError("Not connected to an MCP server. Call connect_to_server first.")
            
        logger.info("Starting chat loop")
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    logger.info("Exiting chat loop")
                    break
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.exit_stack.aclose()