"""
Conversation Manager module for handling LLM conversations.
"""

import logging
import json
from typing import Dict, List, Any, Optional

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

from src.utils.decorators import message_processing, tool_execution
from .llm_service import LLMService
from .server_registry import ServerRegistry
from .tool_processor import ToolExecutor

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
        self.server_management_tools = self._create_server_management_tools()
        logger.info("Conversation initialized")
    
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
    
    @message_processing
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
        
        # Start conversation
        return await self._run_conversation(messages, all_tools)
    
    @task(name="run_conversation")
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
        # Set association properties for conversation context
        Traceloop.set_association_properties({
            "message_count": len(messages),
            "has_history": len(messages) > 1
        })
        
        # Get initial response from LLM
        response = await self.llm_client.get_completion(messages, tools)
        
        # Process response and handle any tool calls
        final_text = []
        await self._process_response(response, messages, tools, final_text)
        
        # Join all parts with newlines, ensuring there's no empty content
        result = "\n".join(part for part in final_text if part and part.strip())
        
        # Track the result size
        Traceloop.set_association_properties({
            "result_size": len(result),
            "result_parts": len(final_text)
        })
        
        return result if result else "No response content received from the LLM."
    
    @task(name="process_llm_response")
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
            # New structure for responses API
            if response.status != "completed":
                final_text.append(f"Response status: {response.status}. Unable to process.")
                return
            
            # Extract output from response
            if not hasattr(response, 'output') or not response.output:
                final_text.append("No output in the response.")
                return
            
            # Check for text content first
            output_text = None
            if hasattr(response, 'output_text') and response.output_text:
                output_text = response.output_text
                final_text.append(output_text)
                content_added = True
            else:
                # Manually extract text from output
                content_added = False
                for item in response.output:
                    if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                text = content_item.text.strip()
                                if text:
                                    final_text.append(text)
                                    content_added = True
                                    output_text = text  # Save for later reference
            
            # Check for tool calls in the response
            tool_calls = []
            for item in response.output:
                if getattr(item, 'type', '') == 'tool_calls':
                    tool_calls.extend(item.tool_calls)
                
                # Alternative format to check
                elif getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if getattr(content_item, 'type', '') == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                            tool_calls.extend(content_item.tool_calls)
            
            # Track the type of response received
            has_tool_calls = bool(tool_calls)
            Traceloop.set_association_properties({
                "response_type": "tool_calls" if has_tool_calls else "text",
                "has_content": content_added,
                "tool_call_count": len(tool_calls) if has_tool_calls else 0
            })
            
            logger.info(f"LLM tool usage decision: {'Used tools' if has_tool_calls else 'Did NOT use any tools'}")
            
            if has_tool_calls:
                # Process each tool call
                for tool_call in tool_calls:
                    await self._process_tool_call(tool_call, messages, tools, final_text)
        
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
            final_text.append(f"Error occurred while processing: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    @task(name="process_tool_call")
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
        
        # Get tool details from the new structure
        # Adapt to the tool_call structure from responses API
        # The exact structure depends on how OpenRouter implements it
        # We'll need to handle potential different structures
        
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        
        # Track the tool call details
        Traceloop.set_association_properties({
            "tool_call_id": tool_id,
            "tool_name": tool_name
        })
        
        # Check if this is an internal server management tool
        if tool_name in ["list_available_servers", "connect_to_server", "list_connected_servers"]:
            await self._handle_server_management_tool(tool_call, messages, tools, final_text)
            return
            
        # Otherwise, process as a regular server tool
        server_name = self.tool_processor.find_server_for_tool(tool_name, tools)
        
        if not server_name or server_name not in self.server_manager.servers:
            error_msg = f"Error: Can't determine which server handles tool '{tool_name}'. You may need to connect to the appropriate server first using connect_to_server."
            logger.error(error_msg)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "server_not_found",
                "server_name": server_name or "unknown"
            })
            return
        
        # Track the server that will handle this tool
        Traceloop.set_association_properties({
            "server_name": server_name
        })
        
        # Execute the tool
        await self._execute_and_process_tool(
            server_name, 
            tool_call, 
            messages, 
            tools, 
            final_text
        )
    
    @tool_execution
    async def _execute_and_process_tool(
        self,
        server_name: str,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Execute a tool and process the result
        
        Args:
            server_name: Name of the server
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        # Get tool details from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args_raw = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        final_text.append(f"[Calling tool {tool_name} from {server_name} server with args {tool_args_raw}]")
        
        try:
            # Parse arguments
            tool_args = json.loads(tool_args_raw)
            
            # Execute the tool
            result = await self.tool_processor.execute_tool(tool_name, tool_args, server_name)
            
            # Process result into text
            result_text = self.tool_processor.extract_result_text(result)
            if result_text:
                final_text.append(f"Tool result: {result_text}")
            
            # Update message history - we need to adapt this for the new API
            self._update_message_history(messages, tool_call, result_text)
            
            # Get follow-up response
            await self._get_follow_up_response(messages, tools, final_text)
            
        except json.JSONDecodeError as e:
            error_msg = f"Error parsing tool arguments: {str(e)}"
            logger.error(error_msg)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "json_decode",
                "error_message": str(e)
            })
            
        except Exception as e:
            error_msg = f"Error executing tool on {server_name} server: {str(e)}"
            logger.error(error_msg, exc_info=True)
            final_text.append(error_msg)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "execution_failed",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
    
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
        # Extract tool information from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        # Add assistant message with tool call
        tool_call_message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                }
            ]
        }
        messages.append(tool_call_message)
        
        # Add tool response message
        tool_response_message = {
            "role": "tool", 
            "tool_call_id": tool_id,
            "content": result_text
        }
        messages.append(tool_response_message)
        
        # Log updated message history for debugging
        logger.debug(f"Updated message history with tool call and result. Now have {len(messages)} messages.")
    
    @task(name="get_follow_up_response")
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
        
        # Track the follow-up request context
        Traceloop.set_association_properties({
            "follow_up": True,
            "message_count": len(messages)
        })
        
        try:
            # Log message count for debugging
            logger.debug(f"Sending {len(messages)} messages for follow-up")
            
            # Using track_llm_call for the follow-up request
            with track_llm_call(vendor="openrouter", type="chat") as span:
                # Report the request to Traceloop
                llm_messages = []
                for msg in messages:
                    if "role" in msg and "content" in msg and msg["content"] is not None:
                        llm_messages.append(LLMMessage(
                            role=msg["role"],
                            content=msg["content"]
                        ))
                    # Skip messages with no content or tool calls
                
                # Report the follow-up request
                span.report_request(
                    model=self.llm_client.model,
                    messages=llm_messages
                )
                
                # Get follow-up response from LLM
                follow_up_response = await self.llm_client.get_completion(messages, available_tools)
                
                # Extract and report the text response for Traceloop
                follow_up_text = ""
                if hasattr(follow_up_response, 'output_text') and follow_up_response.output_text:
                    follow_up_text = follow_up_response.output_text
                else:
                    # Try to extract text from the output structure
                    for item in follow_up_response.output:
                        if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                            for content_item in item.content:
                                if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                    follow_up_text = content_item.text
                                    break
                            if follow_up_text:
                                break
                
                # Report the response to Traceloop
                span.report_response(
                    self.llm_client.model,
                    [follow_up_text] if follow_up_text else []
                )
            
            # Check for text content in the response
            output_text = None
            if hasattr(follow_up_response, 'output_text') and follow_up_response.output_text:
                output_text = follow_up_response.output_text.strip()
                if output_text:
                    logger.debug(f"Got follow-up content: {output_text[:100]}...")
                    final_text.append(output_text)
                    
                    # Track successful follow-up with content
                    Traceloop.set_association_properties({
                        "follow_up_status": "success_with_content",
                        "content_length": len(output_text)
                    })
            else:
                # Try to manually extract text from the response
                for item in follow_up_response.output:
                    if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                text = content_item.text.strip()
                                if text:
                                    logger.debug(f"Got follow-up content: {text[:100]}...")
                                    final_text.append(text)
                                    
                                    # Track successful follow-up with content
                                    Traceloop.set_association_properties({
                                        "follow_up_status": "success_with_content",
                                        "content_length": len(text)
                                    })
                                    break
            
            # Check for tool calls
            tool_calls = []
            for item in follow_up_response.output:
                if getattr(item, 'type', '') == 'tool_calls':
                    tool_calls.extend(item.tool_calls)
                
                # Alternative format to check
                elif getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if getattr(content_item, 'type', '') == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                            tool_calls.extend(content_item.tool_calls)
            
            # Process tool calls if any
            if tool_calls:
                tool_call_count = len(tool_calls)
                logger.info(f"Found {tool_call_count} tool calls in follow-up")
                
                # Track tool calls
                Traceloop.set_association_properties({
                    "follow_up_status": "tool_calls",
                    "tool_call_count": tool_call_count
                })
                
                # We use individual tool processing to avoid recursion issues
                for tool_call in tool_calls:
                    await self._process_tool_call(tool_call, messages, available_tools, final_text)
        
        except Exception as e:
            logger.error(f"Error in follow-up API call: {str(e)}", exc_info=True)
            final_text.append(f"Error in follow-up response: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "follow_up_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
    
    @task(name="handle_server_management_tool")
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
        # Extract tool information from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args_raw = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args_raw = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        logger.info(f"Handling server management tool: {tool_name}")
        
        # Track the management tool being used
        Traceloop.set_association_properties({
            "management_tool": tool_name
        })
        
        try:
            # Parse arguments if any
            tool_args = {}
            if tool_args_raw:
                try:
                    tool_args = json.loads(tool_args_raw)
                except json.JSONDecodeError as e:
                    final_text.append(f"Error parsing arguments: {str(e)}")
                    
                    # Track the error
                    Traceloop.set_association_properties({
                        "error": "json_decode",
                        "error_message": str(e)
                    })
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
                
                # Track the error
                Traceloop.set_association_properties({
                    "error": "unknown_management_tool"
                })
        
        except Exception as e:
            logger.error(f"Error handling server management tool: {str(e)}", exc_info=True)
            final_text.append(f"Error: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "management_tool_execution",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

    @task(name="list_available_servers")
    async def _handle_list_available_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_available_servers tool"""
        # Extract tool ID from potentially different structures
        if hasattr(tool_call, 'id'):
            tool_id = tool_call.id
        else:
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
        available_servers = await self.server_manager.get_available_servers()
        
        # Track available servers
        Traceloop.set_association_properties({
            "available_server_count": len(available_servers),
            "available_servers": ",".join(available_servers.keys())
        })
        
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
                server_type = info.get("type", "unknown")
                source = info.get("source", "unknown")
                server_list.append(f"{server_name} ({server_type} from {source})")
            
            result_text = f"Available servers ({len(available_servers)}):\n" + "\n".join(server_list)
        else:
            result_text = "No available servers found"
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))

    @task(name="connect_to_server")
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
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "missing_server_name"
            })
            return
            
        server_name = args["server_name"]
        
        # Track the server being connected to
        Traceloop.set_association_properties({
            "target_server": server_name
        })
        
        # Get available servers
        available_servers = await self.server_manager.get_available_servers()
        
        if server_name not in available_servers:
            error_msg = f"Server '{server_name}' not found. Available servers: {', '.join(available_servers.keys())}"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({"error": error_msg}))
            
            # Track the error
            Traceloop.set_association_properties({
                "error": "server_not_found",
                "available_servers": ",".join(available_servers.keys())
            })
            return
        
        # Connect to the server
        try:
            connection_result = await self.server_manager.connect_to_configured_server(server_name)
            result_text = f"Successfully connected to server: {server_name}"
            final_text.append(f"[Server management] {result_text}")
            
            # Track successful connection
            Traceloop.set_association_properties({
                "connection_status": "success",
                "tool_count": len(connection_result.get("tools", []))
            })
            
            # Update message history
            self._update_message_history(messages, tool_call, json.dumps({
                "success": True,
                "server_name": server_name,
                "message": result_text
            }))
            
            # Get follow-up response with updated tools
            updated_tools = self.server_manager.collect_all_tools() + self.server_management_tools
            await self._get_follow_up_response(messages, updated_tools, final_text)
            
        except Exception as e:
            error_msg = f"Failed to connect to server '{server_name}': {str(e)}"
            final_text.append(f"[Server management] Error: {error_msg}")
            self._update_message_history(messages, tool_call, json.dumps({
                "success": False,
                "error": error_msg
            }))
            
            # Track the error
            Traceloop.set_association_properties({
                "connection_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })

    @task(name="list_connected_servers")
    async def _handle_list_connected_servers(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """Handle list_connected_servers tool"""
        connected_servers = self.server_manager.get_connected_servers()
        
        # Track connected servers
        Traceloop.set_association_properties({
            "connected_server_count": len(connected_servers),
            "connected_servers": ",".join(connected_servers.keys())
        })
         
        # Format for JSON response
        result = {
            "connected_servers": {},
            "count": len(connected_servers)
        }
        
        for server_name, tools in connected_servers.items():
            result["connected_servers"][server_name] = tools
        
        # Format for display
        if connected_servers:
            server_list = []
            for server_name, tools in connected_servers.items():
                server_list.append(f"{server_name} - Available tools: {', '.join(tools)}")
            
            result_text = f"Connected servers ({len(connected_servers)}):\n" + "\n".join(server_list)
        else:
            result_text = "No connected servers. Use connect_to_server to connect to a server."
        
        final_text.append(f"[Server management] {result_text}")
        
        # Update message history
        self._update_message_history(messages, tool_call, json.dumps(result))