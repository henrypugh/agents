"""
Response Processor module for handling LLM response parsing and processing.

This module provides Pydantic-integrated processing for LLM responses,
including tool call handling and follow-up response management.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable, Set, TypeVar, Union
from pydantic import TypeAdapter, ValidationError

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

from src.client.llm_service import LLMService
from src.client.tool_processor import ToolExecutor
from src.utils.schemas import (
    Message,
    ToolCall,
    LLMResponse,
    FinishReason,
    MessageRole,
    OpenAIAdapter,
    MessageHistory
)

logger = logging.getLogger(__name__)

class ServerManagementProtocol(Protocol):
    """Protocol defining the server management handler interface"""
    async def handle_server_management_tool(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str],
        response_processor: 'ResponseProcessor'
    ) -> None:
        ...

class ResponseProcessorConfig:
    """Configuration for response processor"""
    def __init__(
        self,
        max_sequential_tools: int = 5,
        max_tool_call_depth: int = 3,
        trace_tool_calls: bool = True,
        validate_responses: bool = True,
        enable_follow_up: bool = True
    ):
        self.max_sequential_tools = max_sequential_tools
        self.max_tool_call_depth = max_tool_call_depth
        self.trace_tool_calls = trace_tool_calls
        self.validate_responses = validate_responses
        self.enable_follow_up = enable_follow_up


class ResponseProcessor:
    """
    Processes LLM responses and handles tool execution with Pydantic validation.
    
    This class handles:
    - Processing of validated LLM responses
    - Tool call extraction and execution
    - Follow-up response management
    """
    
    def __init__(
        self,
        llm_client: LLMService,
        tool_processor: ToolExecutor,
        server_handler: ServerManagementProtocol,
        config: Optional[ResponseProcessorConfig] = None
    ):
        """
        Initialize the response processor
        
        Args:
            llm_client: LLM client instance
            tool_processor: Tool processor instance
            server_handler: Server management handler
            config: Optional configuration
        """
        self.llm_client = llm_client
        self.tool_processor = tool_processor
        self.server_handler = server_handler
        self.config = config or ResponseProcessorConfig()
        
        # Set of tools that are server management tools
        self.server_management_tools: Set[str] = {
            "list_available_servers", 
            "connect_to_server", 
            "list_connected_servers"
        }
        
        # Create type adapter for validation
        self.llm_response_validator = TypeAdapter(LLMResponse)
        self.tool_call_validator = TypeAdapter(ToolCall)
        
        # Initialize call depth counter
        self._tool_call_depth = 0
        self._sequential_tool_count = 0
        
        logger.info("ResponseProcessor initialized")
    
    @task(name="process_llm_response")
    async def process_response(
        self,
        response: LLMResponse,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process an LLM response and handle tool calls
        
        Args:
            response: Validated LLM response
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        # Reset counters for each top-level response
        self._tool_call_depth = 0
        self._sequential_tool_count = 0
        
        try:
            # Check for error in response
            if response.finish_reason == FinishReason.ERROR:
                error_msg = response.content or "Unknown error in LLM response"
                final_text.append(f"Error: {error_msg}")
                return
                
            # Extract text content if available
            if response.content:
                final_text.append(response.content)
                
            # Track the type of response received
            has_tool_calls = response.has_tool_calls()
            Traceloop.set_association_properties({
                "response_type": "tool_calls" if has_tool_calls else "text",
                "has_content": response.content is not None,
                "tool_call_count": len(response.tool_calls) if has_tool_calls else 0
            })
            
            logger.info(f"LLM tool usage decision: {'Used tools' if has_tool_calls else 'Did NOT use any tools'}")
            
            # Process tool calls if any
            if has_tool_calls and response.tool_calls:
                for tool_call in response.tool_calls:
                    # Validate the tool call if configured
                    if self.config.validate_responses:
                        try:
                            validated_tool_call = self.tool_call_validator.validate_python(tool_call)
                            await self.process_tool_call(validated_tool_call, messages, tools, final_text)
                        except ValidationError as e:
                            logger.warning(f"Invalid tool call: {e}")
                            error_msg = f"Error: Invalid tool call format - {str(e)}"
                            final_text.append(error_msg)
                    else:
                        await self.process_tool_call(tool_call, messages, tools, final_text)
        
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
            final_text.append(f"Error occurred while processing: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    @task(name="process_tool_call")
    async def process_tool_call(
        self,
        tool_call: ToolCall,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process a single tool call
        
        Args:
            tool_call: Validated tool call model
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        # Increment depth counter
        self._tool_call_depth += 1
        self._sequential_tool_count += 1
        
        # Check if we've exceeded maximum depth or sequential count
        if self._tool_call_depth > self.config.max_tool_call_depth:
            logger.warning(f"Maximum tool call depth ({self.config.max_tool_call_depth}) exceeded")
            final_text.append(f"Error: Maximum tool call depth exceeded")
            self._tool_call_depth -= 1
            return
            
        if self._sequential_tool_count > self.config.max_sequential_tools:
            logger.warning(f"Maximum sequential tool count ({self.config.max_sequential_tools}) exceeded")
            final_text.append(f"Error: Too many sequential tool calls")
            self._tool_call_depth -= 1
            return
        
        logger.debug(f"Processing tool call: {tool_call.tool_name}")
        
        # Track the tool call details
        if self.config.trace_tool_calls:
            Traceloop.set_association_properties({
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.tool_name,
                "tool_call_depth": self._tool_call_depth,
                "sequential_tool_count": self._sequential_tool_count,
                "tool_call_start": time.time()
            })
        
        try:
            # Check if this is an internal server management tool
            if tool_call.tool_name in self.server_management_tools:
                await self.server_handler.handle_server_management_tool(
                    tool_call, messages, tools, final_text, self
                )
            else:
                # Delegate to the tool processor
                await self.tool_processor.process_external_tool(
                    tool_call, 
                    messages, 
                    tools, 
                    final_text,
                    self.get_follow_up_response
                )
        except Exception as e:
            logger.error(f"Error processing tool call: {e}", exc_info=True)
            final_text.append(f"Error processing tool call: {str(e)}")
            
            # Track the error
            if self.config.trace_tool_calls:
                Traceloop.set_association_properties({
                    "tool_call_error": str(e),
                    "error_type": type(e).__name__
                })
        finally:
            # Always decrement the depth counter
            self._tool_call_depth -= 1
            
            # Reset sequential counter if we're back at the top level
            if self._tool_call_depth == 0:
                self._sequential_tool_count = 0
                
            # Track tool call completion
            if self.config.trace_tool_calls:
                Traceloop.set_association_properties({
                    "tool_call_complete": True,
                    "tool_call_end": time.time()
                })
    
    @task(name="get_follow_up_response")
    async def get_follow_up_response(
        self,
        messages: List[Message],
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
        # Skip follow-up if disabled
        if not self.config.enable_follow_up:
            logger.info("Follow-up responses disabled, skipping")
            return
            
        logger.info("Getting follow-up response with tool results")
        
        # Track the follow-up request context
        Traceloop.set_association_properties({
            "follow_up": True,
            "message_count": len(messages),
            "follow_up_start": time.time()
        })
        
        try:
            # Convert Message models to OpenAI format for Traceloop tracking
            openai_messages = OpenAIAdapter.messages_to_openai(messages)
            
            # Using track_llm_call for the follow-up request
            with track_llm_call(vendor="openrouter", type="chat") as span:
                # Report the request to Traceloop
                llm_messages = []
                for msg in openai_messages:
                    if "role" in msg and "content" in msg and msg["content"] is not None:
                        llm_messages.append(LLMMessage(
                            role=msg["role"],
                            content=msg["content"]
                        ))
                
                # Report the follow-up request
                span.report_request(
                    model=self.llm_client.model,
                    messages=llm_messages
                )
                
                # Get follow-up response from LLM
                follow_up_response = await self.llm_client.get_completion(openai_messages, available_tools)
                
                # Extract and report the text response for Traceloop
                follow_up_text = follow_up_response.content
                
                # Report the response to Traceloop
                span.report_response(
                    self.llm_client.model,
                    [follow_up_text] if follow_up_text else []
                )
            
            # Add text content to final result if available
            if follow_up_response.content:
                logger.debug(f"Got follow-up content: {follow_up_response.content[:100]}...")
                final_text.append(follow_up_response.content)
                
                # Track successful follow-up with content
                Traceloop.set_association_properties({
                    "follow_up_status": "success_with_content",
                    "content_length": len(follow_up_response.content),
                    "follow_up_end": time.time()
                })
            
            # Process tool calls if any
            if follow_up_response.has_tool_calls() and follow_up_response.tool_calls:
                tool_call_count = len(follow_up_response.tool_calls)
                logger.info(f"Found {tool_call_count} tool calls in follow-up")
                
                # Track tool calls
                Traceloop.set_association_properties({
                    "follow_up_status": "tool_calls",
                    "tool_call_count": tool_call_count
                })
                
                # Process each tool call individually to avoid recursion issues
                for tool_call in follow_up_response.tool_calls:
                    await self.process_tool_call(tool_call, messages, available_tools, final_text)
        
        except Exception as e:
            logger.error(f"Error in follow-up API call: {str(e)}", exc_info=True)
            final_text.append(f"Error in follow-up response: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "follow_up_status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "follow_up_end": time.time()
            })
            
    def update_message_history(
        self,
        messages: List[Message],
        tool_call: ToolCall,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result using MessageHistory utility
        
        Args:
            messages: Conversation history
            tool_call: Tool call
            result_text: Result text
        """
        MessageHistory.add_tool_interaction(messages, tool_call, result_text)
        
    def set_max_tool_call_depth(self, depth: int) -> None:
        """
        Set maximum tool call depth
        
        Args:
            depth: Maximum depth
        """
        if depth < 1:
            logger.warning(f"Invalid tool call depth: {depth}, using 1")
            self.config.max_tool_call_depth = 1
        else:
            self.config.max_tool_call_depth = depth
            
    def set_max_sequential_tools(self, count: int) -> None:
        """
        Set maximum sequential tool count
        
        Args:
            count: Maximum count
        """
        if count < 1:
            logger.warning(f"Invalid sequential tool count: {count}, using 1")
            self.config.max_sequential_tools = 1
        else:
            self.config.max_sequential_tools = count
            
    def enable_follow_up(self, enable: bool) -> None:
        """
        Enable or disable follow-up responses
        
        Args:
            enable: Whether to enable follow-up
        """
        self.config.enable_follow_up = enable
        logger.info(f"Follow-up responses {'enabled' if enable else 'disabled'}")