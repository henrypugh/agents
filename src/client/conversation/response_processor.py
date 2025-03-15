"""
Response Processor module for handling LLM response parsing and processing.

This module provides Pydantic-integrated processing for LLM responses,
including tool call handling and follow-up response management.
"""

import logging
import json
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable, Set

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
    MessageRole
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
        server_handler: ServerManagementProtocol
    ):
        """
        Initialize the response processor
        
        Args:
            llm_client: LLM client instance
            tool_processor: Tool processor instance
            server_handler: Server management handler
        """
        self.llm_client = llm_client
        self.tool_processor = tool_processor
        self.server_handler = server_handler
        
        # Set of tools that are server management tools
        self.server_management_tools: Set[str] = {
            "list_available_servers", 
            "connect_to_server", 
            "list_connected_servers"
        }
        
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
        logger.debug(f"Processing tool call: {tool_call.tool_name}")
        
        # Track the tool call details
        Traceloop.set_association_properties({
            "tool_call_id": tool_call.id,
            "tool_name": tool_call.tool_name
        })
        
        # Check if this is an internal server management tool
        if tool_call.tool_name in self.server_management_tools:
            await self.server_handler.handle_server_management_tool(
                tool_call, messages, tools, final_text, self
            )
            return
            
        # Delegate to the tool processor
        await self.tool_processor.process_external_tool(
            tool_call, 
            messages, 
            tools, 
            final_text,
            self.get_follow_up_response
        )
    
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
        logger.info("Getting follow-up response with tool results")
        
        # Track the follow-up request context
        Traceloop.set_association_properties({
            "follow_up": True,
            "message_count": len(messages)
        })
        
        try:
            # Convert Message models to OpenAI format for Traceloop tracking
            openai_messages = [message.to_openai_format() for message in messages]
            
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
                    "content_length": len(follow_up_response.content)
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
                "error_message": str(e)
            })