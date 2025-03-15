"""
Message Processor module for handling conversation flow and message management.

This module provides a Pydantic-integrated approach to handling messages
in conversation, including proper validation and type safety throughout.
"""

import logging
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

import json 

from src.client.llm_service import LLMService
from src.utils.schemas import (
    Message, 
    MessageRole, 
    ToolCall,
    LLMResponse
)

logger = logging.getLogger(__name__)

class ResponseProcessorProtocol(Protocol):
    """Protocol defining the response processor interface"""
    async def process_response(
        self, 
        response: LLMResponse,
        messages: List[Message],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        ...

class MessageProcessor:
    """
    Manages conversation flow and message processing with Pydantic validation.
    
    This class handles the core message processing logic, including:
    - Running conversations with the LLM
    - Maintaining properly validated message history
    - Managing conversation context
    """
    
    def __init__(self, llm_client: LLMService):
        """
        Initialize the message processor
        
        Args:
            llm_client: LLM client instance for generating responses
        """
        self.llm_client = llm_client
        logger.info("MessageProcessor initialized")
    
    @task(name="run_conversation")
    async def run_conversation(
        self, 
        messages: List[Message], 
        tools: List[Dict[str, Any]],
        response_processor: ResponseProcessorProtocol
    ) -> str:
        """
        Run a conversation with the LLM using tools
        
        Args:
            messages: Validated conversation history
            tools: Available tools
            response_processor: Response processor to handle LLM responses
            
        Returns:
            Generated response text
        """
        # Set association properties for conversation context
        Traceloop.set_association_properties({
            "message_count": len(messages),
            "has_history": len(messages) > 1
        })
        
        # Convert messages to OpenAI format for the LLM service
        openai_messages = [message.to_openai_format() for message in messages]
        
        # Get initial response from LLM
        response = await self.llm_client.get_completion(openai_messages, tools)
        
        # Process response and handle any tool calls
        final_text = []
        await response_processor.process_response(response, messages, tools, final_text)
        
        # Join all parts with newlines, ensuring there's no empty content
        result = "\n".join(part for part in final_text if part and part.strip())
        
        # Track the result size
        Traceloop.set_association_properties({
            "result_size": len(result),
            "result_parts": len(final_text)
        })
        
        return result if result else "No response content received from the LLM."
    
    @staticmethod
    def update_message_history(
        messages: List[Message],
        tool_call: ToolCall,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result using Pydantic models
        
        Args:
            messages: Validated conversation history to update
            tool_call: Tool call model
            result_text: Result of tool execution
        """
        try:
            # Create assistant message with tool call
            assistant_message = Message.assistant(tool_calls=[tool_call])
            
            # Create tool response message
            tool_message = Message.tool(tool_call_id=tool_call.id, content=result_text)
            
            # Add messages to history
            messages.append(assistant_message)
            messages.append(tool_message)
            
            logger.debug(f"Updated message history with tool call and result. Now have {len(messages)} messages.")
        except Exception as e:
            logger.error(f"Error updating message history: {e}")
            # If using Pydantic models fails, fall back to manual dictionary creation
            _update_message_history_dict(messages, tool_call, result_text)

def _update_message_history_dict(
    messages: List[Any],
    tool_call: Any,
    result_text: str
) -> None:
    """
    Legacy method to update message history with dictionaries
    
    Args:
        messages: Conversation history list (could be either Message objects or dicts)
        tool_call: Tool call (could be either a ToolCall object or other format)
        result_text: Result of tool execution
    """
    # Extract tool information from potentially different structures
    if hasattr(tool_call, 'id'):
        tool_id = tool_call.id
    elif hasattr(tool_call, 'function'):
        tool_id = getattr(tool_call, 'id', 'unknown_id')
    else:
        tool_id = getattr(tool_call, 'id', 'unknown_id')
        
    # Extract tool name
    if hasattr(tool_call, 'tool_name'):
        tool_name = tool_call.tool_name
    elif hasattr(tool_call, 'function'):
        tool_name = tool_call.function.name
    else:
        tool_name = getattr(tool_call, 'name', 'unknown')
        
    # Extract arguments
    if hasattr(tool_call, 'arguments'):
        tool_args = tool_call.arguments
    elif hasattr(tool_call, 'function'):
        tool_args = tool_call.function.arguments
    else:
        tool_args = getattr(tool_call, 'arguments', '{}')
        
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
                    "arguments": tool_args if isinstance(tool_args, str) else json.dumps(tool_args)
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
    
    logger.debug(f"Updated message history with legacy method. Now have {len(messages)} messages.")