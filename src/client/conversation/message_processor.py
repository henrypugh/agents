"""
Message Processor module for handling conversation flow and message management.

This module provides a Pydantic-integrated approach to handling messages
in conversation, including proper validation and type safety throughout.
"""

import logging
from typing import Dict, List, Any, Optional, Protocol, Callable, Awaitable, TypeVar, Type, Union, cast
from pydantic import TypeAdapter

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

import json 

from src.client.llm_service import LLMService
from src.utils.schemas import (
    Message, 
    MessageRole, 
    ToolCall,
    LLMResponse,
    OpenAIAdapter,
    MessageHistory
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

class MessageProcessorConfig:
    """Configuration for message processor"""
    def __init__(
        self,
        max_history_length: int = 100,
        validate_messages: bool = True,
        trace_content: bool = True
    ):
        self.max_history_length = max_history_length
        self.validate_messages = validate_messages
        self.trace_content = trace_content

class MessageProcessor:
    """
    Manages conversation flow and message processing with Pydantic validation.
    
    This class handles the core message processing logic, including:
    - Running conversations with the LLM
    - Maintaining properly validated message history
    - Managing conversation context
    """
    
    def __init__(
        self, 
        llm_client: LLMService,
        config: Optional[MessageProcessorConfig] = None
    ):
        """
        Initialize the message processor
        
        Args:
            llm_client: LLM client instance for generating responses
            config: Optional configuration for message processing
        """
        self.llm_client = llm_client
        self.config = config or MessageProcessorConfig()
        
        # Create TypeAdapter for validating messages
        self.message_validator = TypeAdapter(Message)
        
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
        # Validate the message history if configured
        if self.config.validate_messages:
            self._validate_message_history(messages)
        
        # Trim history if it exceeds max length
        if len(messages) > self.config.max_history_length:
            logger.warning(f"Trimming conversation history from {len(messages)} to {self.config.max_history_length} messages")
            messages = self._trim_history(messages)
        
        # Set association properties for conversation context
        Traceloop.set_association_properties({
            "message_count": len(messages),
            "has_history": len(messages) > 1,
            "tool_count": len(tools)
        })
        
        # Trace message content if configured
        if self.config.trace_content:
            self._trace_message_content(messages)
        
        # Convert messages to OpenAI format
        openai_messages = OpenAIAdapter.messages_to_openai(messages)
        
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
    
    def _validate_message_history(self, messages: List[Message]) -> None:
        """
        Validate message history using Pydantic
        
        Args:
            messages: Message history to validate
            
        Raises:
            ValueError: If any message is invalid
        """
        try:
            for i, message in enumerate(messages):
                # Skip if already a validated Message instance
                if isinstance(message, Message):
                    continue
                    
                # Try to validate the message
                if isinstance(message, dict):
                    # Convert dictionary to Message
                    validated = self.message_validator.validate_python(message)
                    # Replace with validated instance (if possible)
                    messages[i] = validated
                else:
                    logger.warning(f"Unknown message format at index {i}: {type(message)}")
        except Exception as e:
            logger.error(f"Error validating message history: {e}")
            # Continue with best effort - don't raise
    
    def _trim_history(self, messages: List[Message]) -> List[Message]:
        """
        Trim conversation history to fit within max length
        
        Args:
            messages: Message history to trim
            
        Returns:
            Trimmed message history
        """
        # Always keep the latest user message
        latest_user_message = messages[-1]
        
        # Always keep system messages (if any)
        system_messages = [m for m in messages if m.role == MessageRole.SYSTEM]
        
        # Calculate how many recent messages to keep
        remaining_slots = self.config.max_history_length - len(system_messages) - 1
        
        # Get the most recent messages (excluding system and the latest)
        non_system_messages = [m for m in messages if m.role != MessageRole.SYSTEM and m != latest_user_message]
        recent_messages = non_system_messages[-remaining_slots:] if remaining_slots > 0 else []
        
        # Combine system messages, recent messages, and latest user message
        return system_messages + recent_messages + [latest_user_message]
    
    def _trace_message_content(self, messages: List[Message]) -> None:
        """
        Trace message content for debugging
        
        Args:
            messages: Message history to trace
        """
        # Extract message info for tracing
        message_info = []
        for msg in messages:
            # Truncate long content
            content_preview = None
            if msg.content:
                content_preview = f"{msg.content[:50]}..." if len(msg.content) > 50 else msg.content
            
            message_info.append({
                "role": msg.role,
                "content_preview": content_preview,
                "has_tool_calls": bool(msg.tool_calls),
                "tool_call_id": msg.tool_call_id
            })
        
        # Add to trace
        Traceloop.set_association_properties({
            "message_previews": message_info
        })
    
    @staticmethod
    def update_message_history(
        messages: List[Message],
        tool_call: ToolCall,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result using MessageHistory utility
        
        Args:
            messages: Validated conversation history to update
            tool_call: Tool call model
            result_text: Result of tool execution
        """
        MessageHistory.add_tool_interaction(messages, tool_call, result_text)