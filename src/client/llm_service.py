"""
LLM Service module for OpenRouter API communication.

This module provides a service for interacting with language models
through the OpenRouter API, with Pydantic validation for all responses.
"""
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic, cast

from openai import OpenAI
from decouple import config
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task

from src.utils.schemas import (
    Message, 
    ToolCall, 
    LLMResponse, 
    LLMRequest,
    FinishReason,
    T
)

logger = logging.getLogger(__name__)

class LLMService:
    """
    Handles communication with language model APIs with Pydantic model validation.
    
    This service wraps the OpenRouter API and provides standardized methods
    for getting completions with proper validation and error handling.
    """
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        """
        Initialize the LLM service
        
        Args:
            model: LLM model identifier to use with OpenRouter
        """
        self.client = OpenAI(
            api_key=config('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        logger.info(f"LLMService initialized with model: {model}")

    @task(name="get_llm_completion")
    async def get_completion(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Get completion from LLM API with validated Pydantic model response
        
        Args:
            messages: List of conversation messages in OpenAI format
            tools: Optional list of tools in OpenAI format
            
        Returns:
            LLMResponse: Validated Pydantic model of the LLM response
            
        Raises:
            Exception: If the LLM API call fails
        """
        try:
            # Track the LLM request
            Traceloop.set_association_properties({
                "messages_count": len(messages),
                "tools_count": len(tools) if tools else 0,
                "model": self.model
            })
            
            # Run the synchronous API call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else []
                )
            )
            
            # Store the original response for backward compatibility
            # This helps with code that expects the original OpenAI format
            
            # Parse and validate the response
            llm_response = LLMResponse.from_openai_response(response)
            
            # Track the response details
            Traceloop.set_association_properties({
                "has_content": llm_response.content is not None,
                "content_length": len(llm_response.content) if llm_response.content else 0,
                "has_tool_calls": llm_response.has_tool_calls(),
                "tool_calls_count": len(llm_response.tool_calls) if llm_response.tool_calls else 0,
                "finish_reason": llm_response.finish_reason
            })
            
            # For backward compatibility, we'll add the original response too
            # This method doesn't actually modify the response, but ensures
            # we're properly handling the different format expectations
            llm_response._original_response = response
            
            return llm_response
            
        except Exception as e:
            logger.error(f"LLM call error: {e}", exc_info=True)
            
            # Track the error
            Traceloop.set_association_properties({
                "error": str(e),
                "error_type": type(e).__name__
            })
            
            # Return an error response that's still a valid LLMResponse
            return LLMResponse(
                content=f"Error: {str(e)}",
                finish_reason=FinishReason.ERROR
            )

    async def get_completion_with_model(
        self, 
        messages: List[Message], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> LLMResponse:
        """
        Get completion using native Pydantic model messages
        
        Args:
            messages: List of Message models
            tools: Optional list of tools in OpenAI format
            
        Returns:
            LLMResponse: Validated model of the LLM response
        """
        # Convert Message models to OpenAI format
        openai_messages = [message.to_openai_format() for message in messages]
        return await self.get_completion(openai_messages, tools)
    
    async def get_completion_from_request(self, request: LLMRequest) -> LLMResponse:
        """
        Get completion using a complete LLMRequest model
        
        Args:
            request: Full LLMRequest model with all parameters
            
        Returns:
            LLMResponse: Validated model of the LLM response
        """
        # Convert the request to OpenAI format
        openai_format = request.to_openai_format()
        return await self.get_completion(
            openai_format["messages"],
            openai_format.get("tools")
        )
    
    @staticmethod
    def extract_tool_calls(response: LLMResponse) -> List[ToolCall]:
        """
        Extract tool calls from LLM response
        
        Args:
            response: LLM response model
            
        Returns:
            List of ToolCall models
        """
        return response.tool_calls or []

    @staticmethod
    def create_user_message(content: str) -> Message:
        """
        Create a simple user message
        
        Args:
            content: Message content
            
        Returns:
            Message: User message model
        """
        return Message.user(content)
    
    @staticmethod
    def create_system_message(content: str) -> Message:
        """
        Create a system message
        
        Args:
            content: Message content
            
        Returns:
            Message: System message model
        """
        return Message.system(content)