"""
LLM Service module for OpenRouter API communication.

This module provides a service for interacting with language models
through the OpenRouter API, with Pydantic validation for all responses.
"""
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Type, TypeVar, Generic, cast, Union

from openai import OpenAI
from decouple import config
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task
from pydantic import TypeAdapter

from src.utils.schemas import (
    Message, 
    ToolCall, 
    LLMResponse, 
    LLMRequest,
    FinishReason,
    OpenAIAdapter,
    T
)

logger = logging.getLogger(__name__)

# Retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_TIMEOUT = 30.0

class RetrySettings:
    """Configuration for retry behavior"""
    def __init__(
        self, 
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        timeout: float = DEFAULT_TIMEOUT
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout


class LLMService:
    """
    Handles communication with language model APIs with Pydantic model validation.
    
    This service wraps the OpenRouter API and provides standardized methods
    for getting completions with proper validation and error handling.
    """
    
    def __init__(
        self, 
        model: str = "google/gemini-2.0-flash-001", 
        retry_settings: Optional[RetrySettings] = None
    ):
        """
        Initialize the LLM service
        
        Args:
            model: LLM model identifier to use with OpenRouter
            retry_settings: Optional settings for retry behavior
        """
        self.client = OpenAI(
            api_key=config('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        self.retry_settings = retry_settings or RetrySettings()
        
        # Prepare validators for response parsing
        self.llm_response_validator = TypeAdapter(LLMResponse)
        
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
            Exception: If the LLM API call fails after retries
        """
        # Track the LLM request
        Traceloop.set_association_properties({
            "messages_count": len(messages),
            "tools_count": len(tools) if tools else 0,
            "model": self.model
        })
        
        # Apply retry logic
        for attempt in range(self.retry_settings.max_retries):
            try:
                # Run the synchronous API call in a thread pool
                loop = asyncio.get_event_loop()
                response = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools if tools else []
                        )
                    ),
                    timeout=self.retry_settings.timeout
                )
                
                # Parse and validate the response
                llm_response = OpenAIAdapter.parse_openai_response(response)
                
                # Track the response details
                Traceloop.set_association_properties({
                    "has_content": llm_response.content is not None,
                    "content_length": len(llm_response.content) if llm_response.content else 0,
                    "has_tool_calls": llm_response.has_tool_calls(),
                    "tool_calls_count": len(llm_response.tool_calls) if llm_response.tool_calls else 0,
                    "finish_reason": llm_response.finish_reason,
                    "attempt": attempt + 1
                })
                
                return llm_response
                
            except asyncio.TimeoutError:
                logger.warning(f"LLM call timed out (attempt {attempt+1}/{self.retry_settings.max_retries})")
                
                # Track the timeout
                Traceloop.set_association_properties({
                    "attempt": attempt + 1,
                    "error": "timeout",
                    "retry": attempt + 1 < self.retry_settings.max_retries
                })
                
                # Last attempt - raise the error
                if attempt + 1 >= self.retry_settings.max_retries:
                    return LLMResponse(
                        content=f"Request timed out after {self.retry_settings.max_retries} attempts",
                        finish_reason=FinishReason.ERROR
                    )
                    
                # Wait before retrying
                await asyncio.sleep(self.retry_settings.retry_delay * (attempt + 1))
                
            except Exception as e:
                logger.error(f"LLM call error (attempt {attempt+1}/{self.retry_settings.max_retries}): {e}", exc_info=True)
                
                # Track the error
                Traceloop.set_association_properties({
                    "attempt": attempt + 1,
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "retry": attempt + 1 < self.retry_settings.max_retries
                })
                
                # Last attempt - return an error response
                if attempt + 1 >= self.retry_settings.max_retries:
                    return LLMResponse(
                        content=f"Error: {str(e)}",
                        finish_reason=FinishReason.ERROR
                    )
                    
                # Wait before retrying
                await asyncio.sleep(self.retry_settings.retry_delay * (attempt + 1))
        
        # This should not be reached due to the returns in the loop,
        # but added as a fallback
        return LLMResponse(
            content="Unknown error occurred during LLM call",
            finish_reason=FinishReason.ERROR
        )

    @task(name="get_completion_with_model")
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
        # Convert Message models to OpenAI format using the adapter
        openai_messages = OpenAIAdapter.messages_to_openai(messages)
        return await self.get_completion(openai_messages, tools)
    
    @task(name="get_completion_from_request")
    async def get_completion_from_request(self, request: LLMRequest) -> LLMResponse:
        """
        Get completion using a complete LLMRequest model
        
        Args:
            request: Full LLMRequest model with all parameters
            
        Returns:
            LLMResponse: Validated model of the LLM response
        """
        # Convert the request to OpenAI format using the adapter
        openai_format = OpenAIAdapter.request_to_openai(request)
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
    
    @staticmethod
    def create_error_response(error_message: str) -> LLMResponse:
        """
        Create an error response
        
        Args:
            error_message: Error message
            
        Returns:
            LLMResponse: Error response model
        """
        return LLMResponse(
            content=f"Error: {error_message}",
            finish_reason=FinishReason.ERROR
        )