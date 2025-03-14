import json
import logging
from typing import Dict, List, Any, Optional, Tuple

from openai import OpenAI
from decouple import config
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

logger = logging.getLogger("LLMService")

class LLMService:
    """Handles communication with the OpenRouter API"""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        self.client = OpenAI(
            api_key=config('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        logger.info(f"LLM Client initialized with model: {model}")

    async def get_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
        """Get a completion from the LLM API using the responses endpoint"""
        try:
            logger.info("Making API call to LLM using responses endpoint")
            logger.debug(f"Sending messages: {json.dumps(messages, indent=2)}")
            
            # Convert the chat messages to a format suitable for the responses API
            input_content, instructions = self._prepare_input_from_messages(messages)
            
            with track_llm_call(vendor="openrouter", type="chat") as span:
                # Report the request to Traceloop
                llm_messages = []
                for msg in messages:
                    llm_messages.append(LLMMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", "")
                    ))
                
                span.report_request(
                    model=self.model,
                    messages=llm_messages
                )
                
                # Prepare request parameters
                request_params = {
                    "model": self.model,
                    "input": input_content,
                }
                
                # Add tools if provided
                if tools:
                    request_params["tools"] = tools
                
                # Add instructions if available
                if instructions:
                    request_params["instructions"] = instructions
                
                # Make the API call
                response = self.client.responses.create(**request_params)
                
                # Extract and report the response
                output_text = self._extract_output_text(response)
                span.report_response(
                    self.model,
                    [output_text] if output_text else []
                )
            
            logger.debug(f"LLM API response received")
            return response
        except Exception as e:
            logger.error(f"Error in LLM API call: {str(e)}", exc_info=True)
            raise
    
    def _prepare_input_from_messages(self, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        """
        Convert chat completion messages to input format for responses API
        
        Returns:
            tuple: (input_content, instructions)
        """
        if not messages:
            return "", None
        
        # Extract instructions from any system messages
        instructions = None
        for msg in messages:
            if msg.get("role") == "system":
                instructions = msg.get("content", "")
                break
        
        # For simple implementation, use the last user message as input
        user_messages = [m for m in messages if m.get("role") == "user"]
        if user_messages:
            input_content = user_messages[-1].get("content", "")
        else:
            # If no user messages, use the last non-system message
            non_system_messages = [m for m in messages if m.get("role") != "system"]
            input_content = non_system_messages[-1].get("content", "") if non_system_messages else ""
        
        return input_content, instructions
    
    def _extract_output_text(self, response: Any) -> str:
        """Extract the output text from a responses API response"""
        try:
            # Use SDK's output_text if available (convenience property)
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text
            
            # Otherwise navigate the response structure
            if hasattr(response, "output") and response.output:
                for item in response.output:
                    if getattr(item, "type", "") == "message" and hasattr(item, "content"):
                        for content_item in item.content:
                            if getattr(content_item, "type", "") == "output_text" and hasattr(content_item, "text"):
                                return content_item.text
            
            logger.warning("Could not find output text in response structure")
            return ""
        except Exception as e:
            logger.error(f"Error extracting output text: {str(e)}", exc_info=True)
            return ""