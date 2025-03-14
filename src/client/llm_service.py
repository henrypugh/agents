import json
import logging
from typing import Dict, List, Any

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
        """Get a completion from the LLM API"""
        try:
            logger.info("Making API call to LLM")
            logger.debug(f"Sending messages: {json.dumps(messages, indent=2)}")
            
            with track_llm_call(vendor="openrouter", type="chat") as span:
                # Report the request to Traceloop - note that we don't pass tools parameter here
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
                
                # Make the actual API call with tools as before
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                )
                
                # Report the response to Traceloop
                span.report_response(
                    self.model,
                    [choice.message.content for choice in response.choices if hasattr(choice.message, "content")]
                )
            
            logger.debug(f"LLM API response received")
            return response
        except Exception as e:
            logger.error(f"Error in LLM API call: {str(e)}", exc_info=True)
            raise