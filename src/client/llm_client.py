import json
import logging
from typing import Dict, List, Any

from openai import OpenAI
from decouple import config

logger = logging.getLogger("LLMClient")

class LLMClient:
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
            # logger.debug(f"Available tools: {json.dumps(tools, indent=2)}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
            )
            
            logger.debug(f"LLM API response received")
            return response
        except Exception as e:
            logger.error(f"Error in LLM API call: {str(e)}", exc_info=True)
            raise