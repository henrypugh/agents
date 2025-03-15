"""
Simplified LLM Service module for OpenRouter API communication.
"""
import json
import logging
from typing import Dict, List, Any, Optional, Union

from openai import OpenAI
from decouple import config
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

# Import your Pydantic schemas 
from src.utils.schemas import ToolCall

logger = logging.getLogger("LLMService")

class LLMService:
    def __init__(self, model: str = "google/gemini-2.0-flash-001"):
        self.client = OpenAI(
            api_key=config('OPENROUTER_API_KEY'),
            base_url="https://openrouter.ai/api/v1"
        )
        self.model = model
        logger.info(f"LLMService initialized with model: {model}")

    async def get_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """Get completion from LLM API (async wrapper around sync API)"""
        try:
            logger.info("Requesting LLM completion")
            
            # Use run_in_executor to run the synchronous API call in a thread pool
            import asyncio
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools if tools else []
                )
            )
            
            logger.debug("LLM response received")
            return response
        except Exception as e:
            logger.error(f"LLM call error: {e}", exc_info=True)
            raise

    @staticmethod
    def parse_tool_calls(response: Any) -> List[Dict[str, Any]]:
        """Parse tool calls from LLM response into dictionary format"""
        tool_calls = []
        if response.choices:
            for tc in getattr(response.choices[0].message, 'tool_calls', []):
                try:
                    tool_calls.append({
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": json.loads(tc.function.arguments)
                        }
                    })
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing tool arguments: {e}")
        return tool_calls
    
    @staticmethod
    def parse_tool_calls_as_models(response: Any) -> List[ToolCall]:
        """Parse tool calls from LLM response into Pydantic models"""
        tool_calls = []
        if response.choices:
            for tc in getattr(response.choices[0].message, 'tool_calls', []):
                try:
                    args = json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        tool_name=tc.function.name,
                        arguments=args
                    ))
                except (json.JSONDecodeError, Exception) as e:
                    logging.error(f"Error parsing tool call: {e}")
        return tool_calls