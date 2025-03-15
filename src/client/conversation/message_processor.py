"""
Message Processor module for handling conversation flow and message management.
"""

import logging
from typing import Dict, List, Any

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop

from src.client.llm_service import LLMService

logger = logging.getLogger("MessageProcessor")

class MessageProcessor:
    """Manages conversation flow and message processing"""
    
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
        messages: List[Dict[str, Any]], 
        tools: List[Dict[str, Any]],
        response_processor
    ) -> str:
        """
        Run a conversation with the LLM using tools
        
        Args:
            messages: Conversation history
            tools: Available tools
            response_processor: Response processor to handle LLM responses
            
        Returns:
            Generated response
        """
        # Set association properties for conversation context
        Traceloop.set_association_properties({
            "message_count": len(messages),
            "has_history": len(messages) > 1
        })
        
        # Get initial response from LLM
        response = await self.llm_client.get_completion(messages, tools)
        
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
    
    def update_message_history(
        self,
        messages: List[Dict[str, Any]],
        tool_call: Any,
        result_text: str
    ) -> None:
        """
        Update message history with tool call and result
        
        Args:
            messages: Conversation history to update
            tool_call: Tool call from LLM
            result_text: Result of tool execution
        """
        # Extract tool information from potentially different structures
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_args = getattr(tool_call, 'arguments', '{}')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
            
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
                        "arguments": tool_args
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
        
        # Log updated message history for debugging
        logger.debug(f"Updated message history with tool call and result. Now have {len(messages)} messages.")