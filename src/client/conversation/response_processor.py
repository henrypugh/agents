"""
Response Processor module for handling LLM response parsing and processing.
"""

import logging
from typing import Dict, List, Any, Optional

from traceloop.sdk.decorators import task
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

from src.client.llm_service import LLMService
from src.client.tool_processor import ToolExecutor
from src.utils.schemas import ToolCall, LLMResponse

logger = logging.getLogger("ResponseProcessor")

class ResponseProcessor:
    """Processes LLM responses and handles tool execution"""
    
    def __init__(
        self,
        llm_client: LLMService,
        tool_processor: ToolExecutor,
        server_handler
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
        logger.info("ResponseProcessor initialized")
    
    @task(name="process_llm_response")
    async def process_response(
        self,
        response: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process an LLM response and handle tool calls
        
        Args:
            response: LLM response object
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        try:
            # New structure for responses API
            if response.status != "completed":
                final_text.append(f"Response status: {response.status}. Unable to process.")
                return
            
            # Extract output from response
            if not hasattr(response, 'output') or not response.output:
                final_text.append("No output in the response.")
                return
            
            # Check for text content first
            content_added = False
            output_text = None
            
            if hasattr(response, 'output_text') and response.output_text:
                output_text = response.output_text
                final_text.append(output_text)
                content_added = True
            else:
                # Manually extract text from output
                for item in response.output:
                    if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                text = content_item.text.strip()
                                if text:
                                    final_text.append(text)
                                    content_added = True
                                    output_text = text  # Save for later reference
            
            # Check for tool calls in the response
            tool_calls = self._extract_tool_calls(response)
            
            # Track the type of response received
            has_tool_calls = bool(tool_calls)
            Traceloop.set_association_properties({
                "response_type": "tool_calls" if has_tool_calls else "text",
                "has_content": content_added,
                "tool_call_count": len(tool_calls) if has_tool_calls else 0
            })
            
            logger.info(f"LLM tool usage decision: {'Used tools' if has_tool_calls else 'Did NOT use any tools'}")
            
            if has_tool_calls:
                # Process each tool call
                for tool_call in tool_calls:
                    await self.process_tool_call(tool_call, messages, tools, final_text)
        
        except Exception as e:
            logger.error(f"Error processing LLM response: {str(e)}", exc_info=True)
            final_text.append(f"Error occurred while processing: {str(e)}")
            
            # Track the error
            Traceloop.set_association_properties({
                "error": str(e),
                "error_type": type(e).__name__
            })
    
    @staticmethod
    def _extract_tool_calls(response: Any) -> List[Any]:
        """Extract tool calls from response object"""
        tool_calls = []
        for item in response.output:
            if getattr(item, 'type', '') == 'tool_calls':
                tool_calls.extend(item.tool_calls)
            
            # Alternative format to check
            elif getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                for content_item in item.content:
                    if getattr(content_item, 'type', '') == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                        tool_calls.extend(content_item.tool_calls)
        return tool_calls
    
    @staticmethod
    def extract_tool_calls_as_models(response: Any) -> List[ToolCall]:
        """Extract tool calls as Pydantic models from response object"""
        tool_calls = []
        try:
            for item in response.output:
                if getattr(item, 'type', '') == 'tool_calls':
                    for tc in item.tool_calls:
                        try:
                            tool_calls.append(ToolCall(
                                id=getattr(tc, 'id', 'unknown_id'),
                                tool_name=tc.function.name,
                                arguments=tc.function.arguments
                            ))
                        except Exception as e:
                            logger.warning(f"Failed to parse tool call: {e}")
                
                # Alternative format
                elif getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if getattr(content_item, 'type', '') == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                            for tc in content_item.tool_calls:
                                try:
                                    tool_calls.append(ToolCall(
                                        id=getattr(tc, 'id', 'unknown_id'),
                                        tool_name=tc.function.name,
                                        arguments=tc.function.arguments
                                    ))
                                except Exception as e:
                                    logger.warning(f"Failed to parse tool call: {e}")
        except Exception as e:
            logger.error(f"Error extracting tool calls as models: {e}")
        
        return tool_calls
    
    @task(name="process_tool_call")
    async def process_tool_call(
        self,
        tool_call: Any,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        final_text: List[str]
    ) -> None:
        """
        Process a single tool call
        
        Args:
            tool_call: Tool call from LLM
            messages: Conversation history
            tools: Available tools
            final_text: List to append text to
        """
        logger.debug(f"Processing tool call: {tool_call}")
        
        # Get tool details from the structure
        if hasattr(tool_call, 'function'):
            tool_name = tool_call.function.name
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        else:
            # Alternative structure
            tool_name = getattr(tool_call, 'name', 'unknown')
            tool_id = getattr(tool_call, 'id', 'unknown_id')
        
        # Track the tool call details
        Traceloop.set_association_properties({
            "tool_call_id": tool_id,
            "tool_name": tool_name
        })
        
        # Check if this is an internal server management tool
        if tool_name in ["list_available_servers", "connect_to_server", "list_connected_servers"]:
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
        messages: List[Dict[str, Any]],
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
            # Using track_llm_call for the follow-up request
            with track_llm_call(vendor="openrouter", type="chat") as span:
                # Report the request to Traceloop
                llm_messages = []
                for msg in messages:
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
                follow_up_response = await self.llm_client.get_completion(messages, available_tools)
                
                # Extract and report the text response for Traceloop
                follow_up_text = self._extract_text_from_response(follow_up_response)
                
                # Report the response to Traceloop
                span.report_response(
                    self.llm_client.model,
                    [follow_up_text] if follow_up_text else []
                )
            
            # Check for text content in the response
            output_text = self._extract_text_from_response(follow_up_response)
            if output_text:
                logger.debug(f"Got follow-up content: {output_text[:100]}...")
                final_text.append(output_text)
                
                # Track successful follow-up with content
                Traceloop.set_association_properties({
                    "follow_up_status": "success_with_content",
                    "content_length": len(output_text)
                })
            
            # Check for tool calls
            tool_calls = self._extract_tool_calls(follow_up_response)
            
            # Process tool calls if any
            if tool_calls:
                tool_call_count = len(tool_calls)
                logger.info(f"Found {tool_call_count} tool calls in follow-up")
                
                # Track tool calls
                Traceloop.set_association_properties({
                    "follow_up_status": "tool_calls",
                    "tool_call_count": tool_call_count
                })
                
                # We use individual tool processing to avoid recursion issues
                for tool_call in tool_calls:
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
    
    @staticmethod
    def _extract_text_from_response(response: Any) -> Optional[str]:
        """Extract text content from LLM response"""
        if hasattr(response, 'output_text') and response.output_text:
            return response.output_text.strip()
        
        # Try to manually extract text from the output structure
        for item in response.output:
            if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                for content_item in item.content:
                    if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                        text = content_item.text.strip()
                        if text:
                            return text
        
        return None
    
    @staticmethod
    def as_llm_response(response: Any) -> Optional[LLMResponse]:
        """Convert raw response to LLMResponse model"""
        try:
            # Extract text content
            content = None
            if hasattr(response, 'output_text') and response.output_text:
                content = response.output_text.strip()
            else:
                for item in response.output:
                    if getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                        for content_item in item.content:
                            if getattr(content_item, 'type', '') == 'output_text' and hasattr(content_item, 'text'):
                                content = content_item.text.strip()
                                break
                        if content:
                            break
            
            # Extract tool calls
            tool_calls = []
            for item in response.output:
                if getattr(item, 'type', '') == 'tool_calls':
                    for tc in item.tool_calls:
                        try:
                            tool_calls.append(ToolCall(
                                id=getattr(tc, 'id', 'unknown_id'),
                                tool_name=tc.function.name,
                                arguments=tc.function.arguments
                            ))
                        except Exception:
                            pass
                elif getattr(item, 'type', '') == 'message' and hasattr(item, 'content'):
                    for content_item in item.content:
                        if getattr(content_item, 'type', '') == 'tool_calls' and hasattr(content_item, 'tool_calls'):
                            for tc in content_item.tool_calls:
                                try:
                                    tool_calls.append(ToolCall(
                                        id=getattr(tc, 'id', 'unknown_id'),
                                        tool_name=tc.function.name,
                                        arguments=tc.function.arguments
                                    ))
                                except Exception:
                                    pass
            
            # Determine finish reason
            finish_reason = "stop"
            if tool_calls:
                finish_reason = "tool_calls"
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        except Exception as e:
            logger.error(f"Error converting to LLMResponse: {e}")
            return None