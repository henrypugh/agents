"""
Core data models for the MCP client.

This module provides standardized Pydantic models for data validation,
serialization, and documentation throughout the application.
"""

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, computed_field
from typing import Optional, List, Dict, Any, Union, Literal, TypeVar, Type, Generic, Annotated, Protocol, cast
from enum import Enum, auto
from datetime import datetime
import uuid
import json
import logging
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

# Generic type for model return values
T = TypeVar('T', bound=BaseModel)


class BaseModelWithGetMethod(BaseModel):
    """Base model with a get method for backward compatibility"""
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Provide dictionary-like get method for backward compatibility
        
        Args:
            key: Attribute name to get
            default: Default value if attribute doesn't exist
            
        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)


class ToolArguments(BaseModelWithGetMethod):
    """Model for tool arguments"""
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    
    model_config = ConfigDict(
        frozen=False,  # Arguments may be modified
        extra="allow"  # Allow extra fields for flexibility
    )


class ToolCall(BaseModelWithGetMethod):
    """Model for LLM-generated tool calls"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique ID for the tool call")
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    server_name: Optional[str] = Field(None, description="Server that provides the tool")

    model_config = ConfigDict(
        frozen=True,  # Tool calls should be immutable
        json_schema_extra={
            "examples": [{
                "id": "call_123",
                "tool_name": "search_web",
                "arguments": {"query": "pydantic tutorials"},
                "server_name": "search-server"
            }]
        }
    )

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API tool call format"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments)
            }
        }
    
    @classmethod
    def from_openai_format(cls, data: Dict[str, Any], server_name: Optional[str] = None) -> "ToolCall":
        """Create from OpenAI API tool call format"""
        try:
            arguments = json.loads(data["function"]["arguments"]) if isinstance(data["function"]["arguments"], str) else data["function"]["arguments"]
        except json.JSONDecodeError:
            # Handle malformed JSON
            logger.warning(f"Malformed JSON in tool arguments: {data['function'].get('arguments', '')}")
            arguments = {}
            
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            tool_name=data["function"]["name"],
            arguments=arguments,
            server_name=server_name
        )
    
    @classmethod
    def from_tool_call(cls, tool_call: Any) -> "ToolCall":
        """Create from various tool call formats"""
        try:
            # Handle already being a ToolCall instance
            if isinstance(tool_call, cls):
                return tool_call
                
            # Handle OpenAI format
            if hasattr(tool_call, 'function'):
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                tool_id = getattr(tool_call, 'id', str(uuid.uuid4())[:8])
                
                # Parse arguments if they're a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {tool_args}")
                        tool_args = {}
                
                return cls(
                    id=tool_id,
                    tool_name=tool_name,
                    arguments=tool_args
                )
                
            # Handle alternative structure
            if hasattr(tool_call, 'name') or hasattr(tool_call, 'tool_name'):
                tool_name = getattr(tool_call, 'tool_name', getattr(tool_call, 'name', 'unknown'))
                tool_args = getattr(tool_call, 'arguments', {})
                tool_id = getattr(tool_call, 'id', str(uuid.uuid4())[:8])
                
                # Parse arguments if they're a string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse tool arguments: {tool_args}")
                        tool_args = {}
                
                return cls(
                    id=tool_id,
                    tool_name=tool_name,
                    arguments=tool_args
                )
                
            # Handle dictionary
            if isinstance(tool_call, dict):
                if 'function' in tool_call:
                    # OpenAI format dict
                    return cls.from_openai_format(tool_call)
                else:
                    # Direct dict
                    return cls.model_validate(tool_call)
                    
            # Unknown format
            raise ValueError(f"Unrecognized tool call format: {type(tool_call)}")
            
        except Exception as e:
            logger.error(f"Error converting to ToolCall: {e}")
            # Return a minimal valid tool call
            return cls(
                id=str(uuid.uuid4())[:8],
                tool_name="unknown_tool",
                arguments={}
            )


class ToolResultStatus(str, Enum):
    """Status values for tool execution results"""
    SUCCESS = "success"
    ERROR = "error" 
    TIMEOUT = "timeout"


class ToolResult(BaseModelWithGetMethod):
    """Model for tool execution results"""
    tool_id: str = Field(..., description="ID of the tool call this result responds to")
    status: ToolResultStatus = Field(..., description="Status of the tool execution (success, error, timeout)")
    result: Optional[Any] = Field(None, description="Result data from successful execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(0.0, description="Time taken to execute the tool in seconds")
    
    model_config = ConfigDict(
        frozen=True,  # Results should be immutable
        json_schema_extra={
            "examples": [{
                "tool_id": "call_123",
                "status": "success",
                "result": "The calculation result is 42",
                "execution_time": 0.35
            }]
        }
    )
    
    @model_validator(mode='after')
    def validate_status_fields(self) -> 'ToolResult':
        """Validate that the appropriate fields are set based on status"""
        if self.status == ToolResultStatus.SUCCESS and self.result is None:
            raise ValueError("Result must be provided for successful tool execution")
        if self.status == ToolResultStatus.ERROR and self.error is None:
            raise ValueError("Error message must be provided for failed tool execution")
        return self
    
    @classmethod
    def success(cls, tool_id: str, result: Any, execution_time: float = 0.0) -> 'ToolResult':
        """Create a success result"""
        return cls(
            tool_id=tool_id,
            status=ToolResultStatus.SUCCESS,
            result=result,
            execution_time=execution_time
        )
    
    @classmethod
    def error(cls, tool_id: str, error: str, execution_time: float = 0.0) -> 'ToolResult':
        """Create an error result"""
        return cls(
            tool_id=tool_id,
            status=ToolResultStatus.ERROR,
            error=error,
            execution_time=execution_time
        )
    
    @classmethod
    def timeout(cls, tool_id: str, execution_time: float) -> 'ToolResult':
        """Create a timeout result"""
        return cls(
            tool_id=tool_id,
            status=ToolResultStatus.TIMEOUT,
            error="Tool execution timed out",
            execution_time=execution_time
        )


class MessageRole(str, Enum):
    """Valid roles for conversation messages"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModelWithGetMethod):
    """Model for conversation messages"""
    role: MessageRole = Field(..., description="Role of the message sender (user, assistant, system, tool)")
    content: Optional[str] = Field(None, description="Content of the message")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls in the message")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message responds to")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this message was created")
    
    model_config = ConfigDict(
        frozen=True,  # Messages should be immutable
        json_schema_extra={
            "examples": [{
                "role": "user",
                "content": "Hello, I need help with something.",
                "timestamp": "2023-01-01T12:00:00Z"
            }]
        }
    )
    
    @model_validator(mode='after')
    def check_content_or_tool_calls(self) -> 'Message':
        """Ensure message has either content or tool calls for assistant messages"""
        if self.role == MessageRole.ASSISTANT:
            if self.content is None and (self.tool_calls is None or len(self.tool_calls) == 0):
                raise ValueError("Assistant messages must have either content or tool calls")
        
        if self.role == MessageRole.TOOL and self.tool_call_id is None:
            raise ValueError("Tool messages must have a tool_call_id")
            
        return self
    
    @computed_field
    @property
    def is_tool_call(self) -> bool:
        """Check if this message contains tool calls"""
        return bool(self.tool_calls and len(self.tool_calls) > 0)
    
    @computed_field
    @property
    def is_tool_response(self) -> bool:
        """Check if this message is a tool response"""
        return self.role == MessageRole.TOOL and bool(self.tool_call_id)
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API message format"""
        result = {"role": self.role}
        
        if self.content is not None:
            result["content"] = self.content
            
        if self.tool_calls:
            result["tool_calls"] = [tc.to_openai_format() for tc in self.tool_calls]
            
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
            
        return result
    
    @classmethod
    def from_openai_format(cls, data: Dict[str, Any]) -> "Message":
        """Create from OpenAI API message format"""
        tool_calls = None
        if "tool_calls" in data and data["tool_calls"]:
            tool_calls = [
                ToolCall.from_openai_format(tc) 
                for tc in data["tool_calls"]
            ]
            
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_call_id=data.get("tool_call_id")
        )
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message"""
        return cls(role=MessageRole.USER, content=content)
    
    @classmethod
    def assistant(cls, content: Optional[str] = None, tool_calls: Optional[List[ToolCall]] = None) -> "Message":
        """Create an assistant message"""
        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tool_calls)
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message"""
        return cls(role=MessageRole.SYSTEM, content=content)
    
    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        """Create a tool message"""
        return cls(role=MessageRole.TOOL, tool_call_id=tool_call_id, content=content)


class ServerStatus(str, Enum):
    """Server connection status values"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    CONNECTING = "connecting"


# Reusable annotated type for server name validation
def normalize_server_name(name: str) -> str:
    """Normalize server name by removing invalid characters"""
    # Only keep alphanumeric chars and hyphens
    normalized = ''.join(c for c in name.lower() if c.isalnum() or c == '-')
    return normalized or "unknown-server"

ServerName = Annotated[str, Field(description="Normalized server name")]


class ServerInfo(BaseModelWithGetMethod):
    """Model for server information"""
    name: str = Field(..., description="Name of the server")
    connected: bool = Field(False, description="Whether server is currently connected")
    tools: List[str] = Field(default_factory=list, description="Available tools on this server")
    connected_at: Optional[datetime] = Field(None, description="When the server was connected")
    status: ServerStatus = Field(ServerStatus.DISCONNECTED, description="Current status of the server connection")
    
    model_config = ConfigDict(
        frozen=True,  # Server info should be immutable
        json_schema_extra={
            "examples": [{
                "name": "search-server",
                "connected": True,
                "tools": ["search_web", "search_news"],
                "connected_at": "2023-01-01T12:00:00Z",
                "status": "connected"
            }]
        }
    )
    
    @model_validator(mode='after')
    def validate_connected_state(self) -> 'ServerInfo':
        """Validate connected status is consistent"""
        if self.connected and self.status != ServerStatus.CONNECTED:
            raise ValueError(f"Server marked as connected but status is {self.status}")
        if not self.connected and self.status == ServerStatus.CONNECTED:
            raise ValueError("Server marked as disconnected but status is 'connected'")
        if self.connected and self.connected_at is None:
            raise ValueError("Connected servers must have a connected_at timestamp")
        return self


# Input schema for tools using TypedDict for better typing
class ToolInputSchema(TypedDict, total=False):
    type: str
    properties: Dict[str, Any]
    required: List[str]


class ServerToolInfo(BaseModelWithGetMethod):
    """Information about a tool available on a server"""
    name: str = Field(..., description="Name of the tool")
    description: Optional[str] = Field(None, description="Description of the tool") 
    input_schema: Dict[str, Any] = Field(..., description="Schema for tool inputs")
    server_name: str = Field(..., description="Name of the server providing this tool")
    
    model_config = ConfigDict(
        frozen=True  # Tool info should be immutable
    )
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API tool format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description or f"Tool: {self.name}",
                "parameters": self.input_schema,
                "metadata": {
                    "server": self.server_name
                }
            }
        }


class LLMRequest(BaseModelWithGetMethod):
    """Model for LLM API requests"""
    messages: List[Message] = Field(..., description="Conversation messages")
    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tools available to the LLM")
    
    model_config = ConfigDict(
        frozen=True  # Requests should be immutable once created
    )
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI API request format"""
        result = {
            "model": self.model,
            "messages": [message.to_openai_format() for message in self.messages],
            "temperature": self.temperature
        }
        
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
            
        if self.tools is not None:
            result["tools"] = self.tools
            
        return result


class FinishReason(str, Enum):
    """Reasons why LLM generation might finish"""
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"


class MessageWrapper(BaseModelWithGetMethod):
    """Wrapper to mimic OpenAI message structure"""
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Allow ToolCall objects
    )


class ChoiceWrapper(BaseModelWithGetMethod):
    """Wrapper to mimic OpenAI choice structure"""
    message: MessageWrapper
    finish_reason: str
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Allow MessageWrapper objects
    )


class LLMResponse(BaseModelWithGetMethod):
    """Model for LLM API responses"""
    content: Optional[str] = Field(None, description="Generated text content")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls generated by the LLM")
    finish_reason: FinishReason = Field(..., description="Reason why the generation finished")
    _original_response: Optional[Any] = None  # Store original response for compatibility
    
    model_config = ConfigDict(
        frozen=True,  # Responses should be immutable
        arbitrary_types_allowed=True  # Allow storing original response
    )
    
    @computed_field
    def choices(self) -> List[Dict[str, Any]]:
        """
        Provide backward compatibility with OpenAI response format
        
        Returns:
            List containing a single choice with message
        """
        # Return a simple dictionary format instead of wrapper objects
        return [{
            "message": {
                "content": self.content,
                "tool_calls": [tc.to_openai_format() for tc in self.tool_calls] if self.tool_calls else None
            },
            "finish_reason": self.finish_reason
        }]
    
    @model_validator(mode='after')
    def validate_content_or_tool_calls(self) -> 'LLMResponse':
        """Validate that finish_reason matches content/tool_calls"""
        if self.finish_reason == FinishReason.TOOL_CALLS and not self.tool_calls:
            raise ValueError("Tool calls must be present when finish_reason is 'tool_calls'")
        return self
    
    @classmethod
    def from_openai_response(cls, response) -> "LLMResponse":
        """Create from OpenAI API response"""
        try:
            choice = response.choices[0]
            message = choice.message
            
            tool_calls = None
            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = [
                    ToolCall.from_openai_format(tc) 
                    for tc in message.tool_calls
                ]
                
            instance = cls(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason
            )
            # Store original response for compatibility
            instance._original_response = response
            return instance
            
        except Exception as e:
            logger.error(f"Error converting OpenAI response to LLMResponse: {e}")
            # Create minimal valid response
            return cls(
                content="Error parsing LLM response",
                finish_reason=FinishReason.ERROR
            )
    
    def has_tool_calls(self) -> bool:
        """Check if this response contains tool calls"""
        return bool(self.tool_calls and len(self.tool_calls) > 0)


class ServerListResponse(BaseModelWithGetMethod):
    """Response model for list_available_servers tool"""
    available_servers: Dict[str, ServerInfo] = Field(default_factory=dict)
    count: int = Field(0, description="Number of available servers")
    
    model_config = ConfigDict(
        frozen=True  # Response should be immutable
    )


class ConnectResponseStatus(str, Enum):
    """Status values for connect response"""
    CONNECTED = "connected"
    ALREADY_CONNECTED = "already_connected"
    ERROR = "error"


class ConnectResponse(BaseModelWithGetMethod):
    """Response model for connect_to_server tool"""
    status: ConnectResponseStatus = Field(..., description="Connection status")
    server: str = Field(..., description="Name of the server")
    tools: Optional[List[str]] = Field(None, description="Available tools")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    tool_count: Optional[int] = Field(None, description="Number of available tools")
    
    model_config = ConfigDict(
        frozen=True  # Response should be immutable
    )
    
    @model_validator(mode='after')
    def validate_error_field(self) -> 'ConnectResponse':
        """Validate error field is present for error status"""
        if self.status == ConnectResponseStatus.ERROR and self.error is None:
            raise ValueError("Error message must be provided when status is 'error'")
        return self


class AgentConfig(BaseModelWithGetMethod):
    """Configuration for the Agent"""
    model: str = Field("google/gemini-2.0-flash-001", description="LLM model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate") 
    retries: int = Field(3, description="Number of retries for LLM calls")
    timeout: float = Field(60.0, description="Timeout for LLM calls in seconds")
    
    model_config = ConfigDict(
        frozen=False  # Config can be modified
    )
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    @field_validator('retries')
    @classmethod
    def validate_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Retries must be non-negative")
        return v
    
    @field_validator('timeout')
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


# Format adapters for centralized conversion logic
class OpenAIAdapter:
    """Centralized adapter for OpenAI format conversions"""
    
    @staticmethod
    def message_to_openai(message: Message) -> Dict[str, Any]:
        """Convert Message to OpenAI format"""
        return message.to_openai_format()
    
    @staticmethod
    def messages_to_openai(messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert list of Messages to OpenAI format"""
        return [OpenAIAdapter.message_to_openai(msg) for msg in messages]
    
    @staticmethod
    def tool_to_openai(tool_info: ServerToolInfo) -> Dict[str, Any]:
        """Convert ServerToolInfo to OpenAI format"""
        return tool_info.to_openai_format()
    
    @staticmethod
    def request_to_openai(request: LLMRequest) -> Dict[str, Any]:
        """Convert LLMRequest to OpenAI format"""
        return request.to_openai_format()
    
    @staticmethod
    def parse_openai_response(response: Any) -> LLMResponse:
        """Parse OpenAI response to LLMResponse"""
        return LLMResponse.from_openai_response(response)
    
    @staticmethod
    def parse_openai_tool_calls(tool_calls_data: List[Dict[str, Any]]) -> List[ToolCall]:
        """Parse OpenAI tool calls to ToolCall models"""
        return [ToolCall.from_openai_format(tc) for tc in tool_calls_data]


# Message history utilities
class MessageHistory:
    """Utilities for managing message history"""
    
    @staticmethod
    def add_tool_interaction(messages: List[Message], tool_call: ToolCall, result: str) -> None:
        """
        Add tool interaction to message history
        
        Args:
            messages: Conversation history
            tool_call: Tool call model
            result: Result of tool execution
        """
        try:
            # Create assistant message with tool call
            assistant_message = Message.assistant(tool_calls=[tool_call])
            
            # Create tool response message
            tool_message = Message.tool(tool_call_id=tool_call.id, content=result)
            
            # Add messages to history
            messages.append(assistant_message)
            messages.append(tool_message)
            
        except Exception as e:
            logger.error(f"Error updating message history: {e}")
            # Fall back to dictionary-based message creation if needed
            try:
                tool_call_message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": json.dumps(tool_call.arguments)
                            }
                        }
                    ]
                }
                
                tool_response_message = {
                    "role": "tool", 
                    "tool_call_id": tool_call.id,
                    "content": result
                }
                
                messages.append(tool_call_message)
                messages.append(tool_response_message)
            except Exception as nested_error:
                logger.error(f"Error in fallback message history update: {nested_error}")