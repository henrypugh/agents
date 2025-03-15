"""
Core data models for the MCP client.

This module provides standardized Pydantic models for data validation,
serialization, and documentation throughout the application.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import uuid
import json

class ToolArguments(BaseModel):
    """Model for tool arguments"""
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

class ToolCall(BaseModel):
    """Model for LLM-generated tool calls"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique ID for the tool call")
    tool_name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")
    server_name: Optional[str] = Field(None, description="Server that provides the tool")

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
            arguments = {}
            
        return cls(
            id=data.get("id", str(uuid.uuid4())[:8]),
            tool_name=data["function"]["name"],
            arguments=arguments,
            server_name=server_name
        )

class ToolResult(BaseModel):
    """Model for tool execution results"""
    tool_id: str = Field(..., description="ID of the tool call this result responds to")
    status: str = Field(..., description="Status of the tool execution (success, error, timeout)")
    result: Optional[Any] = Field(None, description="Result data from successful execution")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(0.0, description="Time taken to execute the tool in seconds")
    
    @validator('status')
    def validate_status(cls, v):
        if v not in ['success', 'error', 'timeout']:
            raise ValueError(f"Status must be 'success', 'error', or 'timeout', got '{v}'")
        return v

class Message(BaseModel):
    """Model for conversation messages"""
    role: str = Field(..., description="Role of the message sender (user, assistant, system, tool)")
    content: Optional[str] = Field(None, description="Content of the message")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls in the message")
    tool_call_id: Optional[str] = Field(None, description="ID of the tool call this message responds to")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this message was created")
    
    @validator('role')
    def validate_role(cls, v):
        valid_roles = ['user', 'assistant', 'system', 'tool']
        if v not in valid_roles:
            raise ValueError(f"Role must be one of {valid_roles}, got '{v}'")
        return v
    
    def is_tool_call(self) -> bool:
        """Check if this message contains tool calls"""
        return bool(self.tool_calls and len(self.tool_calls) > 0)
    
    def is_tool_response(self) -> bool:
        """Check if this message is a tool response"""
        return self.role == "tool" and bool(self.tool_call_id)
    
    @validator('tool_calls', 'content', always=True)
    def check_content_or_tool_calls(cls, v, values):
        """Ensure message has either content or tool calls for assistant messages"""
        if 'role' in values and values['role'] == 'assistant':
            if v is None and 'content' in values and values['content'] is None and 'tool_calls' in values and values['tool_calls'] is None:
                raise ValueError("Assistant messages must have either content or tool calls")
        return v
    
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

class ServerInfo(BaseModel):
    """Model for server information"""
    name: str = Field(..., description="Name of the server")
    connected: bool = Field(False, description="Whether server is currently connected")
    tools: List[str] = Field(default_factory=list, description="Available tools on this server")
    connected_at: Optional[datetime] = Field(None, description="When the server was connected")
    status: str = Field("disconnected", description="Current status of the server connection")
    
    @validator('status')
    def validate_status(cls, v):
        valid_statuses = ['connected', 'disconnected', 'error', 'connecting']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}, got '{v}'")
        return v

class ServerToolInfo(BaseModel):
    """Information about a tool available on a server"""
    name: str = Field(..., description="Name of the tool")
    description: Optional[str] = Field(None, description="Description of the tool") 
    input_schema: Dict[str, Any] = Field(..., description="Schema for tool inputs")
    server_name: str = Field(..., description="Name of the server providing this tool")
    
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

class LLMRequest(BaseModel):
    """Model for LLM API requests"""
    messages: List[Message] = Field(..., description="Conversation messages")
    model: str = Field(..., description="LLM model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="Tools available to the LLM")
    
    @validator('temperature')
    def validate_temperature(cls, v):
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

class LLMResponse(BaseModel):
    """Model for LLM API responses"""
    content: Optional[str] = Field(None, description="Generated text content")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="Tool calls generated by the LLM")
    finish_reason: str = Field(..., description="Reason why the generation finished")
    
    @classmethod
    def from_openai_response(cls, response) -> "LLMResponse":
        """Create from OpenAI API response"""
        choice = response.choices[0]
        message = choice.message
        
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = [
                ToolCall.from_openai_format(tc) 
                for tc in message.tool_calls
            ]
            
        return cls(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason
        )

class AgentConfig(BaseModel):
    """Configuration for the Agent"""
    model: str = Field("google/gemini-2.0-flash-001", description="LLM model to use")
    temperature: float = Field(0.7, description="Temperature for generation")
    max_tokens: Optional[int] = Field(None, description="Max tokens to generate") 
    retries: int = Field(3, description="Number of retries for LLM calls")
    timeout: float = Field(60.0, description="Timeout for LLM calls in seconds")
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        return v
    
    @validator('retries')
    def validate_retries(cls, v):
        if v < 0:
            raise ValueError("Retries must be non-negative")
        return v
    
    @validator('timeout')
    def validate_timeout(cls, v):
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v