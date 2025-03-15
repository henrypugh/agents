# schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ToolArguments(BaseModel):
    args: Dict[str, Any]


class ToolCall(BaseModel):
    tool_name: str
    arguments: ToolArguments
    server_name: str


class ToolResult(BaseModel):
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None


class ServerConnectionStatus(BaseModel):
    status: str
    server: str
    message: Optional[str] = None
    error: Optional[str] = None


class LLMMessage(BaseModel):
    role: str
    content: Optional[str] = None


class ServerInfo(BaseModel):
    server_name: str
    connected: bool
    tools: List[str] = Field(default_factory=list)
    connected_at: datetime


class ToolExecutionResult(BaseModel):
    tool_name: str
    server: str
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[Any] = None
    error: Optional[str] = None