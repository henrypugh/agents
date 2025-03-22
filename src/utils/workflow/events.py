"""
Event classes for the workflow system.

This module provides the base event class and essential event types for 
the event-driven workflow system.
"""

from datetime import datetime
from uuid import uuid4
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

class BaseEvent(BaseModel):
    """
    Base class for all events in the workflow system.
    
    Events are immutable data objects that trigger steps in a workflow.
    Each event has a unique ID and timestamp for tracking and debugging.
    """
    id: str = Field(default_factory=lambda: str(uuid4())[:8], description="Unique ID for this event")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this event was created")
    
    model_config = ConfigDict(frozen=True)  # Events are immutable
    
    def __str__(self) -> str:
        """String representation of the event"""
        return f"{self.__class__.__name__}(id={self.id})"


class StartEvent(BaseEvent):
    """
    Initiates a workflow with input data.
    
    A StartEvent is the entry point of any workflow. It contains input data
    that will be processed by the first step(s) of the workflow.
    """
    input: Dict[str, Any] = Field(default_factory=dict, description="Input data for the workflow")
    
    def __init__(self, **data):
        """Initialize with any keyword arguments as input data"""
        # If input isn't explicitly provided, use all kwargs as input
        if 'input' not in data:
            input_data = {k: v for k, v in data.items() if k not in ('id', 'timestamp')}
            super().__init__(input=input_data)
        else:
            super().__init__(**data)


class StopEvent(BaseEvent):
    """
    Completes a workflow with result data.
    
    A StopEvent signals the end of a workflow execution. It contains the final
    result of the workflow, which will be returned to the caller.
    """
    result: Any = None
    
    def __init__(self, result: Any = None, **data):
        """Initialize with a result value"""
        super().__init__(result=result, **data)


class ErrorEvent(BaseEvent):
    """
    Indicates an error occurred during workflow execution.
    
    An ErrorEvent contains information about an error that occurred during
    the execution of a workflow step. It can be used for error handling and
    recovery steps.
    """
    error_message: str
    error_type: str = Field(default="unknown", description="Type of error")
    step_name: Optional[str] = None
    traceback: Optional[str] = None
    
    @classmethod
    def from_exception(cls, exception: Exception, step_name: Optional[str] = None) -> 'ErrorEvent':
        """Create an ErrorEvent from an exception"""
        import traceback
        return cls(
            error_message=str(exception),
            error_type=exception.__class__.__name__,
            step_name=step_name,
            traceback=traceback.format_exc()
        )


class MessageEvent(BaseEvent):
    """
    Represents a message in conversation.
    
    A MessageEvent contains a message with content and role (e.g., user, assistant)
    that can be used in conversation workflows.
    """
    content: str
    role: str
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ToolEvent(BaseEvent):
    """
    Represents a tool call or tool result.
    
    A ToolEvent contains information about a tool to be executed or the result
    of a tool execution. It can be used to trigger tool execution steps or
    process tool results.
    """
    tool_name: str
    data: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments or results")
    is_result: bool = False
    server_name: Optional[str] = None
    
    @classmethod
    def tool_call(cls, tool_name: str, arguments: Dict[str, Any], server_name: Optional[str] = None) -> 'ToolEvent':
        """Create a ToolEvent for a tool call"""
        return cls(
            tool_name=tool_name,
            data=arguments,
            is_result=False,
            server_name=server_name
        )
    
    @classmethod
    def tool_result(cls, tool_name: str, result: Any) -> 'ToolEvent':
        """Create a ToolEvent for a tool result"""
        return cls(
            tool_name=tool_name,
            data={"result": result},
            is_result=True
        )


class Event(BaseEvent):
    """
    Generic event for custom data.
    
    This is a convenience class for creating custom events without having to
    create new subclasses. For production use, it's recommended to create
    proper subclasses for better type safety.
    """
    data: Dict[str, Any] = Field(default_factory=dict, description="Custom event data")
    event_type: str = Field("generic", description="Type identifier for this event")