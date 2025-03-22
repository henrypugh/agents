"""
Workflow context for maintaining state and service access.

This module provides the WorkflowContext class which maintains state and
provides access to services for workflow steps.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set, TypeVar, Type, Union, Generic
import asyncio
from datetime import datetime

from traceloop.sdk import Traceloop

# Import event types
from .events import BaseEvent

logger = logging.getLogger(__name__)

# Delayed type annotation for Agent to avoid circular imports
AgentType = TypeVar('AgentType')

class WorkflowContext:
    """
    Maintains state and service access for workflows.
    
    The context object provides:
    - A key-value store for state data
    - Access to agent services
    - Event history tracking
    - Streaming of events to listeners
    """
    
    def __init__(self, agent: AgentType):
        """
        Initialize the workflow context.
        
        Args:
            agent: The agent instance that provides access to services
        """
        self.agent = agent
        self.data: Dict[str, Any] = {}  # Key-value store for workflow state
        self.events: List[BaseEvent] = []  # History of events
        self._event_streams: List[asyncio.Queue] = []  # Streams for event listeners
        self._start_time = time.time()
        
        # Set tracking properties
        self.workflow_id = f"wf_{int(self._start_time)}"
        Traceloop.set_association_properties({
            "workflow_id": self.workflow_id,
            "start_time": self._start_time
        })
        
        logger.debug(f"Created workflow context {self.workflow_id}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the context.
        
        Args:
            key: The key to get
            default: The default value if the key doesn't exist
            
        Returns:
            The value associated with the key, or the default if not found
        """
        return self.data.get(key, default)
    
    async def set(self, key: str, value: Any) -> None:
        """
        Set a value in the context.
        
        Args:
            key: The key to set
            value: The value to associate with the key
        """
        self.data[key] = value
    
    def record_event(self, event: BaseEvent) -> None:
        """
        Record an event in history and send to streams.
        
        Args:
            event: The event to record
        """
        self.events.append(event)
        
        # Stream event to all listeners
        for stream in self._event_streams:
            try:
                stream.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event stream queue full, dropping event {event}")
        
        # Track the event in Traceloop
        Traceloop.set_association_properties({
            "event_type": event.__class__.__name__,
            "event_id": event.id,
            "event_timestamp": event.timestamp.isoformat()
        })
    
    def register_stream(self) -> asyncio.Queue:
        """
        Register a new event stream.
        
        Returns:
            An asyncio Queue that will receive events
        """
        queue = asyncio.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self._event_streams.append(queue)
        return queue
    
    def unregister_stream(self, queue: asyncio.Queue) -> None:
        """
        Unregister an event stream.
        
        Args:
            queue: The queue to unregister
        """
        if queue in self._event_streams:
            self._event_streams.remove(queue)
    
    # Convenience properties to access agent services
    
    @property
    def llm_client(self):
        """Get the LLM service from the agent"""
        return self.agent.llm_client
    
    @property
    def server_manager(self):
        """Get the server manager from the agent"""
        return self.agent.server_manager
    
    @property
    def tool_processor(self):
        """Get the tool processor from the agent"""
        return self.agent.tool_processor
    
    @property
    def conversation_manager(self):
        """Get the conversation manager from the agent"""
        return self.agent.conversation_manager
    
    # Event collection helpers
    
    async def collect_events(
        self, 
        trigger_event: BaseEvent,
        event_types: List[Type[BaseEvent]],
        timeout: float = 60.0
    ) -> Optional[List[BaseEvent]]:
        """
        Collect a specific set of events.
        
        This is useful for gathering results from multiple parallel steps.
        
        Args:
            trigger_event: The event that triggered the collection
            event_types: List of event types to collect (in order)
            timeout: Maximum time to wait for all events
            
        Returns:
            List of collected events in the requested order, or None if still collecting
        """
        # Create a unique collection key based on step and event types
        collection_key = f"collect_{trigger_event.id}_{'_'.join(et.__name__ for et in event_types)}"
        
        # Get existing collection or create a new one
        collection = await self.get(collection_key, {
            "events": {},  # Events collected so far, by type name
            "start_time": time.time(),
            "complete": False,
            "result": None
        })
        
        # If already complete, return the result
        if collection["complete"]:
            return collection["result"]
        
        # Check if we have events of all required types
        events_dict = collection["events"]
        
        # Add the trigger event to our collection if it matches any requested type
        trigger_event_cls = trigger_event.__class__
        for i, event_type in enumerate(event_types):
            if issubclass(trigger_event_cls, event_type):
                type_name = event_type.__name__
                if type_name not in events_dict:
                    events_dict[type_name] = trigger_event
                break
        
        # Check if we have all the events we need
        all_collected = all(et.__name__ in events_dict for et in event_types)
        
        # Check for timeout
        timed_out = (time.time() - collection["start_time"]) > timeout
        
        if all_collected or timed_out:
            # We're done collecting
            collection["complete"] = True
            
            if all_collected:
                # Return events in the requested order
                result = [events_dict[et.__name__] for et in event_types]
                collection["result"] = result
                await self.set(collection_key, collection)
                return result
            else:
                # Timed out without collecting all events
                logger.warning(f"Collection {collection_key} timed out after {timeout}s")
                collection["result"] = None
                await self.set(collection_key, collection)
                return None
        
        # Not done collecting yet
        await self.set(collection_key, collection)
        return None
    
    def send_event(self, event: BaseEvent) -> None:
        """
        Send an event to the workflow.
        
        This records the event but doesn't immediately process it.
        The workflow engine will pick it up on the next cycle.
        
        Args:
            event: The event to send
        """
        self.record_event(event)
        
    # Stream event helper
    def write_event_to_stream(self, event: BaseEvent) -> None:
        """
        Write an event to all registered streams.
        
        This is used for streaming partial results to listeners.
        Unlike send_event, this doesn't add the event to the workflow for processing.
        
        Args:
            event: The event to write to streams
        """
        # Only stream the event, don't add to workflow history
        for stream in self._event_streams:
            try:
                stream.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(f"Event stream queue full, dropping event {event}")