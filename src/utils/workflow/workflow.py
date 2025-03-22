"""
Workflow base class for defining event-driven workflows.

This module provides the Workflow base class and related classes for
defining and executing event-driven workflows.
"""

import logging
import asyncio
import inspect
import time
from typing import Dict, List, Any, Optional, Type, Union, Set, Awaitable, TypeVar, Generic

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

from .events import BaseEvent, StartEvent, StopEvent, ErrorEvent
from .context import WorkflowContext
from .step import Step

from typing import Any, AsyncIterator, Dict


logger = logging.getLogger(__name__)

# Delayed type annotation for Agent to avoid circular imports
AgentType = TypeVar('AgentType')

class WorkflowResult:
    """
    Represents the result of a workflow execution.
    
    This class provides access to the final result of the workflow,
    as well as methods for streaming events during execution.
    """
    
    def __init__(self, workflow_future: asyncio.Future, context: WorkflowContext):
        """
        Initialize the workflow result.
        
        Args:
            workflow_future: Future that will resolve to the final result
            context: Workflow context for accessing events
        """
        self.future = workflow_future
        self.ctx = context
        self.run_id = context.workflow_id
    
    def __await__(self):
        """
        Allow awaiting the result directly.
        
        Returns:
            The final result of the workflow
        """
        return self.future.__await__()
    
    async def stream_events(self) -> 'AsyncIterator[BaseEvent]':
        """
        Stream events from the workflow as they occur.
        
        This method returns an async iterator that yields events as they
        are produced by the workflow. The iterator stops when the workflow
        completes.
        
        Yields:
            Events from the workflow
        """
        # Register a stream with the context
        queue = self.ctx.register_stream()
        
        try:
            # Yield events until the workflow completes
            while not self.future.done():
                try:
                    # Wait for next event with timeout to check if workflow is done
                    event = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield event
                except asyncio.TimeoutError:
                    # No event available, check if workflow is done
                    continue
                    
            # Drain any remaining events
            while not queue.empty():
                event = queue.get_nowait()
                yield event
                
        finally:
            # Clean up the stream
            self.ctx.unregister_stream(queue)


class Workflow:
    """
    Base class for defining event-driven workflows.
    
    A Workflow consists of steps that are triggered by events. Each step
    can process an event and produce new events, which in turn trigger
    other steps.
    """
    
    def __init__(self, agent: AgentType, timeout: float = 60.0, verbose: bool = False):
        """
        Initialize the workflow.
        
        Args:
            agent: Agent that provides access to services
            timeout: Maximum execution time in seconds
            verbose: Whether to log detailed information
        """
        self.agent = agent
        self.timeout = timeout
        self.verbose = verbose
        self.ctx = WorkflowContext(agent)
        
        # Custom nested workflows
        self._custom_workflows = {}
        
        # Discover steps in this instance
        self.steps_by_event = self._discover_steps()
        
        # Ordered list of all steps (for debugging/visualization)
        self.all_steps = [step for steps in self.steps_by_event.values() for step in steps]
        
        if self.verbose:
            for event_cls, steps in self.steps_by_event.items():
                logger.info(f"Workflow {self.__class__.__name__} handles {event_cls.__name__} with steps: "
                           f"{[step.name for step in steps]}")
    
    def _discover_steps(self) -> Dict[Type[BaseEvent], List[Step]]:
        """
        Discover steps in this workflow instance.
        
        This looks for Step objects defined as attributes of this instance,
        as well as steps registered with the @step decorator.
        
        Returns:
            Dictionary mapping event types to lists of steps
        """
        steps_by_event: Dict[Type[BaseEvent], List[Step]] = {}
        
        # First, look for Step objects defined as attributes
        for attr_name in dir(self):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(self, attr_name)
            if isinstance(attr, Step):
                # Register this step for each of its input event types
                for event_type in attr.input_event_types:
                    if event_type not in steps_by_event:
                        steps_by_event[event_type] = []
                    steps_by_event[event_type].append(attr)
        
        # Then check for steps registered with @step decorator
        if hasattr(self.__class__, '_steps'):
            for step_obj in self.__class__._steps:
                # For each input event type, register the step
                for event_type in step_obj.input_event_types:
                    if event_type not in steps_by_event:
                        steps_by_event[event_type] = []
                    steps_by_event[event_type].append(step_obj)
        
        return steps_by_event
    
    def add_workflows(self, **workflows: 'Workflow') -> None:
        """
        Add custom workflows to be used by steps.
        
        This allows steps to delegate processing to nested workflows.
        
        Args:
            **workflows: Mapping of parameter names to workflow instances
        """
        self._custom_workflows.update(workflows)
    
    def _find_steps_for_event(self, event: BaseEvent) -> List[Step]:
        """
        Find steps that can handle the given event.
        
        A step can handle an event if the event type is a subclass of any
        of the step's input event types.
        
        Args:
            event: The event to handle
            
        Returns:
            List of steps that can handle the event
        """
        event_cls = event.__class__
        matching_steps = []
        
        # Check each registered event type
        for event_type, steps in self.steps_by_event.items():
            if issubclass(event_cls, event_type):
                matching_steps.extend(steps)
        
        return matching_steps
    
    @workflow(name="execute_workflow")
    async def handle_event(self, event: BaseEvent) -> List[BaseEvent]:
        """
        Handle an event by dispatching it to matching steps.
        
        This method finds all steps that can handle the event and executes them.
        It returns all events produced by those steps.
        
        Args:
            event: The event to handle
            
        Returns:
            List of events produced by the steps
        """
        # Record the event in context
        self.ctx.record_event(event)
        
        if self.verbose:
            logger.info(f"Handling event: {event}")
        
        # Find steps that can handle this event
        steps = self._find_steps_for_event(event)
        
        if not steps:
            if self.verbose:
                logger.info(f"No steps found to handle event: {event}")
            return []
        
        # Execute all matching steps with capacity
        all_events = []
        for step in steps:
            if step.has_capacity():
                if self.verbose:
                    logger.info(f"Running step {step.name}")
                    
                # Execute the step
                events = await step.execute(self.ctx, event)
                
                if self.verbose:
                    if events:
                        event_types = [e.__class__.__name__ for e in events]
                        logger.info(f"Step {step.name} produced events: {event_types}")
                    else:
                        logger.info(f"Step {step.name} produced no events")
                
                all_events.extend(events)
            else:
                if self.verbose:
                    logger.info(f"Step {step.name} is at capacity, skipping")
        
        return all_events
    
    async def _run_with_timeout(self, **input_data) -> Any:
        """
        Run the workflow with a timeout.
        
        Args:
            **input_data: Initial input data for the workflow
            
        Returns:
            The final result of the workflow
            
        Raises:
            asyncio.TimeoutError: If the workflow execution times out
        """
        # Create a start event with the input data
        start_event = StartEvent(input=input_data)
        
        # Process events until we reach a StopEvent or run out of events
        event_queue = [start_event]
        final_result = None
        
        # Track all seen event IDs to avoid processing duplicates
        seen_events: Set[str] = set()
        
        # Start time for timeout
        start_time = time.time()
        
        while event_queue:
            # Check for timeout
            if time.time() - start_time > self.timeout:
                logger.warning(f"Workflow {self.__class__.__name__} timed out after {self.timeout}s")
                raise asyncio.TimeoutError(f"Workflow execution timed out after {self.timeout}s")
            
            # Get the next event
            event = event_queue.pop(0)
            
            # Skip if we've seen this event before
            if event.id in seen_events:
                continue
                
            # Mark as seen
            seen_events.add(event.id)
            
            # Check for stop condition
            if isinstance(event, StopEvent):
                final_result = event.result
                break
            
            # Handle errors by propagating them
            if isinstance(event, ErrorEvent):
                logger.error(f"Error in workflow: {event.error_message} ({event.error_type})")
                if event.step_name:
                    logger.error(f"Step: {event.step_name}")
                if event.traceback:
                    logger.error(f"Traceback: {event.traceback}")
                    
                # Don't stop the workflow for errors unless configured to do so
                # final_result = event
                # break
            
            # Process the event and add resulting events to queue
            new_events = await self.handle_event(event)
            event_queue.extend(new_events)
        
        # Record metrics
        execution_time = time.time() - start_time
        Traceloop.set_association_properties({
            "workflow_execution_time": execution_time,
            "event_count": len(seen_events),
            "final_result_type": type(final_result).__name__ if final_result is not None else "None"
        })
        
        if self.verbose:
            logger.info(f"Workflow {self.__class__.__name__} completed in {execution_time:.2f}s "
                       f"with {len(seen_events)} events")
        
        return final_result
    
    def run(self, stepwise: bool = False, **input_data) -> WorkflowResult:
        """
        Run the workflow asynchronously.
        
        This method starts the workflow execution and returns a WorkflowResult
        that can be awaited to get the final result.
        
        Args:
            stepwise: Whether to run the workflow step by step
            **input_data: Initial input data for the workflow
            
        Returns:
            WorkflowResult object that resolves to the final result
        """
        # Create the future that will hold the result
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        
        if stepwise:
            # For stepwise execution, just create the result object
            # The caller will use run_step to advance execution
            self._stepwise_queue = [StartEvent(input=input_data)]
            self._stepwise_seen = set()
            self._stepwise_future = future
            
            # Create and return the result object
            handler = WorkflowResult(future, self.ctx)
            handler.run_step = self._run_step
            return handler
        else:
            # For normal execution, start the workflow immediately
            async def run_workflow():
                try:
                    result = await self._run_with_timeout(**input_data)
                    future.set_result(result)
                except Exception as e:
                    logger.exception(f"Error running workflow: {e}")
                    future.set_exception(e)
            
            # Start the workflow
            asyncio.create_task(run_workflow())
            
            # Return the result object
            return WorkflowResult(future, self.ctx)
    
    async def _run_step(self) -> List[BaseEvent]:
        """
        Run a single step of the workflow.
        
        This method is used for stepwise execution. It processes the next
        event in the queue and returns the events produced by the steps.
        
        Returns:
            List of events produced by the step
        """
        if not self._stepwise_queue:
            # No more events to process
            self._stepwise_future.set_result(None)
            return []
        
        # Get the next event
        event = self._stepwise_queue.pop(0)
        
        # Skip if we've seen this event before
        if event.id in self._stepwise_seen:
            return []
            
        # Mark as seen
        self._stepwise_seen.add(event.id)
        
        # Check for stop condition
        if isinstance(event, StopEvent):
            # Set the result and return empty
            self._stepwise_future.set_result(event.result)
            return []
        
        # Process the event
        new_events = await self.handle_event(event)
        
        # Add to queue for next step - these will be read but not processed
        # until the caller passes them to send_event
        return new_events