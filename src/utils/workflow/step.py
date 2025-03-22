"""
Step definition for workflow steps.

This module provides the Step class and step decorator for defining
workflow steps in a type-safe manner.
"""

import logging
import inspect
import time
from typing import Callable, Awaitable, Union, List, Dict, Type, TypeVar, get_type_hints, Any, Optional

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import task

from .events import BaseEvent, ErrorEvent
from .context import WorkflowContext

logger = logging.getLogger(__name__)

# Type for step functions
StepFunc = TypeVar('StepFunc', bound=Callable[..., Awaitable[Union[BaseEvent, List[BaseEvent], None]]])

class Step:
    """
    Defines a workflow step.
    
    A Step wraps a function that processes events and produces new events.
    Steps have input and output event types, which are used to determine
    when the step should be executed and how to handle its output.
    """
    
    def __init__(self, func: StepFunc, name: Optional[str] = None):
        """
        Initialize a step.
        
        Args:
            func: The async function to execute when the step is triggered
            name: Optional custom name for the step (defaults to function name)
        """
        self.func = func
        self.name = name or func.__name__
        self.num_workers = getattr(func, '_num_workers', 1)
        
        # Extract event types from function signature
        self.input_event_types = self._extract_input_event_types()
        self.output_event_types = self._extract_output_event_types()
        
        # Track current workers
        self._current_workers = 0
        
        logger.debug(f"Created step '{self.name}' accepting {[t.__name__ for t in self.input_event_types]} -> "
                     f"{[t.__name__ for t in self.output_event_types]}")
    
    def _extract_input_event_types(self) -> List[Type[BaseEvent]]:
        """
        Extract input event types from function signature.
        
        This looks at the type annotations of the 'event' parameter or
        any parameter that is a subclass of BaseEvent.
        
        Returns:
            List of input event types
        """
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)
        
        event_types = []
        
        # Find parameters that are BaseEvent subclasses
        for param_name, param in sig.parameters.items():
            if param_name == 'ctx' or param_name == 'self':
                continue
                
            if param_name in type_hints:
                param_type = type_hints[param_name]
                
                # Handle Union types
                if hasattr(param_type, '__origin__') and param_type.__origin__ is Union:
                    # Get all args from Union that are BaseEvent subclasses
                    for arg in param_type.__args__:
                        if isinstance(arg, type) and issubclass(arg, BaseEvent):
                            event_types.append(arg)
                
                # Handle direct BaseEvent subclasses
                elif isinstance(param_type, type) and issubclass(param_type, BaseEvent):
                    event_types.append(param_type)
        
        return event_types if event_types else [BaseEvent]
    
    def _extract_output_event_types(self) -> List[Type[BaseEvent]]:
        """
        Extract output event types from function return annotation.
        
        Returns:
            List of output event types
        """
        type_hints = get_type_hints(self.func)
        
        if 'return' in type_hints:
            return_type = type_hints['return']
            
            # Handle Union types in return annotation
            if hasattr(return_type, '__origin__') and return_type.__origin__ is Union:
                return [arg for arg in return_type.__args__ 
                        if isinstance(arg, type) and issubclass(arg, BaseEvent)]
            
            # Handle direct BaseEvent subclass
            elif isinstance(return_type, type) and issubclass(return_type, BaseEvent):
                return [return_type]
            
            # Handle optional returns (None | BaseEvent)
            elif return_type is None or return_type == type(None):
                return []
        
        # Default to BaseEvent if no annotation or non-BaseEvent annotation
        return [BaseEvent]
    
    def can_handle(self, event: BaseEvent) -> bool:
        """
        Check if this step can handle the given event.
        
        A step can handle an event if the event type is a subclass of any
        of the step's input event types.
        
        Args:
            event: The event to check
            
        Returns:
            True if the step can handle the event, False otherwise
        """
        event_cls = event.__class__
        return any(issubclass(event_cls, input_type) for input_type in self.input_event_types)
    
    def has_capacity(self) -> bool:
        """
        Check if this step has capacity to handle more events.
        
        Returns:
            True if the step has capacity, False otherwise
        """
        return self._current_workers < self.num_workers
    
    @task(name="execute_step")
    async def execute(self, ctx: WorkflowContext, event: BaseEvent) -> List[BaseEvent]:
        """
        Execute the step with the given event.
        
        This method is decorated with Traceloop's task decorator to provide
        observability for step execution.
        
        Args:
            ctx: The workflow context
            event: The event to process
            
        Returns:
            List of events produced by the step
        """
        if not self.can_handle(event):
            logger.warning(f"Step '{self.name}' cannot handle event of type {event.__class__.__name__}")
            return []
            
        if not self.has_capacity():
            logger.warning(f"Step '{self.name}' is at capacity, rejecting event {event}")
            return []
            
        # Set tracing properties
        Traceloop.set_association_properties({
            "step_name": self.name,
            "event_id": event.id,
            "event_type": event.__class__.__name__,
            "step_start_time": time.time()
        })
        
        self._current_workers += 1
        try:
            # Call the step function with appropriate arguments
            sig = inspect.signature(self.func)
            kwargs = {}
            
            # Determine what arguments to pass
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    # For methods, add self
                    kwargs['self'] = getattr(self, '_instance', None)
                elif param_name == 'ctx':
                    # Always pass the context
                    kwargs['ctx'] = ctx
                elif self.can_handle(event):
                    # Pass the event to the appropriately named parameter
                    kwargs[param_name] = event
            
            # Execute the step function
            start_time = time.time()
            result = await self.func(**kwargs)
            execution_time = time.time() - start_time
            
            # Process the result
            events = []
            if result is not None:
                if isinstance(result, list):
                    events.extend(result)
                else:
                    events.append(result)
            
            # Record tracing properties for completion
            Traceloop.set_association_properties({
                "step_status": "success",
                "execution_time": execution_time,
                "output_event_count": len(events),
                "output_event_types": [event.__class__.__name__ for event in events]
            })
            
            return events
            
        except Exception as e:
            logger.exception(f"Error executing step '{self.name}': {e}")
            
            # Create an error event
            error_event = ErrorEvent.from_exception(e, step_name=self.name)
            
            # Record tracing properties for error
            Traceloop.set_association_properties({
                "step_status": "error",
                "error_type": error_event.error_type,
                "error_message": error_event.error_message
            })
            
            return [error_event]
            
        finally:
            self._current_workers -= 1


def step(func: Optional[StepFunc] = None, *, 
         num_workers: int = 1,
         name: Optional[str] = None,
         workflow: Optional[Type] = None) -> Any:
    """
    Decorator for workflow steps.
    
    This decorator can be used in two ways:
    1. As a simple decorator: @step
    2. With parameters: @step(num_workers=5)
    
    It creates a Step object and attaches it to either:
    - The method's class (for bound methods)
    - The provided workflow class (for unbound functions)
    
    Args:
        func: The function to decorate
        num_workers: Maximum number of concurrent executions for this step
        name: Custom name for the step
        workflow: For unbound functions, the workflow class to attach to
        
    Returns:
        The decorated function or a decorator function
    """
    def decorator(fn: StepFunc) -> StepFunc:
        # Store num_workers on the function for Step to access
        fn._num_workers = num_workers
        
        # Create the Step object
        step_obj = Step(fn, name=name)
        
        if workflow is not None:
            # For unbound functions, attach to the provided workflow class
            if not hasattr(workflow, '_steps'):
                workflow._steps = []
            workflow._steps.append(step_obj)
            return fn
        else:
            # Return the Step object (which will handle __get__ for bound methods)
            return step_obj
    
    # Handle both @step and @step() syntax
    if func is None:
        return decorator
    else:
        return decorator(func)


# Add descriptor behavior to Step for class method support
def __get__(self, instance, owner):
    """
    Descriptor protocol method to support using @step on class methods.
    
    This allows the step to be accessed as both a class attribute and
    an instance method.
    """
    if instance is None:
        return self
    
    # Store a reference to the instance for later use
    self._instance = instance
    return self

# Add the __get__ method to Step class
Step.__get__ = __get__