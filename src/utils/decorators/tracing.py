"""
Decorators for standardizing tracing and context management.

This module provides decorators for consistent use of Traceloop tracing.
"""

import functools
import hashlib
import time
import asyncio
import inspect
from typing import Dict, Any, Optional, Callable
from traceloop.sdk import Traceloop

def trace_context(
    context_generator: Optional[Callable[..., Dict[str, Any]]] = None,
    operation_type: Optional[str] = None,
    include_args: bool = False
):
    """
    Standardize Traceloop context tracing for functions.
    
    Automatically sets operation context at start, tracks timing, 
    and handles error reporting consistently.
    
    Args:
        context_generator: Optional function that takes func args and returns 
                           additional context properties
        operation_type: Type of operation (defaults to function name)
        include_args: Whether to include function args in the context
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create a unique operation ID
            op_type = operation_type or func.__name__
            op_id = hashlib.md5(f"{op_type}:{time.time()}".encode()).hexdigest()[:12]
            start_time = time.time()
            
            # Create initial context properties
            context = {
                "operation_id": op_id,
                "operation_type": op_type,
                "status": "started",
                "start_time": start_time
            }
            
            # Include args if requested
            if include_args:
                # Get parameter names
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Match args to param names, skip 'self'
                if param_names and param_names[0] == 'self':
                    arg_dict = dict(zip(param_names[1:], args[1:]))
                else:
                    arg_dict = dict(zip(param_names, args))
                    
                # Add kwargs
                arg_dict.update(kwargs)
                
                # Add safe string representations of args to context
                for key, value in arg_dict.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        context[f"arg_{key}"] = value
                    elif hasattr(value, "__len__"):
                        context[f"arg_{key}_length"] = len(value)
                    else:
                        context[f"arg_{key}_type"] = type(value).__name__
            
            # Add custom context properties if provided
            if context_generator:
                try:
                    custom_context = context_generator(*args, **kwargs)
                    if custom_context:
                        context.update(custom_context)
                except Exception as e:
                    # Don't let context generation errors break functionality
                    context["context_generation_error"] = str(e)
            
            # Set initial context
            Traceloop.set_association_properties(context)
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Set completion context
                Traceloop.set_association_properties({
                    "operation_id": op_id,
                    "status": "completed",
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })
                
                return result
            except Exception as e:
                # Set error context
                Traceloop.set_association_properties({
                    "operation_id": op_id,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for synchronous functions
            op_type = operation_type or func.__name__
            op_id = hashlib.md5(f"{op_type}:{time.time()}".encode()).hexdigest()[:12]
            start_time = time.time()
            
            # Create initial context properties
            context = {
                "operation_id": op_id,
                "operation_type": op_type,
                "status": "started",
                "start_time": start_time
            }
            
            # Include args if requested
            if include_args:
                # Get parameter names
                sig = inspect.signature(func)
                param_names = list(sig.parameters.keys())
                
                # Match args to param names, skip 'self'
                if param_names and param_names[0] == 'self':
                    arg_dict = dict(zip(param_names[1:], args[1:]))
                else:
                    arg_dict = dict(zip(param_names, args))
                    
                # Add kwargs
                arg_dict.update(kwargs)
                
                # Add safe string representations of args to context
                for key, value in arg_dict.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        context[f"arg_{key}"] = value
                    elif hasattr(value, "__len__"):
                        context[f"arg_{key}_length"] = len(value)
                    else:
                        context[f"arg_{key}_type"] = type(value).__name__
            
            # Add custom context properties if provided
            if context_generator:
                try:
                    custom_context = context_generator(*args, **kwargs)
                    if custom_context:
                        context.update(custom_context)
                except Exception as e:
                    # Don't let context generation errors break functionality
                    context["context_generation_error"] = str(e)
            
            # Set initial context
            Traceloop.set_association_properties(context)
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                
                # Set completion context
                Traceloop.set_association_properties({
                    "operation_id": op_id,
                    "status": "completed",
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })
                
                return result
            except Exception as e:
                # Set error context
                Traceloop.set_association_properties({
                    "operation_id": op_id,
                    "status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "end_time": time.time(),
                    "duration": time.time() - start_time
                })
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator