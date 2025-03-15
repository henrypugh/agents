"""
Decorators for debugging function inputs and outputs.

This module provides decorators for detailed logging of function calls.
"""

import functools
import logging
import json
import asyncio
import time
from typing import Optional, List, Any

logger = logging.getLogger("decorators")

def debug_io(
    log_level: int = logging.DEBUG,
    max_length: int = 1000,
    exclude_args: Optional[List[str]] = None
):
    """
    Debug function inputs and outputs with detailed logging.
    
    Only logs at the specified level, so it can be left in production code
    and controlled via log level configuration.
    
    Args:
        log_level: Logging level to use (e.g., logging.DEBUG)
        max_length: Maximum length for logged values
        exclude_args: List of argument names to exclude from logging
    """
    exclude_args = exclude_args or []
    
    def _truncate(value: Any) -> str:
        """Truncate and format a value for logging"""
        str_value = str(value)
        if len(str_value) > max_length:
            return f"{str_value[:max_length]}... [truncated, total length: {len(str_value)}]"
        return str_value
    
    def _safe_repr(obj: Any) -> str:
        """Create a safe string representation of an object"""
        if obj is None:
            return "None"
        
        try:
            if isinstance(obj, (dict, list)):
                return _truncate(json.dumps(obj, default=str))
            elif isinstance(obj, (int, float, bool, str)):
                return _truncate(obj)
            elif hasattr(obj, "__dict__"):
                return f"{type(obj).__name__}({_truncate(str(obj.__dict__))})"
            else:
                return f"{type(obj).__name__}({_truncate(str(obj))})"
        except Exception as e:
            return f"<Error representing object: {str(e)}>"
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if logger.isEnabledFor(log_level):
                # Prepare args representation
                args_repr = []
                for i, arg in enumerate(args):
                    if i == 0 and arg.__class__.__name__ in func.__qualname__:
                        # This is likely the 'self' parameter
                        args_repr.append(f"self:{arg.__class__.__name__}")
                    else:
                        args_repr.append(_safe_repr(arg))
                
                # Prepare kwargs representation
                kwargs_repr = {
                    k: _safe_repr(v) for k, v in kwargs.items() 
                    if k not in exclude_args
                }
                
                # Log input
                logger.log(
                    log_level, 
                    f"CALL: {func.__module__}.{func.__qualname__}("
                    f"args={args_repr}, kwargs={kwargs_repr})"
                )
                
            # Call the function
            start_time = asyncio.get_event_loop().time()
            try:
                result = await func(*args, **kwargs)
                
                # Log output
                if logger.isEnabledFor(log_level):
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.log(
                        log_level,
                        f"RETURN: {func.__module__}.{func.__qualname__} "
                        f"(duration={duration:.6f}s) => {_safe_repr(result)}"
                    )
                
                return result
            except Exception as e:
                # Log exception
                if logger.isEnabledFor(log_level):
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.log(
                        log_level,
                        f"EXCEPTION: {func.__module__}.{func.__qualname__} "
                        f"(duration={duration:.6f}s) => {type(e).__name__}: {str(e)}"
                    )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar implementation for synchronous functions
            if logger.isEnabledFor(log_level):
                # Same args/kwargs representation as in async version
                args_repr = []
                for i, arg in enumerate(args):
                    if i == 0 and arg.__class__.__name__ in func.__qualname__:
                        args_repr.append(f"self:{arg.__class__.__name__}")
                    else:
                        args_repr.append(_safe_repr(arg))
                
                kwargs_repr = {
                    k: _safe_repr(v) for k, v in kwargs.items() 
                    if k not in exclude_args
                }
                
                logger.log(
                    log_level, 
                    f"CALL: {func.__module__}.{func.__qualname__}("
                    f"args={args_repr}, kwargs={kwargs_repr})"
                )
            
            # Call the function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                if logger.isEnabledFor(log_level):
                    duration = time.time() - start_time
                    logger.log(
                        log_level,
                        f"RETURN: {func.__module__}.{func.__qualname__} "
                        f"(duration={duration:.6f}s) => {_safe_repr(result)}"
                    )
                
                return result
            except Exception as e:
                if logger.isEnabledFor(log_level):
                    duration = time.time() - start_time
                    logger.log(
                        log_level,
                        f"EXCEPTION: {func.__module__}.{func.__qualname__} "
                        f"(duration={duration:.6f}s) => {type(e).__name__}: {str(e)}"
                    )
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator