"""
Decorators for standardized error handling.

This module provides decorators for consistent error handling across the codebase.
"""

import functools
import logging
import traceback
from typing import Optional, List, Callable, Any

logger = logging.getLogger("decorators")

def async_error_handler(
    error_handler: Optional[Callable] = None, 
    output_param: str = "final_text"
):
    """
    Standardized error handling for async functions.
    
    Ensures errors are properly logged and optionally added to an output parameter.
    
    Args:
        error_handler: Optional custom error handling function
        output_param: Name of the parameter to add error messages to (if it exists)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log the error with traceback
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                
                # Add error to output parameter if it exists
                if output_param in kwargs and isinstance(kwargs[output_param], list):
                    kwargs[output_param].append(f"Error in {func.__name__}: {str(e)}")
                else:
                    # Try to find a list parameter in args
                    for arg in args:
                        if isinstance(arg, list) and all(isinstance(item, str) for item in arg if item):
                            arg.append(f"Error in {func.__name__}: {str(e)}")
                            break
                
                # Custom error handling if provided
                if error_handler:
                    return await error_handler(e, *args, **kwargs)
                
                # Re-raise the exception
                raise
                
        return wrapper
    return decorator

# For backward compatibility with any existing code
async_tool_error_handler = async_error_handler