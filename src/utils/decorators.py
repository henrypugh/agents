"""
Minimal decorators for standardizing error handling in tool execution.
"""

import functools
import logging
import traceback

logger = logging.getLogger("decorators")

def async_tool_error_handler(func):
    """
    Simple error handler for tool execution methods that adds errors to conversation.
    
    This decorator standardizes error handling for tool execution, ensuring that
    errors are properly logged and added to the conversation response.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Log the error with traceback for debugging
            logger.error(f"Tool execution error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            
            # Try to get the final_text list from the arguments to add the error message
            final_text = None
            for arg in args:
                if isinstance(arg, list) and all(isinstance(item, str) for item in arg if item):
                    final_text = arg
                    break
            
            # If we found the final_text list, add the error message
            if final_text is not None:
                final_text.append(f"Error executing tool: {str(e)}")
            
            # Re-raise the exception for proper error handling upstream
            raise
            
    return wrapper