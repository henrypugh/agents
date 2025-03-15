"""
Decorators for performance and timeout handling.

This module provides decorators for adding timeout protection to operations.
"""

import asyncio
import functools
import logging

logger = logging.getLogger("decorators")

def async_timeout(seconds=30):
    """
    Add timeout protection to async functions.
    
    Prevents operations from hanging indefinitely by applying a timeout.
    
    Args:
        seconds: Maximum seconds to wait before raising TimeoutError
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.error(
                    f"Operation {func.__name__} timed out after {seconds} seconds"
                )
                raise TimeoutError(
                    f"Operation {func.__name__} timed out after {seconds} seconds"
                )
        return wrapper
    return decorator