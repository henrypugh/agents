"""
Decorators for resilience and retry capabilities.

This module provides decorators for adding retry mechanisms to functions
that interact with external services.
"""

import asyncio
import functools
import logging
from typing import Type, List, Optional, Union

logger = logging.getLogger("decorators")

def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retry_on: Optional[Union[Type[Exception], List[Type[Exception]]]] = None,
    retry_if: Optional[callable] = None
):
    """
    Add retry capability to async functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        retry_on: Exception types to retry on (defaults to Exception)
        retry_if: Optional function that takes the exception and returns 
                 True if operation should be retried
    """
    retry_on = retry_on or Exception
    if not isinstance(retry_on, (list, tuple)):
        retry_on = [retry_on]
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except tuple(retry_on) as e:
                    last_exception = e
                    
                    # Check if we should retry based on custom function
                    should_retry = True
                    if retry_if is not None:
                        should_retry = retry_if(e)
                    
                    if attempt < max_retries and should_retry:
                        logger.warning(
                            f"Retry {attempt+1}/{max_retries} for {func.__name__} "
                            f"after error: {str(e)}"
                        )
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        if not should_retry:
                            logger.error(
                                f"Not retrying {func.__name__} based on retry_if"
                            )
                        else:
                            logger.error(
                                f"Max retries ({max_retries}) reached for {func.__name__}"
                            )
                        raise
                except Exception as e:
                    # Don't retry exceptions not in retry_on
                    logger.error(f"Non-retryable error in {func.__name__}: {str(e)}")
                    raise
            
            # This should never be reached, but just in case
            raise last_exception
            
        return wrapper
    return decorator