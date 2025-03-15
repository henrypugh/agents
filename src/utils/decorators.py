# src/utils/decorators.py
"""
Role-aligned decorators for MCP client operations.

This module provides decorators that capture the role of each function,
standardizing error handling, tracing, and timeout behaviors.
"""

import asyncio
import functools
import hashlib
import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task, tool
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

logger = logging.getLogger(__name__)

def server_connection(
    workflow_name: Optional[str] = None,
    timeout_seconds: int = 60,
    max_retries: int = 2,
    retry_delay: float = 1.0
):
    """
    Decorator for server connection operations.
    
    Use this for methods that establish or manage connections to MCP servers.
    
    Args:
        workflow_name: Name for the workflow trace (defaults to function name)
        timeout_seconds: Maximum time for connection operation
        max_retries: Number of retry attempts for connection issues
        retry_delay: Delay between retry attempts in seconds
    """
    def decorator(func):
        # Apply Traceloop workflow decorator
        decorated_func = workflow(name=workflow_name or func.__name__)(func)
        
        @functools.wraps(decorated_func)
        async def wrapper(*args, **kwargs):
            retries = 0
            server_name = kwargs.get("server_name", 
                            getattr(args[1] if len(args) > 1 else None, "name", "unknown"))
            
            # Generate unique connection ID
            connection_id = f"{server_name}-{uuid.uuid4().hex[:6]}"
            
            # Set initial tracing properties
            Traceloop.set_association_properties({
                "connection_id": connection_id,
                "server_name": server_name,
                "operation": func.__name__,
                "connection_attempt": 0
            })
            
            while True:
                try:
                    # Update retry counter in tracing
                    Traceloop.set_association_properties({
                        "connection_attempt": retries + 1
                    })
                    
                    # Execute with timeout
                    start_time = asyncio.get_event_loop().time()
                    result = await asyncio.wait_for(
                        decorated_func(*args, **kwargs), 
                        timeout=timeout_seconds
                    )
                    
                    # Track success
                    elapsed = asyncio.get_event_loop().time() - start_time
                    Traceloop.set_association_properties({
                        "connection_status": "success",
                        "duration_seconds": round(elapsed, 3)
                    })
                    
                    return result
                    
                except (ConnectionError, asyncio.TimeoutError) as e:
                    retries += 1
                    if retries >= max_retries:
                        # Track failure after max retries
                        Traceloop.set_association_properties({
                            "connection_status": "failed",
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "max_retries_reached": True,
                            "total_attempts": retries
                        })
                        
                        logger.error(
                            f"Failed to connect to {server_name} after {retries} attempts: {str(e)}"
                        )
                        raise
                        
                    # Track retry
                    Traceloop.set_association_properties({
                        "retrying": True,
                        "retry_count": retries,
                        "retry_reason": type(e).__name__
                    })
                    
                    logger.info(
                        f"Retrying connection to {server_name} ({retries}/{max_retries})"
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(retry_delay)
                    
                except Exception as e:
                    # Track non-retriable error
                    Traceloop.set_association_properties({
                        "connection_status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                    
                    logger.error(
                        f"Error connecting to {server_name}: {str(e)}", 
                        exc_info=True
                    )
                    raise
                    
        return wrapper
    return decorator


def tool_execution(
    name: Optional[str] = None,
    timeout_seconds: int = 30,
    error_tracking: bool = True
):
    """
    Decorator for tool execution operations.
    
    Use this for methods that execute tools on MCP servers or process tool results.
    
    Args:
        name: Optional override for the tool name (defaults to function name)
        timeout_seconds: Maximum time in seconds for tool execution
        error_tracking: Whether to track execution errors in Traceloop
    """
    def decorator(func):
        # Apply Traceloop tool decorator
        decorated_func = tool(name=name or func.__name__)(func)
        
        @functools.wraps(decorated_func)
        async def wrapper(*args, **kwargs):
            # Generate execution context
            tool_name = name or func.__name__
            tool_args_preview = str({k: v for k, v in kwargs.items() if k != 'self'})[:50]
            execution_id = hashlib.md5(f"{tool_name}:{tool_args_preview}".encode()).hexdigest()[:12]
            
            # Get server name if available in args
            server_name = kwargs.get("server_name", "unknown")
            
            # Set up tracing context
            Traceloop.set_association_properties({
                "execution_id": execution_id,
                "tool_name": tool_name,
                "server_name": server_name,
                "args_count": len(kwargs) - (1 if 'self' in kwargs else 0),
                "args_preview": tool_args_preview
            })
            
            try:
                # Execute with timeout
                start_time = asyncio.get_event_loop().time()
                result = await asyncio.wait_for(
                    decorated_func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
                
                # Track success and duration
                elapsed = asyncio.get_event_loop().time() - start_time
                result_preview = str(result)[:50]
                
                if error_tracking:
                    Traceloop.set_association_properties({
                        "execution_status": "success",
                        "duration_seconds": round(elapsed, 3),
                        "result_preview": f"{result_preview}..." if len(str(result)) > 50 else result_preview
                    })
                
                return result
                
            except asyncio.TimeoutError:
                if error_tracking:
                    Traceloop.set_association_properties({
                        "execution_status": "timeout",
                        "error_type": "TimeoutError",
                        "timeout_seconds": timeout_seconds
                    })
                
                logger.error(f"Tool execution timed out after {timeout_seconds}s: {tool_name}")
                raise
                
            except Exception as e:
                if error_tracking:
                    Traceloop.set_association_properties({
                        "execution_status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    })
                
                logger.error(f"Error executing tool {tool_name}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def llm_completion(
    task_name: Optional[str] = None,
    track_inputs: bool = True,
    track_outputs: bool = True,
    timeout_seconds: int = 120
):
    """
    Decorator for LLM completion operations.
    
    Use this for methods that make direct calls to LLM APIs or process responses.
    
    Args:
        task_name: Name for the task trace (defaults to function name)
        track_inputs: Whether to track input messages in Traceloop
        track_outputs: Whether to track output text in Traceloop
        timeout_seconds: Maximum time for LLM operation in seconds
    """
    def decorator(func):
        # Apply Traceloop task decorator
        decorated_func = task(name=task_name or func.__name__)(func)
        
        @functools.wraps(decorated_func)
        async def wrapper(*args, **kwargs):
            # Extract messages if provided in kwargs
            messages = kwargs.get("messages", [])
            query_preview = ""
            
            # Find user message to preview in tracing
            if track_inputs and messages:
                for msg in messages:
                    if msg.get("role") == "user" and msg.get("content"):
                        content = msg.get("content", "")
                        query_preview = content[:50] + "..." if len(content) > 50 else content
                        break
            
            # Generate call ID
            completion_id = hashlib.md5((query_preview or str(uuid.uuid4())).encode()).hexdigest()[:12]
            
            # Set tracing properties
            Traceloop.set_association_properties({
                "completion_id": completion_id,
                "query_preview": query_preview,
                "message_count": len(messages),
                "has_tools": bool(kwargs.get("tools", []))
            })
            
            try:
                # Execute with timeout
                start_time = asyncio.get_event_loop().time()
                result = await asyncio.wait_for(
                    decorated_func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                
                # Calculate duration
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Track result in tracing if enabled
                if track_outputs and result:
                    response_text = ""
                    # Extract text from various potential response structures
                    if hasattr(result, 'output_text') and result.output_text:
                        response_text = result.output_text
                    elif hasattr(result, 'choices') and result.choices:
                        response_text = result.choices[0].message.content
                    elif hasattr(result, 'output') and result.output:
                        for item in result.output:
                            if hasattr(item, 'content'):
                                for content_item in item.content:
                                    if getattr(content_item, 'type', '') == 'output_text':
                                        response_text = content_item.text
                                        break
                    
                    # Track response preview and metrics
                    Traceloop.set_association_properties({
                        "completion_status": "success",
                        "duration_seconds": round(elapsed, 3),
                        "response_preview": response_text[:50] + "..." if len(response_text) > 50 else response_text,
                        "response_length": len(response_text) if response_text else 0
                    })
                
                return result
                
            except asyncio.TimeoutError:
                Traceloop.set_association_properties({
                    "completion_status": "timeout",
                    "error_type": "TimeoutError",
                    "timeout_seconds": timeout_seconds
                })
                
                logger.error(f"LLM completion timed out after {timeout_seconds}s")
                raise
                
            except Exception as e:
                # Track error
                Traceloop.set_association_properties({
                    "completion_status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                logger.error(f"Error in LLM completion: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def message_processing(
    workflow_name: Optional[str] = None,
    track_messages: bool = True,
    timeout_seconds: int = 180
):
    """
    Decorator for message processing operations.
    
    Use this for methods that handle conversation flow and message processing.
    
    Args:
        workflow_name: Name for the workflow trace (defaults to function name)
        track_messages: Whether to track message content in Traceloop
        timeout_seconds: Maximum time for processing operation in seconds
    """
    def decorator(func):
        # Apply Traceloop workflow decorator
        decorated_func = workflow(name=workflow_name or func.__name__)(func)
        
        @functools.wraps(decorated_func)
        async def wrapper(*args, **kwargs):
            # Extract messages or query if available
            messages = kwargs.get("messages", [])
            query = kwargs.get("query", "")
            
            # Generate processing ID
            processing_id = str(uuid.uuid4())[:8]
            
            # Get query preview for tracing
            if track_messages:
                if query:
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                elif messages:
                    # Find last user message
                    for msg in reversed(messages):
                        if msg.get("role") == "user" and msg.get("content"):
                            content = msg.get("content", "")
                            query_preview = content[:50] + "..." if len(content) > 50 else content
                            break
                    else:
                        query_preview = ""
                else:
                    query_preview = ""
            else:
                query_preview = ""
            
            # Set tracing properties
            Traceloop.set_association_properties({
                "processing_id": processing_id,
                "query_preview": query_preview,
                "message_count": len(messages) if messages else 0,
                "tools_available": len(kwargs.get("tools", []))
            })
            
            try:
                # Execute with timeout
                start_time = asyncio.get_event_loop().time()
                result = await asyncio.wait_for(
                    decorated_func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                
                # Calculate duration
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Track result
                result_preview = str(result)[:50] + "..." if len(str(result)) > 50 else str(result)
                Traceloop.set_association_properties({
                    "processing_status": "success",
                    "duration_seconds": round(elapsed, 3),
                    "result_preview": result_preview,
                    "result_length": len(str(result))
                })
                
                return result
                
            except asyncio.TimeoutError:
                Traceloop.set_association_properties({
                    "processing_status": "timeout",
                    "error_type": "TimeoutError",
                    "timeout_seconds": timeout_seconds
                })
                
                logger.error(f"Message processing timed out after {timeout_seconds}s")
                raise
                
            except Exception as e:
                # Track error
                Traceloop.set_association_properties({
                    "processing_status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                logger.error(f"Error in message processing: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def resource_cleanup(
    task_name: Optional[str] = None,
    track_resources: bool = True
):
    """
    Decorator for resource cleanup operations.
    
    Use this for methods that clean up resources like server connections.
    
    Args:
        task_name: Name for the task trace (defaults to function name)
        track_resources: Whether to track resource details in Traceloop
    """
    def decorator(func):
        # Apply Traceloop task decorator
        decorated_func = task(name=task_name or func.__name__)(func)
        
        @functools.wraps(decorated_func)
        async def wrapper(*args, **kwargs):
            # Get resource information if available
            instance = args[0] if args else None
            resource_count = 0
            
            if track_resources and instance:
                # Try to detect resources from instance attributes
                if hasattr(instance, 'servers'):
                    resource_count = len(instance.servers)
                    resource_names = ','.join(instance.servers.keys()) if instance.servers else ""
                elif hasattr(instance, 'connections'):
                    resource_count = len(instance.connections)
                    resource_names = ','.join(instance.connections.keys()) if instance.connections else ""
                else:
                    resource_names = ""
            else:
                resource_names = ""
            
            # Set tracing properties
            cleanup_id = str(uuid.uuid4())[:8]
            Traceloop.set_association_properties({
                "cleanup_id": cleanup_id,
                "resource_count": resource_count,
                "resource_names": resource_names
            })
            
            try:
                # Execute cleanup
                start_time = asyncio.get_event_loop().time()
                result = await decorated_func(*args, **kwargs)
                
                # Calculate duration
                elapsed = asyncio.get_event_loop().time() - start_time
                
                # Track successful cleanup
                Traceloop.set_association_properties({
                    "cleanup_status": "success",
                    "duration_seconds": round(elapsed, 3),
                    "resources_cleaned": resource_count
                })
                
                return result
                
            except Exception as e:
                # Track error
                Traceloop.set_association_properties({
                    "cleanup_status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                })
                
                logger.error(f"Error during resource cleanup: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator

def wrap_llm_call(func):
    """
    Helper decorator for wrapping track_llm_call with appropriate error handling.
    
    Use this around functions that already use track_llm_call.
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in LLM call: {str(e)}", exc_info=True)
            raise
    return wrapper