# Decorator Usage Guide

This guide shows how to effectively use the new decorators in your codebase. These decorators are designed to standardize common patterns, reduce code duplication, and improve error handling and observability.

## Import Patterns

```python
# For most files, use this import pattern:
from src.utils.decorators import async_error_handler, async_timeout, async_retry

# For files that need access to all decorators:
import src.utils.decorators as decorators
```

## Decorator Composition Order

When using multiple decorators, follow this order for best results:

```python
@workflow(name="operation_name")          # Framework decorators first
@trace_context(...)                       # Tracing second
@async_timeout(seconds=30)                # Timeout third
@async_retry(...)                         # Retry fourth
@async_error_handler(...)                 # Error handling last
@debug_io(...)                            # Debug at the innermost level
async def my_function(...):
    # Implementation
```

## Example Applications

### Error Handling Example

```python
# In src/client/conversation.py
from src.utils.decorators import async_error_handler

@task(name="process_tool_call")
@async_error_handler(output_param="final_text")
async def _process_tool_call(
    self,
    tool_call: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    final_text: List[str]
) -> None:
    """Process a single tool call"""
    # Implementation without try/except blocks
    tool_name = tool_call.function.name if hasattr(tool_call, 'function') else getattr(tool_call, 'name', 'unknown')
    server_name = self.tool_processor.find_server_for_tool(tool_name, tools)
    
    if not server_name or server_name not in self.server_manager.servers:
        final_text.append(f"Error: Can't determine which server handles tool '{tool_name}'.")
        return
    
    # Execute the tool
    await self._execute_and_process_tool(server_name, tool_call, messages, tools, final_text)
```

### Timeout Example

```python
# In src/client/server_registry.py
from src.utils.decorators import async_timeout

@workflow(name="connect_to_configured_server")
@async_timeout(seconds=45)  # Server connections may need more time
async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
    """Connect to a server from configuration"""
    # Implementation without manual timeout handling
    if server_name in self.servers:
        logger.info(f"Server {server_name} is already connected")
        tools = self.servers[server_name].get_tool_names()
        return {
            "status": "already_connected",
            "server": server_name,
            "tools": tools,
            "tool_count": len(tools)
        }
    
    server_config = self.config_manager.get_server_config(server_name)
    server_params = self._create_config_server_params(server_name, server_config)
    session = await self._create_server_session(server_params)
    server = ServerInstance(server_name, session)
    await server.initialize()
    self.servers[server_name] = server
    
    return {
        "status": "connected",
        "server": server_name,
        "tools": server.get_tool_names(),
        "tool_count": len(server.get_tool_names())
    }
```

### Retry Example

```python
# In src/client/llm_service.py
from src.utils.decorators import async_retry

@async_retry(
    max_retries=3,
    retry_on=[ConnectionError, TimeoutError],
    retry_delay=1.0,
    backoff_factor=2.0
)
async def get_completion(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
    """Get a completion from the LLM API"""
    logger.info("Making API call to LLM")
    
    # Convert the chat messages for the responses API
    input_content, instructions = self._prepare_input_from_messages(messages)
    
    # Prepare request parameters
    request_params = {
        "model": self.model,
        "input": input_content,
    }
    
    # Add tools if provided
    if tools:
        request_params["tools"] = tools
    
    # Add instructions if available
    if instructions:
        request_params["instructions"] = instructions
    
    # Make the API call without retry logic
    response = self.client.responses.create(**request_params)
    
    return response
```

### Tracing Context Example

```python
# In src/client/simple_agent.py
from src.utils.decorators import trace_context

def _plan_context_generator(self, task: str) -> Dict[str, Any]:
    """Generate context information for plan creation"""
    return {
        "task_id": hashlib.md5(task.encode()).hexdigest()[:12],
        "task_preview": task[:50] + "..." if len(task) > 50 else task,
        "task_length": len(task)
    }

@task(name="create_plan")
@trace_context(context_generator=_plan_context_generator)
async def _create_plan(self, task: str) -> Dict[str, Any]:
    """Create a plan for the given task"""
    # Get available tools and servers
    available_tools = []
    for server_name, server in self.server_manager.servers.items():
        if hasattr(server, 'get_tool_names'):
            tools = server.get_tool_names()
            for tool in tools:
                available_tools.append(f"{tool} (from {server_name})")
    
    # Create planning prompt
    prompt = f"""
    I need you to create a plan to complete this task: {task}
    
    Available tools: {', '.join(available_tools)}
    
    Create a detailed plan with specific steps...
    """
    
    # Get the plan from LLM
    messages = [{"role": "user", "content": prompt}]
    response = await self.llm_client.get_completion(messages, [])
    plan_text = response.choices[0].message.content
    
    # Parse plan as JSON
    try:
        plan = json.loads(plan_text)
        return plan
    except json.JSONDecodeError:
        return {"steps": []}
```

### Debug I/O Example

```python
# In src/client/server_registry.py
from src.utils.decorators import debug_io

@workflow(name="connect_to_configured_server")
@debug_io(exclude_args=["api_key", "credentials"])
async def connect_to_configured_server(self, server_name: str) -> Dict[str, Any]:
    """Connect to a server from configuration"""
    # Implementation without manual debug logging
    # ...the same implementation as above
```

## When to Use Each Decorator

1. **Use `async_error_handler` for**:
   - Functions that interact with external systems
   - Functions that modify shared state
   - Functions that need standardized error handling

2. **Use `async_timeout` for**:
   - Network operations
   - External API calls
   - Operations that could potentially hang

3. **Use `async_retry` for**:
   - Operations with transient failures
   - External API calls
   - Network operations

4. **Use `trace_context` for**:
   - High-level operations that need detailed tracing
   - Complex workflows
   - Operations where timing and context are important

5. **Use `debug_io` for**:
   - Complex functions during development
   - Integration points between components
   - Functions with complex input/output relationships

## Benefits of Using These Decorators

1. **Reduced Code Duplication**
   - Eliminates repetitive try/except blocks
   - Standardizes common patterns across the codebase

2. **Improved Readability**
   - Business logic becomes more visible without being cluttered by error handling
   - Decorators clearly signal the function's behavior

3. **Better Error Handling**
   - Ensures consistent error logging
   - Standardizes how errors are reported to users

4. **Enhanced Observability**
   - Standardizes tracing patterns
   - Provides consistent timing information

5. **More Robust Code**
   - Adds timeouts to prevent operations from hanging
   - Implements retry for resilience against transient failures