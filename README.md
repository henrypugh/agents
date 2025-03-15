# MCP Client - Developer Reference

This document serves as a personal reference for ongoing development of the MCP Client project. It documents the architecture, code structure, implementation details, and development notes to help quickly re-orient when returning to the codebase.

## Architecture Overview

The project follows a modular architecture organized around these core components:

```
┌─────────────────────────────────────────┐
│             Agent (main)                │
└─┬─────────────────┬──────────────────┬──┘
  │                 │                  │
  ▼                 ▼                  ▼
┌─────────┐   ┌───────────────┐   ┌──────────────┐
│LLMService│   │ServerRegistry │   │Conversation  │
└─────────┘   └───────┬───────┘   └───────┬──────┘
                      │                   │
                      ▼                   │
              ┌────────────────┐         │
              │ServerInstance  │◄────────┘
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐         ┌────────────────┐
              │  MCP Servers   │────────►│External APIs   │
              └────────────────┘         └────────────────┘
```

### Components and Interactions

1. **Main Module (`main.py`)**
   - Entry point that parses CLI args and initializes core components
   - Manages pre-connection to servers and the main chat loop

2. **Agent (`src/client/agent.py`)**
   - Core orchestrator that delegates to specialized components
   - Maintains the chat loop and handles user interactions
   - Exposes public API for connecting to servers and processing queries

3. **Conversation (`src/client/conversation/conversation.py`)**
   - Manages LLM interaction and conversation flow
   - Processes queries through the LLM
   - Delegates to specialized handlers for different aspects of conversation
   - Coordinates message processing, response handling, and server management

4. **MessageProcessor (`src/client/conversation/message_processor.py`)**
   - Handles message formatting and LLM interactions
   - Manages conversation history and context
   - Runs the core conversation loop with the LLM

5. **ResponseProcessor (`src/client/conversation/response_processor.py`)**
   - Processes LLM responses and handle tool execution
   - Routes tool calls to appropriate handlers
   - Manages follow-up responses after tool execution

6. **ServerManagementHandler (`src/client/conversation/server_management.py`)**
   - Handles server-related tools and operations
   - Creates server management tools for LLM use
   - Processes server connection and discovery requests

7. **ServerRegistry (`src/client/server_registry.py`)**
   - Manages creation and lifecycle of server connections
   - Stores server connection instances
   - Handles discovery of available servers
   - Collects tools from all connected servers

8. **ToolExecutor (`src/client/tool_processor.py`)**
   - Processes tool calls from the LLM
   - Routes tools to appropriate servers
   - Handles tool execution and error handling

9. **ServerInstance (`src/client/server_instance.py`)**
   - Encapsulates connection to individual MCP servers
   - Handles initialization and tool discovery
   - Executes tool calls on specific servers

10. **LLMService (`src/client/llm_service.py`)**
    - Handles communication with the OpenRouter API
    - Formats messages and tools for the LLM
    - Processes LLM responses

11. **ServerConfig (`src/client/server_config.py`)**
    - Manages server configurations and environment variables
    - Handles loading and parsing of server configuration files
    - Processes environment variables for server connections

12. **SimpleAgent (`simple_agent.py`)**
    - Self-directed agent for code analysis tasks
    - Creates and executes plans using available tools
    - Generates recommendations based on findings

13. **Agent Runner (`agent_runner.py`)**
    - Runner script for SimpleAgent
    - Initializes and runs the SimpleAgent for code analysis tasks
    - Processes command line arguments and task inputs

## Refactoring History & Design Decisions

### Initial Implementation
- Started with monolithic architecture in a single `mcp_client.py` file
- Implemented basic functionality for connecting to MCP servers
- Added LLM integration with OpenRouter

### First Refactoring
- Split monolithic design into specialized components
- Introduced `ServerRegistry` to handle server connection lifecycle
- Created `Conversation` to handle LLM interaction
- Added `ToolExecutor` to handle tool routing and execution
- Separated `ServerInstance` to encapsulate individual server connections
- Updated `Agent` to orchestrate these components

### Second Refactoring
- Renamed components for clarity and consistency
- Added `ServerConfig` for better configuration management
- Implemented `SimpleAgent` for self-directed code analysis tasks
- Added observability with Traceloop integration
- Improved error handling and recovery mechanisms
- Enhanced tool execution flow

### Third Refactoring
- Decomposed `Conversation` into specialized subcomponents
- Created `MessageProcessor` for handling message formatting and LLM interactions
- Added `ResponseProcessor` for processing LLM responses and tool execution
- Implemented `ServerManagementHandler` for server-related operations
- Enhanced tracing and observability throughout the codebase
- Added resource cleanup decorators for better resource management
- Improved server connection handling with better error recovery

### Key Design Decisions

1. **Component Separation**
   - Each component has single responsibility
   - Components communicate through well-defined interfaces
   - Makes testing and future changes easier

2. **Dependency Injection**
   - Components receive dependencies through constructor
   - Allows for easier testing and flexibility
   - Avoids tight coupling between components

3. **Async Design**
   - Using asyncio throughout codebase
   - Allows for efficient handling of multiple server connections
   - Enables non-blocking I/O for LLM and server communication

4. **Error Handling Strategy**
   - Exception handling at appropriate levels
   - Lower-level components propagate errors
   - Higher-level components handle display and recovery
   - Decorators for consistent error handling patterns

5. **Configuration Management**
   - Server configurations stored in external JSON
   - Environment variables for sensitive information
   - Server configs dynamically processed at runtime

6. **Observability**
   - Traceloop integration for tracing and monitoring
   - Consistent logging throughout the codebase
   - Detailed metrics on operations and performance
   - Association properties for context tracking

7. **Resource Management**
   - AsyncExitStack for managing async resources
   - Resource cleanup decorators for consistent cleanup
   - Server-specific resource management
   - Proper cleanup on application exit

## Component Breakdown

### Agent
- **Purpose**: Main entry point and coordinator
- **Responsibilities**:
  - Initialize components
  - Manage server connections
  - Process user queries
  - Handle chat loop
- **Key Methods**:
  - `connect_to_server()`: Connect to server by script path
  - `connect_to_configured_server()`: Connect to named server
  - `process_query()`: Process user query with LLM and tools
  - `chat_loop()`: Run interactive chat loop
  - `cleanup()`: Clean up resources

### Conversation
- **Purpose**: Manage LLM conversations
- **Responsibilities**:
  - Coordinate message processing and response handling
  - Delegate to specialized handlers
  - Process queries through LLM
  - Manage server tools and tool execution
- **Key Methods**:
  - `process_query()`: Process user query
  - `_initialize_components()`: Set up conversation components
  - `_collect_tools()`: Collect tools from servers and add server management tools

### MessageProcessor
- **Purpose**: Handle message formatting and LLM interactions
- **Responsibilities**:
  - Format messages for LLM
  - Run conversation with LLM
  - Update message history
- **Key Methods**:
  - `run_conversation()`: Run conversation with LLM
  - `update_message_history()`: Update conversation history
  - `format_messages()`: Format messages for LLM

### ResponseProcessor
- **Purpose**: Process LLM responses and handle tool execution
- **Responsibilities**:
  - Process LLM responses
  - Handle tool calls
  - Get follow-up responses
- **Key Methods**:
  - `process_response()`: Process LLM response
  - `process_tool_call()`: Process tool call from LLM
  - `get_follow_up_response()`: Get follow-up response after tool execution

### ServerManagementHandler
- **Purpose**: Handle server-related tools and operations
- **Responsibilities**:
  - Create server management tools
  - Handle server connection requests
  - Process server discovery requests
- **Key Methods**:
  - `create_server_management_tools()`: Create server management tools
  - `handle_server_management_tool()`: Handle server management tool calls
  - `_handle_list_available_servers()`: Handle list_available_servers tool
  - `_handle_connect_to_server()`: Handle connect_to_server tool

### ServerRegistry
- **Purpose**: Manage server connections
- **Responsibilities**:
  - Create and store server connections
  - Initialize server sessions
  - Discover available servers
  - Collect tools from servers
- **Key Methods**:
  - `connect_to_server()`: Connect to server by script path
  - `connect_to_configured_server()`: Connect to configured server
  - `disconnect_server()`: Disconnect from a server
  - `get_server()`: Get server connection by name
  - `collect_all_tools()`: Collect tools from all servers
  - `get_connected_servers()`: Get info about connected servers
  - `get_available_servers()`: Get all available servers
  - `cleanup()`: Clean up server connections

### ToolExecutor
- **Purpose**: Process tool calls
- **Responsibilities**:
  - Find server for tool
  - Execute tool on appropriate server
  - Extract results from tool execution
- **Key Methods**:
  - `find_server_for_tool()`: Find server for tool
  - `execute_tool()`: Execute tool on server
  - `extract_result_text()`: Extract text from tool result
  - `process_external_tool()`: Process external tool call

### ServerInstance
- **Purpose**: Encapsulate connection to MCP server
- **Responsibilities**:
  - Maintain connection to server
  - Discover tools from server
  - Execute tools on server
  - Format tools for LLM
- **Key Methods**:
  - `initialize()`: Initialize connection to server
  - `refresh_tools()`: Refresh list of tools
  - `execute_tool()`: Execute tool on server
  - `get_tool_names()`: Get list of tool names
  - `get_openai_format_tools()`: Get tools in OpenAI format
  - `cleanup()`: Clean up server resources

### LLMService
- **Purpose**: Communicate with LLM API
- **Responsibilities**:
  - Format messages and tools for LLM
  - Send requests to OpenRouter API
  - Process LLM responses
- **Key Methods**:
  - `get_completion()`: Get completion from LLM

### ServerConfig
- **Purpose**: Manage server configurations
- **Responsibilities**:
  - Load and parse configuration files
  - Retrieve server configurations
  - Process environment variables
- **Key Methods**:
  - `load_config()`: Load server configuration file
  - `get_server_config()`: Get configuration for specific server
  - `process_environment_variables()`: Process environment variables

### SimpleAgent
- **Purpose**: Self-directed agent for code analysis
- **Responsibilities**:
  - Create plans for tasks
  - Execute plans using available tools
  - Generate recommendations
- **Key Methods**:
  - `execute_task()`: Execute a self-directed task
  - `_create_plan()`: Create a plan for a task
  - `_execute_plan()`: Execute the created plan
  - `_generate_recommendations()`: Generate recommendations from results

### Agent Runner
- **Purpose**: Run SimpleAgent for code analysis tasks
- **Responsibilities**:
  - Initialize and run SimpleAgent
  - Process command line arguments and task inputs
- **Key Methods**:
  - `run()`: Run SimpleAgent

## Examples

The project includes example implementations in the `examples/` directory:

1. **SimpleAgent (`examples/simple_agent.py`)**
   - Self-directed agent for code analysis tasks
   - Creates and executes plans using available tools
   - Generates recommendations based on findings

2. **Agent Runner (`examples/agent_runner.py`)**
   - Runner script for SimpleAgent
   - Initializes and runs the SimpleAgent for code analysis tasks
   - Processes command line arguments and task inputs

These examples demonstrate how to build specialized agents using the core MCP Client components. See the README in the examples directory for more details on running and creating examples.

## Developer Cheatsheet

### Common Commands

**Running the application:**
```bash
# Dynamic server connection (default)
python main.py

# Pre-connect to server script
python main.py server/main.py

# Pre-connect to configured server
python main.py --server brave-search

# Pre-connect to multiple servers
python main.py server/main.py --server brave-search

# Run SimpleAgent with a specific task
python agent_runner.py "Review my codebase structure"
```

**Development commands:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (requires .env file)
# OPENROUTER_API_KEY=your-openrouter-api-key
# BRAVE_API_KEY=your-brave-api-key
# DEFAULT_LLM_MODEL=google/gemini-2.0-flash-001
# TRACELOOP_API_KEY=your-traceloop-api-key

# Run with verbose logging
DEBUG=1 python main.py

# Connect to specific server and inspect tools
python main.py --server brave-search
```

### Common Debugging Patterns

**Server connection issues:**
```python
# Check if server is connected
server = server_registry.get_server("server-name")
if server:
    # Server is connected
    tools = server.get_tool_names()
    print(f"Connected to {server_name} with tools: {tools}")
else:
    # Server is not connected
    print(f"Not connected to {server_name}")
```

**Tool execution issues:**
```python
# Execute tool with try/except
try:
    result = await server_instance.execute_tool(tool_name, tool_args)
    print(f"Tool execution successful: {result}")
except Exception as e:
    print(f"Tool execution failed: {e}")
```

## Technical Debt & Areas for Improvement

1. **Error Handling**
   - Need more specific exception types
   - Better recovery mechanisms for server connection failures
   - More graceful handling of LLM API errors

2. **Testing**
   - No unit tests currently implemented
   - Need mocking framework for external dependencies
   - Should add integration tests for complete flows

3. **Configuration Management**
   - Server configuration scattered across files
   - Environment variable handling needs improvement
   - No validation for configuration values

4. **Logging Improvements**
   - More structured logging needed
   - Different log levels for components
   - Better formatting for debug information

5. **Security Considerations**
   - API key handling could be improved
   - No input validation for user queries
   - No rate limiting for LLM API calls

6. **Dependency Management**
   - No formal dependency injection
   - Components tightly coupled in some areas
   - No container for managing component lifecycle

## Dynamic Server Connection Implementation

The dynamic server connection is a core feature that allows the LLM to discover and connect to servers as needed:

### Implementation Details

1. **Server Management Tools**
   - The `ServerManagementHandler` creates special server management tools for the LLM
   - These tools allow the LLM to list, connect to, and use servers

2. **Server Discovery**
   - The `ServerRegistry.get_available_servers()` method discovers available servers:
     - Reads server configurations from `server_config.json`
     - Looks for common server script paths in the project

3. **Server Connection Flow**
   1. LLM requests list of available servers via `list_available_servers` tool
   2. Client returns available server information
   3. LLM decides which server to connect to based on tools needed
   4. LLM requests connection via `connect_to_server` tool
   5. Client connects to server and initializes session
   6. LLM receives updated tool list including new server tools
   7. LLM can now use tools from the connected server

4. **Connection Lifecycle Management**
   - Connections are maintained throughout the session
   - The `AsyncExitStack` in `ServerRegistry` ensures proper cleanup
   - Connections can be manually disconnected or automatically closed on exit
   - The `disconnect_server` method allows for explicit disconnection

5. **Server Connector Tools**
   - The `server_connector.py` module provides tools for connecting to external servers
   - These tools allow for dynamic connection to servers from within a server
   - Enables nested server connections for complex workflows

### Critical Implementation Notes

- **Server Initialization**: Server initialization must complete within timeout (30s default)
- **Session Management**: Each server has its own MCP session that must be maintained
- **Tool Attribution**: Each tool is tagged with its server name for proper routing
- **Server Connection Reuse**: Existing connections are reused when possible
- **Connection Cleanup**: All connections must be properly closed on application exit
- **Resource Management**: Each component manages its own resources with AsyncExitStack

## Data Flow

### Query Processing Flow

```
User Query -> Agent -> Conversation -> MessageProcessor -> LLMService -> OpenRouter API
                           ↓
                      LLM Response
                           ↓
                     ResponseProcessor
                           ↓
                     Tool Call Needed?
                     /           \
                  No             Yes
                   ↓               ↓
            Final Response   ToolExecutor/ServerManagementHandler
                                  ↓
                           ServerRegistry
                                  ↓
                          ServerInstance
                                  ↓
                             MCP Server
                                  ↓
                           Tool Execution
                                  ↓
                            Tool Result
                                  ↓
                        Follow-up Response
                                  ↓
                           Final Response
```

### Tool Execution Flow

1. LLM generates tool call with name and arguments
2. `ResponseProcessor.process_tool_call()` processes the tool call
3. If it's a server management tool, `ServerManagementHandler` handles it
4. Otherwise, `ToolExecutor.process_external_tool()` processes the tool
5. `ToolExecutor.find_server_for_tool()` identifies the server for the tool
6. `ToolExecutor.execute_tool()` executes the tool on the appropriate server
7. `ServerInstance.execute_tool()` sends the tool call to the MCP server
8. Server executes the tool and returns the result
9. `ToolExecutor.extract_result_text()` extracts text from the result
10. `MessageProcessor.update_message_history()` updates conversation history
11. `ResponseProcessor.get_follow_up_response()` gets follow-up response from LLM
12. Final response is returned to the user

### Server Connection Flow

1. `ServerRegistry.connect_to_configured_server()` or `connect_to_server()` is called
2. Server parameters are created using config or script path
3. `ServerRegistry._create_server_session()` creates MCP session
4. `ServerInstance.initialize()` initializes the server connection
5. `ServerInstance.refresh_tools()` discovers available tools
6. Connection is stored in `ServerRegistry.servers` dictionary
7. Tools are available for use through `ServerRegistry.collect_all_tools()`

### Server Disconnection Flow

1. `ServerRegistry.disconnect_server()` is called with server name
2. Server instance is retrieved from registry
3. `ServerInstance.cleanup()` is called to clean up resources
4. Server is removed from registry
5. Status information is returned

## Implementation Details to Remember

1. **Error Handling in Async Context**
   - Always use `try/except` inside async functions
   - Be careful with async context managers
   - Ensure `AsyncExitStack` is properly closed
   - Use decorators for consistent error handling

2. **LLM Tool Format**
   - Tools must be formatted in OpenAI function calling format
   - Each tool needs a unique name
   - Tool arguments must be properly specified with JSON Schema
   - Server attribution is added in tool metadata

3. **JSON-RPC Handling**
   - MCP uses JSON-RPC for communication
   - Error responses need proper error codes
   - Responses must match request IDs

4. **Environment Variable Processing**
   - Environment variables in server config use `${VAR_NAME}` syntax
   - Variables are resolved at runtime using `os.getenv()`
   - Missing variables are logged as warnings

5. **Tool Execution Timeouts**
   - Tool execution has a 30-second timeout by default
   - Long-running tools should implement progress reporting
   - Timeouts can cause resource leaks if not properly handled

6. **Session Initialization**
   - Server sessions must be properly initialized
   - Server capabilities are negotiated during initialization
   - Session initialization failures must be properly handled

7. **Message History Management**
   - LLM conversation history must be properly maintained
   - Tool calls and results must be correctly formatted
   - Message history can grow large and needs monitoring

8. **Resource Cleanup**
   - All server connections must be properly closed
   - AsyncExitStack ensures proper cleanup of async resources
   - Manual resource cleanup may be needed in some error cases
   - Use resource_cleanup decorator for consistent cleanup

9. **Traceloop Integration**
   - Use Traceloop's decorators for tracing workflows, tasks, and tools
   - Set appropriate association properties for context
   - Use manual tracking for fine-grained control
   - Track important events and metrics