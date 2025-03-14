# MCP Client - Developer Reference

This document serves as a personal reference for ongoing development of the MCP Client project. It documents the architecture, code structure, implementation details, and development notes to help quickly re-orient when returning to the codebase.

## Architecture Overview

The project follows a modular architecture organized around these core components:

```
┌─────────────────────────────────────────┐
│             MCP Client (main)           │
└─┬─────────────────┬──────────────────┬──┘
  │                 │                  │
  ▼                 ▼                  ▼
┌─────────┐   ┌───────────────┐   ┌──────────────┐
│LLM Client│   │Server Manager │   │Conversation  │
└─────────┘   └───────┬───────┘   │Manager       │
                      │           └───────┬──────┘
                      ▼                   │
              ┌────────────────┐         │
              │Server Connection│◄────────┘
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

2. **MCP Client (`src/client/mcp_client.py`)**
   - Core orchestrator that delegates to specialized components
   - Maintains the chat loop and handles user interactions
   - Exposes public API for connecting to servers and processing queries

3. **Conversation Manager (`src/client/conversation_manager.py`)**
   - Manages LLM interaction and conversation flow
   - Processes queries through the LLM
   - Handles tool calls and results processing
   - Contains server management tools logic

4. **Server Manager (`src/client/server_manager.py`)**
   - Manages creation and lifecycle of server connections
   - Stores server connection instances
   - Handles discovery of available servers
   - Collects tools from all connected servers

5. **Tool Processor (`src/client/tool_processor.py`)**
   - Processes tool calls from the LLM
   - Routes tools to appropriate servers
   - Handles tool execution and error handling

6. **Server Connection (`src/client/server_connection.py`)**
   - Encapsulates connection to individual MCP servers
   - Handles initialization and tool discovery
   - Executes tool calls on specific servers

7. **LLM Client (`src/client/llm_client.py`)**
   - Handles communication with the OpenRouter API
   - Formats messages and tools for the LLM
   - Processes LLM responses

## Refactoring History & Design Decisions

### Initial Implementation
- Started with monolithic architecture in a single `mcp_client.py` file
- Implemented basic functionality for connecting to MCP servers
- Added LLM integration with OpenRouter

### First Refactoring (Current)
- Split monolithic design into specialized components
- Introduced `ServerRegistry` to handle server connection lifecycle
- Created `Conversation` to handle LLM interaction
- Added `ToolExecutor` to handle tool routing and execution
- Separated `ServerInstance` to encapsulate individual server connections
- Updated `Agent` to orchestrate these components

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

5. **Configuration Management**
   - Server configurations stored in external JSON
   - Environment variables for sensitive information
   - Server configs dynamically processed at runtime

## Component Breakdown

### MCP Client
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

### Server Manager
- **Purpose**: Manage server connections
- **Responsibilities**:
  - Create and store server connections
  - Initialize server sessions
  - Discover available servers
  - Collect tools from servers
- **Key Methods**:
  - `connect_to_server()`: Connect to server by script path
  - `connect_to_configured_server()`: Connect to configured server
  - `get_server()`: Get server connection by name
  - `collect_all_tools()`: Collect tools from all servers
  - `get_connected_servers()`: Get info about connected servers
  - `get_available_servers()`: Get all available servers
  - `cleanup()`: Clean up server connections

### Conversation Manager
- **Purpose**: Manage LLM conversations
- **Responsibilities**:
  - Process queries through LLM
  - Handle tool calls
  - Update conversation history
  - Process server management tools
- **Key Methods**:
  - `process_query()`: Process user query
  - `_run_conversation()`: Run conversation with LLM
  - `_process_response()`: Process LLM response
  - `_process_tool_call()`: Process tool call from LLM
  - `_handle_server_management_tool()`: Handle server management tools
  - `_get_follow_up_response()`: Get follow-up response after tool execution

### Tool Processor
- **Purpose**: Process tool calls
- **Responsibilities**:
  - Find server for tool
  - Execute tool on appropriate server
  - Extract results from tool execution
- **Key Methods**:
  - `find_server_for_tool()`: Find server for tool
  - `execute_tool()`: Execute tool on server
  - `extract_result_text()`: Extract text from tool result

### Server Connection
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

### LLM Client
- **Purpose**: Communicate with LLM API
- **Responsibilities**:
  - Format messages and tools for LLM
  - Send requests to OpenRouter API
  - Process LLM responses
- **Key Methods**:
  - `get_completion()`: Get completion from LLM

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
```

**Development commands:**
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (requires .env file)
# OPENROUTER_API_KEY=your-openrouter-api-key
# BRAVE_API_KEY=your-brave-api-key
# DEFAULT_LLM_MODEL=google/gemini-2.0-flash-001

# Run with verbose logging
DEBUG=1 python main.py

# Connect to specific server and inspect tools
python main.py --server brave-search --inspect-tools
```

### Common Debugging Patterns

**Server connection issues:**
```python
# Check if server is connected
server = server_manager.get_server("server-name")
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
    result = await server_connection.execute_tool(tool_name, tool_args)
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
   - The `Conversation` adds special server management tools to the LLM
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

### Critical Implementation Notes

- **Server Initialization**: Server initialization must complete within timeout (30s default)
- **Session Management**: Each server has its own MCP session that must be maintained
- **Tool Attribution**: Each tool is tagged with its server name for proper routing
- **Server Connection Reuse**: Existing connections are reused when possible
- **Connection Cleanup**: All connections must be properly closed on application exit

## Data Flow

### Query Processing Flow

```
User Query -> MCP Client -> Conversation Manager -> LLM Client -> OpenRouter API
                                     ↓
                                LLM Response
                                     ↓
                             Tool Call Needed?
                             /           \
                          No             Yes
                           ↓               ↓
                    Final Response   Tool Processor
                                          ↓
                                   Server Manager
                                          ↓
                                  Server Connection
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
2. `Conversation._process_tool_call()` processes the tool call
3. `ToolExecutor.find_server_for_tool()` identifies the server for the tool
4. `ToolExecutor.execute_tool()` executes the tool on the appropriate server
5. `ServerInstance.execute_tool()` sends the tool call to the MCP server
6. Server executes the tool and returns the result
7. `ToolExecutor.extract_result_text()` extracts text from the result
8. `Conversation._update_message_history()` updates conversation history
9. `Conversation._get_follow_up_response()` gets follow-up response from LLM
10. Final response is returned to the user

### Server Connection Flow

1. `ServerRegistry.connect_to_configured_server()` or `connect_to_server()` is called
2. Server parameters are created using config or script path
3. `ServerRegistry._create_server_session()` creates MCP session
4. `ServerInstance.initialize()` initializes the server connection
5. `ServerInstance.refresh_tools()` discovers available tools
6. Connection is stored in `ServerRegistry.servers` dictionary
7. Tools are available for use through `ServerRegistry.collect_all_tools()`

## Implementation Details to Remember

1. **Error Handling in Async Context**
   - Always use `try/except` inside async functions
   - Be careful with async context managers
   - Ensure `AsyncExitStack` is properly closed

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

