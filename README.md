# MCP Client

A modular client for interacting with MCP (Machine Control Protocol) servers using OpenRouter as an LLM provider.

## Project Structure

The project is organised into several modules, each with a specific responsibility:

### Client Components
- **main.py**: Entry point for the application
- **src/client/mcp_client.py**: Main client for interacting with MCP servers
- **src/client/llm_client.py**: Client for communicating with the OpenRouter API
- **src/client/tool_manager.py**: Manages tool operations and formatting
- **src/utils/logger_setup.py**: Configures logging for the application

### Server Components
The server code has been reorganised into a modular structure:

```
server/
├── main.py                 # Entry point that imports and registers all components
├── tools/                  # Directory for all tool implementations
│   ├── math_tools.py       # Math-related tools (add, multiply)
│   ├── health_tools.py     # Health calculations (BMI)
│   └── external_data.py    # Weather and other API tools
├── resources/              # Directory for all resource implementations
│   └── greetings.py        # Greeting resources
├── prompts/                # Directory for prompt templates (for future use)
└── utils/                  # Utility functions shared across modules
    └── api_helpers.py      # Helper functions for API calls
```

## Important Notes

### Tool Decorator Usage

The current version of MCP (0.2.0) used in this project supports the `@mcp.tool()` decorator without additional parameters. Tools are defined as follows:

```python
@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b
```

For organisational purposes, tools are grouped by module (math_tools.py, health_tools.py, etc.) rather than by categories within the decorator.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your-openrouter-api-key
   ```

## Usage

### Running with the New Structure

With the refactored server structure, run the client using:

```bash
python src/client/mcp_client.py server/main.py
```

Or directly from the main application entry point:

```bash
python main.py server/main.py
```

### How Client-Server Interaction Works

When you run the client with the server script, the following happens:

1. The client launches the server script as a subprocess using stdio communication
2. The server initialises, registering all tools, resources, and prompts
3. The client establishes a connection and queries the server for available tools
4. The MCP protocol facilitates communication between client and server
5. The client displays the tools to the LLM, which can then decide to use them

### Connection Flow Explained

1. **Server Initialisation**:
   - The `main.py` script creates a FastMCP server instance
   - It imports and registers all tools, resources, and prompts from their respective modules
   - Each module contains a `register_X` function that adds its components to the server

2. **Client Connection**:
   - The client creates a `StdioServerParameters` object specifying the server script
   - It establishes a connection using `stdio_client`
   - A `ClientSession` is created to handle the communication

3. **Tool Discovery**:
   - The client calls `await session.list_tools()` to get available tools from the server
   - The server responds with a list of available tools and their metadata
   - This happens in `mcp_client.py` in the `connect_to_server` method

4. **Tool Execution**:
   - When the LLM decides to use a tool, the client sends a request to the server
   - The server executes the requested tool and returns the result
   - The client processes the result and sends it back to the LLM

## Available Tools

The server provides the following tools:

1. **Math Operations**:
   - `add`: Adds two numbers
   - `multiply`: Multiplies two numbers

2. **Health Calculations**:
   - `calculate_bmi`: Calculates BMI from weight (kg) and height (m)

3. **External Data**:
   - `fetch_weather`: Fetches weather information for a location based on coordinates

4. **Resources**:
   - `greeting://{name}`: Returns a personalised greeting

## Example Client-Server Interaction

Here's a simplified example of how the client interacts with the server:

```python
# Client code simplified for illustration
async def connect_to_server(server_script_path):
    # Launch server subprocess
    server_params = StdioServerParameters(command="python", args=[server_script_path])
    
    # Connect to server via stdio
    async with stdio_client(server_params) as (read, write):
        # Create client session
        async with ClientSession(read, write) as session:
            # Initialise connection
            await session.initialize()
            
            # THIS IS WHERE THE CLIENT DISCOVERS TOOLS
            tools_response = await session.list_tools()
            available_tools = [Tool(tool.name, tool.description, tool.inputSchema) 
                              for tool in tools_response.tools]
            
            # Use tool based on LLM decision
            result = await session.call_tool("add", {"a": 5, "b": 3})
            return result
```

## Extending

You can extend the server with additional tools by:

1. Adding new tool modules in the `server/tools/` directory
2. Creating new resource modules in the `server/resources/` directory
3. Implementing registration functions for your new components
4. Updating the appropriate `__init__.py` file to import and register your components

## Debugging

Detailed logging is provided for debugging and development purposes. Logs are displayed in the console by default.