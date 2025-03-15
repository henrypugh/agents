# MCP Client

![Status: Active Development](https://img.shields.io/badge/Status-Active_Development-green)
![Type: Agent Framework](https://img.shields.io/badge/Type-Agent_Framework-blue)

An advanced Pydantic-integrated framework for building LLM-powered agents that can dynamically connect to Model Context Protocol (MCP) servers and execute tools.

## 🚀 Features

- **Dynamic Server Connection**: Connect to servers on-demand as needs arise
- **Validated Data Models**: End-to-end Pydantic integration for robust type safety
- **Modular Architecture**: Cleanly separated components with well-defined interfaces
- **Tracing & Observability**: Comprehensive logging and error handling
- **Self-directing Agents**: Agents that can plan and execute complex tasks

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Key Components](#-key-components)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Examples](#-examples)
- [Advanced Usage](#-advanced-usage)
- [Development Guide](#-development-guide)
- [Troubleshooting](#-troubleshooting)

## 🏗 Architecture

The project follows a modular architecture with clearly defined components:

```
┌─────────────────────────────────────────┐
│             Agent (main)                │
└─┬─────────────┬──────────────────┬──────┘
  │             │                  │
  ▼             ▼                  ▼
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

## 🧩 Key Components

### Agent (`src/client/agent.py`)
- Main entry point and coordinator
- Manages server connections, chat flow, and user interactions
- Provides methods for processing queries and managing conversation

### Conversation (`src/client/conversation/`)
- Orchestrates conversation flow and tool execution
- Manages message processing and response handling
- Delegates to specialized handlers for different aspects

### ServerRegistry (`src/client/server_registry.py`)
- Manages discovery and lifecycle of server connections
- Handles tool collection from connected servers
- Provides environment management for server execution

### LLMService (`src/client/llm_service.py`)
- Communicates with LLM API through OpenRouter
- Handles validation of requests and responses
- Provides retry logic and error handling

### ToolExecutor (`src/client/tool_processor.py`)
- Executes tools from LLM requests
- Routes tool calls to appropriate servers
- Manages tool execution results

### SimpleAgent (`examples/simple_agent.py`)
- Self-directing agent implementation
- Creates and executes plans to complete user tasks
- Generates recommendations based on analysis

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mcp-client.git
cd mcp-client
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your API keys
OPENROUTER_API_KEY=your-openrouter-api-key
BRAVE_API_KEY=your-brave-api-key
DEFAULT_LLM_MODEL=google/gemini-2.0-flash-001
TRACELOOP_API_KEY=your-traceloop-api-key
```

## 🚦 Quick Start

Run the basic client to start an interactive session:

```bash
python main.py
```

Pre-connect to a server at startup:

```bash
# Connect to local server
python main.py server/main.py

# Connect to configured server
python main.py --server brave-search

# Connect to multiple servers
python main.py server/main.py --server brave-search
```

## 📚 Examples

### Interactive Chat

```python
from src.client.agent import Agent

async def main():
    agent = Agent()
    await agent.chat_loop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Process a Query

```python
from src.client.agent import Agent

async def main():
    agent = Agent()
    
    # Connect to a server
    await agent.connect_to_configured_server("brave-search")
    
    # Process a query
    response = await agent.process_query("Find information about pydantic")
    print(response)
    
    # Cleanup
    await agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Self-directed Agent

```bash
# Run SimpleAgent with a specific task
python examples/agent_runner.py "Analyze my codebase structure and suggest improvements"
```

## 🔧 Advanced Usage

### Customizing Agent Configuration

```python
from src.client.agent import Agent, AgentSettings
from src.client.server_registry import ServerRegistryConfig, ConnectionSettings
from src.client.llm_service import RetrySettings

# Create configurations
agent_settings = AgentSettings(
    trace_queries=True,
    max_history_length=50
)

server_config = ServerRegistryConfig(
    connection_settings=ConnectionSettings(
        timeout=30.0,
        max_retries=3
    ),
    validate_servers=True
)

llm_retry_settings = RetrySettings(
    max_retries=3,
    retry_delay=1.0,
    timeout=45.0
)

# Initialize with custom configurations
agent = Agent(
    model="anthropic/claude-3-opus-20240229",
    agent_settings=agent_settings,
    server_config=server_config,
    llm_retry_settings=llm_retry_settings
)
```

### Using System Prompts

```python
from src.client.agent import Agent

async def main():
    agent = Agent()
    
    # Process query with system prompt
    response = await agent.process_system_prompt(
        system_prompt="You are a helpful AI assistant specializing in Python programming.",
        user_query="How do I use Pydantic with FastAPI?"
    )
    
    print(response)
    await agent.cleanup()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## 🛠 Development Guide

### Project Organization

- `src/client/`: Core client components
- `src/utils/`: Utilities and helper functions
- `server/`: Server implementations
- `examples/`: Example agent implementations
- `tests/`: Test cases

### Common Commands

```bash
# Run with verbose logging
DEBUG=1 python main.py

# Run SimpleAgent
python examples/agent_runner.py "Your task here"
```

### Creating a Custom Agent

1. Create a new Python file for your agent
2. Import the necessary components
3. Define your agent class
4. Implement the required methods
5. Create a runner script

Example skeleton:

```python
from src.client.agent import Agent
from src.client.llm_service import LLMService
from src.client.server_registry import ServerRegistry
from src.client.tool_processor import ToolExecutor

class MyCustomAgent:
    def __init__(self, llm_client, server_manager, tool_processor):
        self.llm_client = llm_client
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        
    async def execute_task(self, task):
        # Implement your task execution logic here
        pass
```

## 🚨 Troubleshooting

### Common Issues

1. **Server Connection Problems**
   - Check if the server is running
   - Verify server configuration in `server_config.json`
   - Check environment variables

2. **LLM API Errors**
   - Verify your OpenRouter API key is valid
   - Check model name is correct
   - Look for rate limiting or timeout issues

3. **Tool Execution Failures**
   - Verify tool arguments are correctly formatted
   - Check server connectivity
   - Look for tool-specific error messages

### Debugging Tips

- Enable debug logging with `DEBUG=1 python main.py`
- Check the server logs for errors
- Examine the Traceloop dashboard for detailed execution traces

## 📄 License

MIT