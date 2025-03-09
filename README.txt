# MCP Client

A modular client for interacting with MCP (Machine Control Protocol) servers using OpenRouter as an LLM provider.

## Project Structure

The project is organized into several modules, each with a specific responsibility:

- **main.py**: Entry point for the application
- **mcp_client.py**: Main client for interacting with MCP servers
- **llm_client.py**: Client for communicating with the OpenRouter API
- **tool_manager.py**: Manages tool operations and formatting
- **logger_setup.py**: Configures logging for the application
- **server.py**: Demo MCP server with example tools

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

1. Start the server in one terminal:
   ```
   python server.py
   ```

2. Start the client in another terminal:
   ```
   python main.py server.py
   ```

3. Once the client is running, you can enter queries. The client will:
   - Send your query to the LLM via OpenRouter
   - The LLM will decide whether to use any of the tools provided by the server
   - If tools are used, their results will be incorporated into the final response

## Available Tools

The demo server includes the following tools:

1. **add**: Adds two numbers
   ```
   Example: "What is 123 + 456?"
   ```

2. **calculate_bmi**: Calculates BMI from weight (kg) and height (m)
   ```
   Example: "Calculate the BMI for someone who weighs 70kg and is 1.75m tall"
   ```

3. **fetch_weather**: Fetches weather information for a location
   ```
   Example: "What's the weather in New York?" (The LLM will convert this to coordinates)
   ```

4. **greeting**: Returns a personalized greeting
   ```
   Example: "Get a greeting for John"
   ```

## Extending

You can extend the server with additional tools by adding new functions decorated with `@mcp.tool()`.

## How It Works

1. The client establishes a connection to the server.
2. The client gets a list of available tools from the server.
3. When you enter a query, it's sent to the LLM via OpenRouter.
4. If the LLM decides to use tools, the client executes the tool calls on the server.
5. Tool results are sent back to the LLM for a final response.
6. The final response is displayed to you.

## Logging

Detailed logging is provided for debugging and development purposes. Logs are displayed in the console by default.