"""
Simple self-directed agent for code analysis tasks.

This module provides a lightweight agent that can analyze code and documentation,
planning its own approach to solving tasks.
"""

import logging
import json
import re
import asyncio
from typing import Dict, List, Any, Optional
from traceloop.sdk.decorators import agent
from src.client.llm_client import LLMClient
from src.client.server_manager import ServerManager
from src.client.tool_processor import ToolProcessor

logger = logging.getLogger("SimpleAgent")

class SimpleAgent:
    """
    A lightweight, self-directing agent that can analyze code and documentation.
    
    This agent can:
    1. Create its own plan for tackling a task
    2. Execute the plan using available tools
    3. Generate recommendations based on its findings
    """
    
    def __init__(
        self, 
        llm_client: LLMClient, 
        server_manager: ServerManager,
        tool_processor: ToolProcessor,
        config: Dict[str, Any] = None
    ):
        """
        Initialize the simple agent
        
        Args:
            llm_client: LLM client for model interactions
            server_manager: Server manager for MCP server connections
            tool_processor: Tool processor for executing tools
            config: Optional configuration dictionary
        """
        self.llm_client = llm_client
        self.server_manager = server_manager
        self.tool_processor = tool_processor
        self.config = config or {}
        self.conversation_history = []
        
        # Set default configuration
        self.config.setdefault("model", "google/gemini-2.0-pro")
        self.config.setdefault("temperature", 0.7)
        self.config.setdefault("max_tokens", 2048)
        
        logger.info("SimpleAgent initialized")
    
    async def execute_task(self, task: str) -> str:
        """
        Execute a self-directed task
        
        Args:
            task: Description of what the agent should do
            
        Returns:
            Results and recommendations
        """
        logger.info(f"Starting self-directed task: {task}")
        
        # Update conversation history
        self.update_conversation_history("user", task)
        
        try:
            # Step 1: Get the agent to formulate a plan
            plan = await self._create_plan(task)
            # print the plan
            logger.info(f"Created plan with {len(plan.get('steps', []))} steps")
            logger.info(f"Plan: {plan}")
            
            # Step 2: Execute the plan
            results = await self._execute_plan(plan)
            logger.info(f"Executed plan with {len(results)} results")
            
            # Step 3: Generate recommendations
            recommendations = await self._generate_recommendations(task, plan, results)
            logger.info("Generated recommendations")
            
            # Update conversation history
            self.update_conversation_history("assistant", recommendations)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}", exc_info=True)
            error_response = f"I encountered an error while working on your task: {str(e)}"
            self.update_conversation_history("assistant", error_response)
            return error_response
    
    def update_conversation_history(self, role: str, content: Any) -> None:
        """
        Update the conversation history
        
        Args:
            role: Role of the message ('user', 'assistant', 'system', 'tool')
            content: Content of the message
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    async def _create_plan(self, task: str) -> Dict[str, Any]:
        """
        Have the agent create its own plan
        
        Args:
            task: User task to plan for
            
        Returns:
            Plan dictionary with steps
        """
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
        
        Create a detailed plan with specific steps. For each step that uses a tool, specify:
        1. Which tool to use
        2. What server provides the tool
        3. What arguments to pass to the tool
        
        Be strategic in your approach. Consider what information you need to gather and in what order.
        Break down complex steps into simpler ones. Make sure each step has a clear purpose.
        
        Format your response as JSON with this structure:
        {{
          "steps": [
            {{
              "description": "Step description",
              "tool": "tool_name",  // or null if no tool needed
              "server": "server_name",  // server that provides the tool
              "args": {{}}  // arguments for the tool
            }}
          ]
        }}
        
        Make sure your plan is comprehensive and addresses all parts of the task.
        """
        
        # Get the plan from LLM
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.get_completion(
            messages,
            [],  # No tools for planning
            # model=self.config["model"],
            # temperature=self.config["temperature"],
            # max_tokens=self.config["max_tokens"]
        )
        
        # Extract JSON plan
        plan_text = response.choices[0].message.content
        
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(1))
            else:
                # Try to extract JSON without code blocks
                json_match = re.search(r'({(?:\s*"steps"\s*:.*?})}', plan_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(1))
                else:
                    logger.warning("Could not extract JSON plan, using manual extraction")
                    # Manual extraction as fallback
                    cleaned_text = re.sub(r'^[^{]*', '', plan_text).strip()
                    cleaned_text = re.sub(r'[^}]*$', '', cleaned_text).strip()
                    plan = json.loads(cleaned_text)
            
            # Ensure plan has steps array
            if "steps" not in plan:
                plan["steps"] = []
                
            return plan
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON plan from: {plan_text}")
            # Return empty plan if parsing fails
            return {"steps": []}
    
    async def _execute_plan(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute the agent's plan
        
        Args:
            plan: Plan dictionary with steps
            
        Returns:
            List of step results
        """
        results = []
        steps = plan.get("steps", [])
        
        for i, step in enumerate(steps):
            step_num = i + 1
            description = step.get("description", f"Step {step_num}")
            tool_name = step.get("tool")
            server_name = step.get("server")
            args = step.get("args", {})
            
            logger.info(f"Executing step {step_num}/{len(steps)}: {description}")
            
            step_result = {
                "step": step_num,
                "description": description,
                "status": "pending"
            }
            
            try:
                # Execute tool if specified
                if tool_name and server_name:
                    # Get server
                    server = self.server_manager.get_server(server_name)
                    if not server:
                        raise ValueError(f"Server not found: {server_name}")
                    
                    # Execute tool
                    logger.info(f"Executing tool: {tool_name} with args: {args}")
                    result = await self.tool_processor.execute_tool(tool_name, args, server_name)
                    
                    # Extract result text
                    result_text = self.tool_processor.extract_result_text(result)
                    
                    # Update step result
                    step_result["tool_result"] = result_text
                    step_result["status"] = "success"
                else:
                    # If no tool, this is an analysis or reasoning step
                    step_result["status"] = "success"
                    step_result["note"] = "No tool execution required for this step"
                    
            except Exception as e:
                logger.error(f"Error executing step {step_num}: {str(e)}", exc_info=True)
                step_result["status"] = "error"
                step_result["error"] = str(e)
            
            # Add step result to results
            results.append(step_result)
            
        return results
    
    async def _generate_recommendations(
        self,
        task: str,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate recommendations based on execution results
        
        Args:
            task: Original user task
            plan: Execution plan
            results: Results of steps executed
            
        Returns:
            Recommendations text
        """
        # Format results for the LLM
        formatted_results = self._format_execution_results(results)
        
        # Create recommendation prompt
        prompt = f"""
        Based on the results of your investigation, provide specific recommendations for this task:
        
        Original task: {task}
        
        Your plan:
        {json.dumps(plan, indent=2)}
        
        Execution results:
        {formatted_results}
        
        Provide clear, specific recommendations based on what you found. Include:
        1. A summary of what you discovered
        2. Specific recommendations for improvements
        3. Any code examples if relevant
        
        Make your response conversational and helpful. Focus on providing actionable insights.
        """
        
        # Get recommendations from LLM
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_client.get_completion(
            messages,
            [],  # No tools for recommendations
            # model=self.config["model"],
            # temperature=self.config["temperature"],
            # max_tokens=self.config["max_tokens"]
        )
        
        return response.choices[0].message.content
    
    def _format_execution_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format execution results as text
        
        Args:
            results: List of step results
            
        Returns:
            Formatted results text
        """
        formatted_results = []
        
        for result in results:
            step = result.get("step", "?")
            description = result.get("description", "Unknown step")
            status = result.get("status", "unknown")
            
            step_text = f"Step {step}: {description} - {status}"
            
            if status == "error":
                step_text += f"\nError: {result.get('error', 'Unknown error')}"
            
            if "tool_result" in result:
                tool_result = result["tool_result"]
                # Truncate long results
                if len(tool_result) > 1000:
                    tool_result = tool_result[:997] + "..."
                step_text += f"\nResult: {tool_result}"
            
            if "note" in result:
                step_text += f"\nNote: {result['note']}"
            
            formatted_results.append(step_text)
        
        return "\n\n".join(formatted_results)