"""
Simple self-directed agent for code analysis tasks.

This module provides a lightweight agent that can analyze code and documentation,
planning its own approach to solving tasks.
"""

import logging
import json
import re
import asyncio
import time
from typing import Dict, List, Any, Optional
import uuid
import hashlib

from traceloop.sdk.decorators import workflow, task, agent, tool
from traceloop.sdk import Traceloop
from traceloop.sdk.tracing.manual import track_llm_call, LLMMessage

from src.client.llm_client import LLMClient
from src.client.server_manager import ServerManager
from src.client.tool_processor import ToolProcessor

logger = logging.getLogger("SimpleAgent")

@agent(name="code_analysis_agent")
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
        
        # Generate a unique ID for this agent instance
        agent_id = str(uuid.uuid4())
        
        # Set global association properties for this agent instance
        Traceloop.set_association_properties({
            "agent_id": agent_id,
            "agent_type": "code_analysis_agent",
            "model": self.config.get("model"),
            "temperature": self.config.get("temperature"),
            "max_tokens": self.config.get("max_tokens")
        })
        
        logger.info("SimpleAgent initialized")
    
    @workflow(name="execute_task")
    async def execute_task(self, task: str) -> str:
        """
        Execute a self-directed task
        
        Args:
            task: Description of what the agent should do
            
        Returns:
            Results and recommendations
        """
        # Generate a unique task ID
        task_id = hashlib.md5(task.encode()).hexdigest()[:12]
        
        # Set association properties for tracing
        Traceloop.set_association_properties({
            "task_id": task_id,
            "task_type": "code_analysis",
            "task_description": task[:200] + "..." if len(task) > 200 else task,
            "task_length": len(task),
            "start_time": time.time()
        })
        
        logger.info(f"Starting self-directed task: {task}")
        
        # Update conversation history
        self.update_conversation_history("user", task)
        
        try:
            # Step 1: Get the agent to formulate a plan
            plan = await self._create_plan(task)
            step_count = len(plan.get('steps', []))
            
            logger.info(f"Created plan with {step_count} steps")
            
            # Track plan creation success
            Traceloop.set_association_properties({
                "plan_created": True,
                "step_count": step_count,
                "plan_status": "created"
            })
            
            # Step 2: Execute the plan
            results = await self._execute_plan(plan)
            
            # Track plan execution completion
            success_count = sum(1 for r in results if r.get("status") == "success")
            error_count = sum(1 for r in results if r.get("status") == "error")
            
            Traceloop.set_association_properties({
                "plan_status": "executed",
                "total_steps": len(results),
                "successful_steps": success_count,
                "error_steps": error_count
            })
            
            logger.info(f"Executed plan with {len(results)} results, {success_count} successful, {error_count} errors")
            
            # Step 3: Generate recommendations
            recommendations = await self._generate_recommendations(task, plan, results)
            
            # Track recommendations generation
            Traceloop.set_association_properties({
                "recommendations_generated": True,
                "recommendations_length": len(recommendations),
                "end_time": time.time()
            })
            
            logger.info("Generated recommendations")
            
            # Update conversation history
            self.update_conversation_history("assistant", recommendations)
            
            # Record user feedback for annotation (could be implemented via UI)
            Traceloop.set_association_properties({
                "task_completed": True,
                "task_success": True
            })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}", exc_info=True)
            error_response = f"I encountered an error while working on your task: {str(e)}"
            self.update_conversation_history("assistant", error_response)
            
            # Track the error
            Traceloop.set_association_properties({
                "task_completed": False,
                "task_success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "end_time": time.time()
            })
            
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
        
        # Track conversation history updates
        Traceloop.set_association_properties({
            "history_length": len(self.conversation_history),
            "last_message_role": role,
            "last_message_length": len(str(content))
        })
    
    @task(name="create_plan")
    async def _create_plan(self, task: str) -> Dict[str, Any]:
        """
        Have the agent create its own plan
        
        Args:
            task: User task to plan for
            
        Returns:
            Plan dictionary with steps
        """
        # Track plan creation start
        plan_creation_id = hashlib.md5(f"plan:{task}".encode()).hexdigest()[:12]
        
        Traceloop.set_association_properties({
            "plan_creation_id": plan_creation_id,
            "planning_stage": "started"
        })
        
        # Get available tools and servers
        available_tools = []
        for server_name, server in self.server_manager.servers.items():
            if hasattr(server, 'get_tool_names'):
                tools = server.get_tool_names()
                for tool in tools:
                    available_tools.append(f"{tool} (from {server_name})")
        
        # Track available tools for planning
        Traceloop.set_association_properties({
            "available_tools_count": len(available_tools)
        })
        
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
        
        # Set prompt version for tracing
        Traceloop.set_prompt(
            "Create a plan template: {task}",
            {"task": task},
            version=1
        )
        
        # Using manual LLM call tracking for planning
        with track_llm_call(vendor="openrouter", type="chat") as span:
            # Track the planning request
            llm_messages = [LLMMessage(role="user", content=prompt)]
            
            span.report_request(
                model=self.llm_client.model,
                messages=llm_messages
            )
            
            # Get the plan from LLM
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.get_completion(
                messages,
                []  # No tools for planning
            )
            
            # Track the planning response
            plan_text = response.choices[0].message.content
            span.report_response(
                self.llm_client.model,
                [plan_text]
            )
        
        # Track processing stage
        Traceloop.set_association_properties({
            "planning_stage": "processing_response",
            "response_length": len(plan_text)
        })
        
        try:
            # Find JSON in the response
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', plan_text, re.DOTALL)
            if json_match:
                plan = json.loads(json_match.group(1))
                extraction_method = "code_block"
            else:
                # Try to extract JSON without code blocks
                json_match = re.search(r'({(?:\s*"steps"\s*:.*?})}', plan_text, re.DOTALL)
                if json_match:
                    plan = json.loads(json_match.group(1))
                    extraction_method = "regex_match"
                else:
                    logger.warning("Could not extract JSON plan, using manual extraction")
                    # Manual extraction as fallback
                    cleaned_text = re.sub(r'^[^{]*', '', plan_text).strip()
                    cleaned_text = re.sub(r'[^}]*$', '', cleaned_text).strip()
                    plan = json.loads(cleaned_text)
                    extraction_method = "manual_fallback"
            
            # Ensure plan has steps array
            if "steps" not in plan:
                plan["steps"] = []
            
            # Track successful plan creation
            Traceloop.set_association_properties({
                "planning_stage": "completed",
                "plan_extraction_method": extraction_method,
                "step_count": len(plan.get("steps", [])),
                "plan_creation_status": "success"
            })
                
            return plan
                
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON plan from: {plan_text}")
            
            # Track parsing failure
            Traceloop.set_association_properties({
                "planning_stage": "parse_error",
                "plan_creation_status": "error",
                "error_type": "json_decode"
            })
            
            # Return empty plan if parsing fails
            return {"steps": []}
    
    @task(name="execute_plan")
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
        
        # Track plan execution start
        Traceloop.set_association_properties({
            "execution_stage": "started",
            "total_steps": len(steps)
        })
        
        for i, step in enumerate(steps):
            step_num = i + 1
            description = step.get("description", f"Step {step_num}")
            tool_name = step.get("tool")
            server_name = step.get("server")
            args = step.get("args", {})
            
            # Generate step ID for tracing
            step_id = hashlib.md5(f"step:{step_num}:{description}".encode()).hexdigest()[:12]
            
            # Add step association properties for tracing
            Traceloop.set_association_properties({
                "step_id": step_id,
                "step_number": step_num,
                "step_description": description[:100] + "..." if len(description) > 100 else description,
                "step_tool": tool_name or "none",
                "step_server": server_name or "none",
                "step_args_count": len(args)
            })
            
            logger.info(f"Executing step {step_num}/{len(steps)}: {description}")
            
            step_result = {
                "step": step_num,
                "description": description,
                "status": "pending"
            }
            
            try:
                # Execute tool if specified
                if tool_name and server_name:
                    # Track tool execution
                    Traceloop.set_association_properties({
                        "step_type": "tool_execution",
                        "execution_status": "started"
                    })
                    
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
                    
                    # Track successful tool execution
                    Traceloop.set_association_properties({
                        "execution_status": "completed",
                        "result_length": len(result_text),
                        "step_status": "success"
                    })
                else:
                    # If no tool, this is an analysis or reasoning step
                    step_result["status"] = "success"
                    step_result["note"] = "No tool execution required for this step"
                    
                    # Track reasoning step
                    Traceloop.set_association_properties({
                        "step_type": "reasoning",
                        "step_status": "success"
                    })
                    
            except Exception as e:
                logger.error(f"Error executing step {step_num}: {str(e)}", exc_info=True)
                step_result["status"] = "error"
                step_result["error"] = str(e)
                
                # Track step execution error
                Traceloop.set_association_properties({
                    "execution_status": "error",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "step_status": "error"
                })
            
            # Add step result to results
            results.append(step_result)
            
            # Track progress through plan
            Traceloop.set_association_properties({
                "steps_completed": step_num,
                "steps_remaining": len(steps) - step_num
            })
            
        # Track plan execution completion
        success_count = sum(1 for r in results if r.get("status") == "success")
        error_count = sum(1 for r in results if r.get("status") == "error")
        
        Traceloop.set_association_properties({
            "execution_stage": "completed",
            "successful_steps": success_count,
            "error_steps": error_count,
            "success_rate": success_count / len(steps) if len(steps) > 0 else 0
        })
        
        return results
    
    @task(name="generate_recommendations")
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
        # Generate recommendation ID for tracing
        recommendation_id = hashlib.md5(f"recommendations:{task}".encode()).hexdigest()[:12]
        
        # Track recommendation generation start
        Traceloop.set_association_properties({
            "recommendation_id": recommendation_id,
            "recommendation_stage": "started",
            "result_count": len(results)
        })
        
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
        
        # Set prompt version for tracing
        Traceloop.set_prompt(
            "Generate recommendations for task: {task}",
            {"task": task},
            version=1
        )
        
        # Using manual LLM call tracking for recommendations
        with track_llm_call(vendor="openrouter", type="chat") as span:
            # Track the recommendation request
            llm_messages = [LLMMessage(role="user", content=prompt)]
            
            span.report_request(
                model=self.llm_client.model,
                messages=llm_messages
            )
            
            # Get recommendations from LLM
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_client.get_completion(
                messages,
                []  # No tools for recommendations
            )
            
            # Track the recommendation response
            recommendation_text = response.choices[0].message.content
            span.report_response(
                self.llm_client.model,
                [recommendation_text]
            )
        
        # Track successful recommendation generation
        Traceloop.set_association_properties({
            "recommendation_stage": "completed",
            "recommendation_length": len(recommendation_text),
            "recommendation_status": "success"
        })
        
        return recommendation_text
    
    @tool(name="format_results")
    def _format_execution_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format execution results as text
        
        Args:
            results: List of step results
            
        Returns:
            Formatted results text
        """
        # Track formatting operation
        Traceloop.set_association_properties({
            "formatting_operation": "execution_results",
            "result_count": len(results)
        })
        
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
        
        final_text = "\n\n".join(formatted_results)
        
        # Track formatting completion
        Traceloop.set_association_properties({
            "formatted_length": len(final_text),
            "formatting_status": "success"
        })
        
        return final_text