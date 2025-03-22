"""
Code Analysis Workflow.

This module provides a workflow for analyzing code and providing recommendations,
similar to what the SimpleAgent does but using the event-driven workflow system.
"""

import logging
import time
import hashlib
from typing import Union, Dict, Any, List, Optional

from src.utils.workflow import (
    Workflow,
    WorkflowContext,
    step,
    StartEvent,
    StopEvent,
    MessageEvent,
    ToolEvent,
    Event,
    ErrorEvent
)

logger = logging.getLogger(__name__)

# Define custom event types
class PlanningEvent(Event):
    """Event to trigger plan creation"""
    query: str

class ExecuteStepEvent(Event):
    """Event to trigger execution of a plan step"""
    plan_id: str
    step_number: int
    description: str
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    arguments: Dict[str, Any] = {}

class StepResultEvent(Event):
    """Event containing the result of a plan step execution"""
    plan_id: str
    step_number: int
    description: str
    status: str
    result: Optional[str] = None
    error: Optional[str] = None

class GenerateRecommendationsEvent(Event):
    """Event to trigger recommendation generation"""
    plan_id: str
    task: str
    results: List[Dict[str, Any]]


class CodeAnalysisWorkflow(Workflow):
    """
    A workflow for analyzing code and providing recommendations.
    
    This workflow:
    1. Creates a plan for the analysis task
    2. Executes each step of the plan
    3. Generates recommendations based on the results
    """
    
    @step
    async def start_analysis(self, ctx: WorkflowContext, event: StartEvent) -> PlanningEvent:
        """
        Start the code analysis process.
        
        Args:
            ctx: Workflow context
            event: Start event with the task description
            
        Returns:
            PlanningEvent to trigger plan creation
        """
        # Extract the task from input
        task = event.input.get("task", "")
        if not task:
            return ErrorEvent(error_message="No task provided")
        
        # Store the task in context
        await ctx.set("task", task)
        
        # Generate a task ID for tracing
        task_id = hashlib.md5(task.encode()).hexdigest()[:12]
        await ctx.set("task_id", task_id)
        
        # Log the start of analysis
        logger.info(f"Starting code analysis task: {task[:100]}...")
        
        # Transition to planning phase
        return PlanningEvent(query=task)
    
    @step
    async def create_plan(self, ctx: WorkflowContext, event: PlanningEvent) -> Union[ExecuteStepEvent, ErrorEvent]:
        """
        Create a plan for the analysis task.
        
        Args:
            ctx: Workflow context
            event: Planning event with the task
            
        Returns:
            ExecuteStepEvent for the first step or ErrorEvent
        """
        task = event.query
        task_id = await ctx.get("task_id")
        
        # Get available tools
        tools = []
        for server_name, server in ctx.server_manager.servers.items():
            if hasattr(server, 'get_tool_names'):
                server_tools = server.get_tool_names()
                for tool in server_tools:
                    tools.append(f"{tool} (from {server_name})")
        
        # Create planning prompt
        prompt = f"""
        I need you to create a plan to complete this task: {task}
        
        Available tools: {', '.join(tools)}
        
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
        """
        
        # Get plan from LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await ctx.llm_client.get_completion(messages, [])
            
            # Extract JSON plan from response
            import re
            import json
            
            plan_text = response.content or ""
            
            # Try to extract JSON from the response
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', plan_text, re.DOTALL)
            if json_match:
                plan_dict = json.loads(json_match.group(1))
                extraction_method = "code_block"
            else:
                # Try to extract JSON without code blocks
                json_match = re.search(r'({(?:\s*"steps"\s*:.*?})}', plan_text, re.DOTALL)
                if json_match:
                    plan_dict = json.loads(json_match.group(1))
                    extraction_method = "regex_match"
                else:
                    logger.warning("Could not extract JSON plan, using manual extraction")
                    # Manual extraction as fallback
                    cleaned_text = re.sub(r'^[^{]*', '', plan_text).strip()
                    cleaned_text = re.sub(r'[^}]*$', '', cleaned_text).strip()
                    plan_dict = json.loads(cleaned_text)
                    extraction_method = "manual_fallback"
            
            # Store the plan in context
            plan_id = hashlib.md5(json.dumps(plan_dict).encode()).hexdigest()[:12]
            await ctx.set("plan_id", plan_id)
            await ctx.set("plan", plan_dict)
            await ctx.set("step_results", [])
            
            # Log the plan
            steps = plan_dict.get("steps", [])
            step_count = len(steps)
            logger.info(f"Created plan with {step_count} steps")
            
            if step_count > 0:
                # Execute the first step
                first_step = steps[0]
                return ExecuteStepEvent(
                    plan_id=plan_id,
                    step_number=1,
                    description=first_step.get("description", ""),
                    tool_name=first_step.get("tool"),
                    server_name=first_step.get("server"),
                    arguments=first_step.get("args", {})
                )
            else:
                return ErrorEvent(error_message="Plan contains no steps")
                
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            return ErrorEvent.from_exception(e, step_name="create_plan")
    
    @step
    async def execute_step(self, ctx: WorkflowContext, event: ExecuteStepEvent) -> Union[StepResultEvent, ErrorEvent]:
        """
        Execute a step of the plan.
        
        Args:
            ctx: Workflow context
            event: Execute step event
            
        Returns:
            StepResultEvent with the result
        """
        plan_id = event.plan_id
        step_number = event.step_number
        description = event.description
        tool_name = event.tool_name
        server_name = event.server_name
        arguments = event.arguments
        
        logger.info(f"Executing step {step_number}: {description}")
        
        # Execute tool if specified
        if tool_name and server_name:
            try:
                # Get server
                server = ctx.server_manager.get_server(server_name)
                if not server:
                    return StepResultEvent(
                        plan_id=plan_id,
                        step_number=step_number,
                        description=description,
                        status="error",
                        error=f"Server '{server_name}' not found"
                    )
                
                # Execute tool
                logger.info(f"Executing tool: {tool_name} with args: {arguments}")
                result = await ctx.tool_processor.execute_tool(tool_name, arguments, server_name)
                
                # Extract result text
                result_text = ctx.tool_processor.extract_result_text(result)
                
                return StepResultEvent(
                    plan_id=plan_id,
                    step_number=step_number,
                    description=description,
                    status="success",
                    result=result_text
                )
            
            except Exception as e:
                logger.error(f"Error executing step {step_number}: {e}")
                return StepResultEvent(
                    plan_id=plan_id,
                    step_number=step_number,
                    description=description,
                    status="error",
                    error=str(e)
                )
        else:
            # No tool step, just a reasoning step
            return StepResultEvent(
                plan_id=plan_id,
                step_number=step_number,
                description=description,
                status="success",
                result="Reasoning step completed"
            )
    
    @step
    async def process_step_result(self, ctx: WorkflowContext, event: StepResultEvent) -> Union[ExecuteStepEvent, GenerateRecommendationsEvent]:
        """
        Process the result of a step and determine next action.
        
        Args:
            ctx: Workflow context
            event: Step result event
            
        Returns:
            ExecuteStepEvent for the next step or GenerateRecommendationsEvent
        """
        plan_id = event.plan_id
        step_number = event.step_number
        
        # Get current plan and results
        plan = await ctx.get("plan", {})
        steps = plan.get("steps", [])
        
        # Store the result
        results = await ctx.get("step_results", [])
        results.append({
            "step": step_number,
            "description": event.description,
            "status": event.status,
            "result": event.result,
            "error": event.error
        })
        await ctx.set("step_results", results)
        
        # Check if there are more steps
        if step_number < len(steps):
            # Execute next step
            next_step = steps[step_number]  # 0-based indexing for steps list
            return ExecuteStepEvent(
                plan_id=plan_id,
                step_number=step_number + 1,
                description=next_step.get("description", ""),
                tool_name=next_step.get("tool"),
                server_name=next_step.get("server"),
                arguments=next_step.get("args", {})
            )
        else:
            # All steps completed, generate recommendations
            task = await ctx.get("task", "")
            return GenerateRecommendationsEvent(
                plan_id=plan_id,
                task=task,
                results=results
            )
    
    @step
    async def generate_recommendations(self, ctx: WorkflowContext, event: GenerateRecommendationsEvent) -> StopEvent:
        """
        Generate recommendations based on the results.
        
        Args:
            ctx: Workflow context
            event: Generate recommendations event
            
        Returns:
            StopEvent with recommendations
        """
        plan_id = event.plan_id
        task = event.task
        results = event.results
        
        # Get the plan
        plan = await ctx.get("plan", {})
        
        # Format results for the LLM
        formatted_results = self._format_execution_results(results)
        
        # Create recommendation prompt
        prompt = f"""
        Based on the results of your investigation, provide specific recommendations for this task:
        
        Original task: {task}
        
        Your plan:
        {str(plan)}
        
        Execution results:
        {formatted_results}
        
        Provide clear, specific recommendations based on what you found. Include:
        1. A summary of what you discovered
        2. Specific recommendations for improvements
        3. Any code examples if relevant
        
        Make your response conversational and helpful. Focus on providing actionable insights.
        """
        
        # Get recommendations from LLM
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await ctx.llm_client.get_completion(messages, [])
            
            recommendation_text = response.content or ""
            
            logger.info("Generated recommendations")
            
            # Return the recommendations
            return StopEvent(result=recommendation_text)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ErrorEvent.from_exception(e, step_name="generate_recommendations")
    
    def _format_execution_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format execution results as text.
        
        Args:
            results: List of step results
            
        Returns:
            Formatted results text
        """
        formatted_results = []
        
        for result in results:
            step = result.get("step")
            description = result.get("description")
            status = result.get("status")
            
            step_text = f"Step {step}: {description} - {status}"
            
            if status == "error" and result.get("error"):
                step_text += f"\nError: {result['error']}"
            
            if result.get("result"):
                tool_result = result["result"]
                # Truncate long results
                if len(tool_result) > 1000:
                    tool_result = tool_result[:997] + "..."
                step_text += f"\nResult: {tool_result}"
            
            formatted_results.append(step_text)
        
        return "\n\n".join(formatted_results)