"""
Event-driven workflow system.

This package provides a framework for building event-driven workflows that
can be used to orchestrate complex tasks and interactions with LLMs, tools,
and external services.

Key components:
- Events: Messages that trigger steps
- Steps: Processing units that handle events and produce new events
- Workflows: Collections of steps that work together
- Context: State and service access for steps
- Visualizer: Tools for visualizing workflow structure
"""

# Import core components for easy access
from .events import BaseEvent, StartEvent, StopEvent, ErrorEvent, MessageEvent, ToolEvent, Event
from .context import WorkflowContext
from .step import Step, step
from .workflow import Workflow, WorkflowResult
from .visualizer import (
    WorkflowVisualizer, 
    draw_workflow, 
    draw_all_possible_flows,
    draw_most_recent_execution
)

# Export these symbols
__all__ = [
    # Event types
    'BaseEvent',
    'StartEvent',
    'StopEvent',
    'ErrorEvent',
    'MessageEvent',
    'ToolEvent',
    'Event',
    
    # Workflow components
    'WorkflowContext',
    'Step',
    'step',
    'Workflow',
    'WorkflowResult',
    
    # Visualization tools
    'WorkflowVisualizer',
    'draw_workflow',
    'draw_all_possible_flows',
    'draw_most_recent_execution',
]