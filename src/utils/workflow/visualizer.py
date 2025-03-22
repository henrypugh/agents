"""
Workflow visualization utilities.

This module provides tools for visualizing workflow structure and event flow
to help with debugging and understanding complex workflows.
"""

import json
import os
from typing import Dict, List, Any, Type, Optional, Set, Union

from .workflow import Workflow
from .step import Step
from .events import BaseEvent

class WorkflowVisualizer:
    """
    Utility for visualizing workflow structure and execution.
    
    This class provides methods to generate graph data and HTML visualizations
    for workflow classes, showing steps, events, and their relationships.
    """
    
    @staticmethod
    def generate_graph_data(workflow_class: Type[Workflow]) -> Dict[str, Any]:
        """
        Generate graph data for a workflow class.
        
        This method analyzes a workflow class to extract its steps and event types,
        and generates a graph data structure that can be used for visualization.
        
        Args:
            workflow_class: Workflow class to analyze
            
        Returns:
            Dictionary containing nodes and edges for visualization
        """
        # Create a dummy agent for analysis
        class MockAgent:
            def __init__(self):
                self.llm_client = None
                self.server_manager = None
                self.tool_processor = None
                self.conversation_manager = None
        
        # Create an instance to analyze (without running it)
        workflow_instance = workflow_class(MockAgent())
        
        nodes = []
        edges = []
        
        # Track event types and step nodes we've created
        event_nodes = {}  # event_type_name -> node_id
        step_nodes = {}   # step_name -> node_id
        
        # Collect all steps
        all_steps = []
        
        # Look for Step objects defined as attributes
        for attr_name in dir(workflow_instance):
            if attr_name.startswith('_'):
                continue
                
            attr = getattr(workflow_instance, attr_name)
            if isinstance(attr, Step):
                all_steps.append(attr)
        
        # Add steps from class definition if available
        if hasattr(workflow_class, '_steps'):
            all_steps.extend(workflow_class._steps)
        
        # Add nodes for each step
        for step in all_steps:
            step_id = f"step_{step.name}"
            step_nodes[step.name] = step_id
            
            nodes.append({
                "data": {
                    "id": step_id,
                    "label": step.name,
                    "type": "step"
                }
            })
        
        # Now create event nodes and build connections
        for step in all_steps:
            step_id = step_nodes[step.name]
            
            # Add input event types and their connections to this step
            for event_type in step.input_event_types:
                event_type_name = event_type.__name__
                
                # Create event node if it doesn't exist
                if event_type_name not in event_nodes:
                    event_id = f"event_{event_type_name}"
                    event_nodes[event_type_name] = event_id
                    
                    nodes.append({
                        "data": {
                            "id": event_id,
                            "label": event_type_name,
                            "type": "event"
                        }
                    })
                else:
                    event_id = event_nodes[event_type_name]
                
                # Add edge from event to step
                edge_id = f"{event_id}_to_{step_id}"
                # Check if edge already exists
                if not any(e["data"]["id"] == edge_id for e in edges):
                    edges.append({
                        "data": {
                            "id": edge_id,
                            "source": event_id,
                            "target": step_id,
                            "label": "triggers"
                        }
                    })
            
            # Add output event types and their connections from this step
            for event_type in step.output_event_types:
                # Skip None type (representing steps that don't return events)
                if event_type is type(None):
                    continue
                    
                event_type_name = event_type.__name__
                
                # Create event node if it doesn't exist
                if event_type_name not in event_nodes:
                    event_id = f"event_{event_type_name}"
                    event_nodes[event_type_name] = event_id
                    
                    nodes.append({
                        "data": {
                            "id": event_id,
                            "label": event_type_name,
                            "type": "event"
                        }
                    })
                else:
                    event_id = event_nodes[event_type_name]
                
                # Add edge from step to event
                edge_id = f"{step_id}_to_{event_id}"
                # Check if edge already exists
                if not any(e["data"]["id"] == edge_id for e in edges):
                    edges.append({
                        "data": {
                            "id": edge_id,
                            "source": step_id,
                            "target": event_id,
                            "label": "produces"
                        }
                    })
        
        return {
            "nodes": nodes,
            "edges": edges
        }
    
    @staticmethod
    def generate_html(workflow_class: Type[Workflow], filename: Optional[str] = None) -> str:
        """
        Generate HTML visualization for a workflow.
        
        This method creates an interactive HTML visualization of a workflow
        using Cytoscape.js, and optionally saves it to a file.
        
        Args:
            workflow_class: Workflow class to visualize
            filename: Optional filename to save the HTML
            
        Returns:
            HTML string containing the visualization
        """
        # Generate graph data
        graph_data = WorkflowVisualizer.generate_graph_data(workflow_class)
        
        # Calculate statistics
        step_count = len([n for n in graph_data['nodes'] if n['data']['type'] == 'step'])
        event_count = len([n for n in graph_data['nodes'] if n['data']['type'] == 'event'])
        
        # Create HTML template with Cytoscape.js
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Workflow Visualization: {workflow_class.__name__}</title>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f9f9f9;
                }}
                #cy {{
                    width: 100%;
                    height: 100vh;
                    position: absolute;
                    top: 0;
                    left: 0;
                }}
                .info-panel {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background-color: rgba(255, 255, 255, 0.9);
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    padding: 15px;
                    z-index: 10;
                    max-width: 300px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                .legend {{
                    margin-top: 15px;
                    padding-top: 10px;
                    border-top: 1px solid #ddd;
                }}
                .legend-item {{
                    margin: 5px 0;
                    display: flex;
                    align-items: center;
                }}
                .legend-color {{
                    width: 15px;
                    height: 15px;
                    margin-right: 8px;
                    border-radius: 3px;
                }}
                h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                h3 {{
                    margin-bottom: 10px;
                    color: #555;
                }}
            </style>
        </head>
        <body>
            <div id="cy"></div>
            <div class="info-panel">
                <h2>Workflow: {workflow_class.__name__}</h2>
                <p><strong>Steps:</strong> {step_count}</p>
                <p><strong>Events:</strong> {event_count}</p>
                <p><em>Drag to move, scroll to zoom</em></p>
                
                <div class="legend">
                    <h3>Legend</h3>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #6FB1FC;"></div>
                        <div>Step (Rectangle)</div>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #F5A45D;"></div>
                        <div>Event (Oval)</div>
                    </div>
                </div>
            </div>
            
            <!-- Load required libraries -->
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.25.0/cytoscape.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape-dagre/2.5.0/cytoscape-dagre.min.js"></script>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    // Register the dagre layout for directed graphs
                    cytoscape.use(cytoscapeDagre);
                    
                    var cy = cytoscape({{
                        container: document.getElementById('cy'),
                        elements: {json.dumps(graph_data)},
                        style: [
                            {{
                                selector: 'node',
                                style: {{
                                    'label': 'data(label)',
                                    'text-valign': 'center',
                                    'text-halign': 'center',
                                    'font-size': '14px',
                                    'width': 'label',
                                    'height': 'label',
                                    'padding': '12px',
                                    'text-wrap': 'wrap',
                                    'text-max-width': '120px',
                                    'color': '#333',
                                    'font-weight': 'bold'
                                }}
                            }},
                            {{
                                selector: 'node[type="step"]',
                                style: {{
                                    'background-color': '#6FB1FC',
                                    'shape': 'rectangle',
                                    'padding': '15px',
                                    'border-width': '1px',
                                    'border-color': '#4A90E2'
                                }}
                            }},
                            {{
                                selector: 'node[type="event"]',
                                style: {{
                                    'background-color': '#F5A45D',
                                    'shape': 'ellipse',
                                    'border-width': '1px',
                                    'border-color': '#E67E22'
                                }}
                            }},
                            {{
                                selector: 'edge',
                                style: {{
                                    'width': 2,
                                    'line-color': '#999',
                                    'target-arrow-color': '#999',
                                    'target-arrow-shape': 'triangle',
                                    'curve-style': 'bezier',
                                    'arrow-scale': 1.2
                                }}
                            }}
                        ],
                        layout: {{
                            name: 'dagre',
                            rankDir: 'LR',  // Left to right layout
                            padding: 50,
                            nodeSep: 60,
                            rankSep: 100,
                            animate: true,
                            animationDuration: 500
                        }}
                    }});
                    
                    // Center the graph
                    cy.fit();
                    cy.center();
                    
                    // Add interactivity - nodes highlight on hover
                    cy.on('mouseover', 'node', function(e) {{
                        var node = e.target;
                        node.style('border-width', '3px');
                        
                        // Highlight connected edges and nodes
                        var connectedEdges = node.connectedEdges();
                        connectedEdges.style('line-color', '#2ECC71');
                        connectedEdges.style('target-arrow-color', '#2ECC71');
                        connectedEdges.style('width', 3);
                    }});
                    
                    cy.on('mouseout', 'node', function(e) {{
                        var node = e.target;
                        node.style('border-width', '1px');
                        
                        // Reset connected edges and nodes
                        var connectedEdges = node.connectedEdges();
                        connectedEdges.style('line-color', '#999');
                        connectedEdges.style('target-arrow-color', '#999');
                        connectedEdges.style('width', 2);
                    }});
                }});
            </script>
        </body>
        </html>
        """
        
        # Optionally save to file
        if filename:
            with open(filename, 'w') as f:
                f.write(html)
            print(f"Workflow visualization saved to {filename}")
        
        return html


def draw_workflow(workflow_class: Type[Workflow], filename: Optional[str] = None) -> str:
    """
    Draw a workflow and optionally save to file.
    
    Args:
        workflow_class: Workflow class to visualize
        filename: Optional filename to save the HTML
        
    Returns:
        HTML string containing the visualization
    """
    return WorkflowVisualizer.generate_html(workflow_class, filename)


def draw_all_possible_flows(workflow_class: Type[Workflow], filename: Optional[str] = None) -> str:
    """
    Draw all possible flows through a workflow.
    
    This is an alias for draw_workflow for compatibility with existing code.
    
    Args:
        workflow_class: Workflow class to visualize
        filename: Optional filename to save the HTML
        
    Returns:
        HTML string containing the visualization
    """
    return draw_workflow(workflow_class, filename)


def draw_most_recent_execution(workflow_instance: Workflow, filename: Optional[str] = None) -> str:
    """
    Draw the most recent execution of a workflow.
    
    This is a simplified version that just shows the workflow structure,
    since tracking the actual execution path would require more instrumentation.
    
    Args:
        workflow_instance: Workflow instance that was executed
        filename: Optional filename to save the HTML
        
    Returns:
        HTML string containing the visualization
    """
    return WorkflowVisualizer.generate_html(workflow_instance.__class__, filename)