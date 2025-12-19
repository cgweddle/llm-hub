"""
Flow Executor
Executes database flows
"""

import sys
import os
import json
import logging
from typing import Dict, Any, Callable, Optional, List
from sqlalchemy.orm import Session
from src.database.database_setup import Flow, Tool
from src.executors.tool_executor import create_executable_function, create_conda_executable_function
from src.utils import get_llm_config_by_name



logger = logging.getLogger(__name__)

class FlowExecutor:
    def __init__(self, session: Session, flow_id: int):
        self.session = session
        self.flow_id = flow_id

        #Load flow from the database
        flow = self.session.query(Flow).filter(Flow.id == flow_id).first()
        if not flow:
            raise ValueError(f"Flow {flow_id} not found")
        self.flow = flow
        self.graph_config = flow.graph_config

        # Caches
        self.executable_functions = {}
        self.tools_cache = {}
        self.execution_trace = []

    def _prepare_tools(self, start_node: Optional[int]):
        nodes_config = self.graph_config["nodes"]
        edges_config = self.graph_config["edges"]

        nodes_to_reload = set()
        if start_node:
            nodes_to_reload = self._get_downstream_nodes(start_node, edges_config)
            nodes_to_reload.add(start_node)
            for node_name in nodes_to_reload:
                node_info = nodes_config.get(node_name)
                
                tool_id = node_info['tool_id']
                self.tools_cache.pop(tool_id, None)
                self.executable_functions.pop(node_name, None)
                logger.info(f"Cleared cache for tool {tool_id}")
        
        # Load / reload tools
        for node_name, node_info in nodes_config.items():
            # Support both 'id' and 'tool_id' keys for backwards compatibility
            tool_id = node_info.get('id') or node_info.get('tool_id')

            # Skip if its not a node to reload
            if start_node and node_name not in nodes_to_reload:
                continue

            tool = self.session.query(Tool).filter(Tool.id == tool_id).first()
            if not tool:
                raise ValueError(f"Tool with ID {tool_id} not found in database")

            # If there is a conda environment associated with the flow, run with that
            if self.conda_env:
                func = create_conda_executable_function(tool, self.conda_env)
            else:
                func = create_executable_function(tool)

            # Load LLM configuration if specified for this node
            llm_config = None
            model_name = node_info.get('model_name')
            if model_name:
                try:
                    llm_config = get_llm_config_by_name(model_name)
                    if llm_config:
                        logger.info(f"Loaded LLM config '{model_name}' for node {node_name}")
                    else:
                        logger.warning(f"LLM config '{model_name}' not found in ~/.llm_hub/config.yaml for node {node_name}")
                except Exception as e:
                    logger.error(f"Failed to load LLM config '{model_name}': {e}")

            self.tools_cache[tool_id] = tool
            self.executable_functions[node_name] = {
                "function": func,
                "tool": tool,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "llm_config": llm_config  # Store LLM config for tools that need it
            }

            if llm_config:
                logger.info(f"Prepared tool: {tool.name} for node {node_name} with LLM: {llm_config.get('name')}")
            else:
                logger.info(f"Prepared tool: {tool.name} for node {node_name}")

    def _get_downstream_nodes(self, start_node: str, edges_config: List[Dict]):
        downstream_nodes = set()
        to_visit = [start_node]
        while to_visit:
            current = to_visit.pop()

            # Find all edges from the current node
            for edge in edges_config:
                if edge['from_node'] == current:
                    next_node = edge['to_node']
                    if next_node not in downstream_nodes:
                        downstream_nodes.add(next_node)
                        to_visit.append(next_node)
        
        return downstream_nodes
    
    def _execute_node(self, node_name: str, input_data: Any):
        if node_name not in self.executable_functions:
            raise ValueError(f"Node {node_name} not found in executable functions")
        
        node_info = self.executable_functions[node_name]
        func = node_info["function"]

        logger.info(f"Executing node: {node_name}")

        try:
            #Execute function
            output = func(**input_data)

            # Record
            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": output,
                "status": "success"
            })
            logger.info(f"Node {node_name} completed successfully")
            return output
        
        except Exception as e:
            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": None,
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Node {node_name} failed: {e}")
            raise

    def _find_next_node(self, current_node: str) -> List[str]:
        """Find the next nodes in the flow"""
        next_nodes = []
        edges_config = self.graph_config["edges"]
        for edge in edges_config:
            if edge['from_node'] == current_node:
                next_nodes.append(edge['to_node'])
        return next_nodes
    
    def _apply_mapping(self, output_data: Any, mapping: Optional[Dict[str, str]], target_input_schema: Dict) -> Dict[str, Any]:
        """
        Apply mapping form source output to target input 

        Args:
            output_data: Output from previous tool
            mapping: [Optional] Field mapping (dict) (or None)
            target_input_schema: Input schema of target tool

        Returns:
            Dict ready to be passed as **kwargs to target tool
        """

        if mapping is None:
            return output_data
        
        # apply field mapping
        mapped_input = {}

        for output_field, input_param in mapping.items():
            if output_field in output_data:
                mapped_input[input_param] = output_data[output_field]
            else:
                logger.warning(f"Output field '{output_field} not found in output data")
        
        return mapped_input
    
    def _execute_from_node(self, start_node: str, initial_input: Any) -> Any:
        """
        Execute flow from a specific node

        Args:
            start_node: Node name to start from
            initial_input: Input data for the start node
        Returns:
            Output from final node
        """
        current_node = start_node
        current_input = initial_input

        while True:
            #Execute node
            output = self._execute_node(current_node, current_input)

            #Fine next nodes
            next_nodes = self._find_next_nodes(current_node)

            #If there's not a next node, exit and return
            if not next_nodes:
                logger.info(f"Reached exit at node: {current_node}")
                return output

            #Just for sequential nodes
            #Implement parallelism for parallel processes
            next_node = next_nodes[0]
            if len(next_nodes) > 1:
                logger.warning(f"Multiple next nodes, just using first: {next_node}")
            
            #Find edge between current and next node
            edge_mapping = None
            for edge in self.graph_config['edges']:
                if edge['from_nodeds'] == current_node and edge['to_node'] == next_node:
                    edge_mapping = edge.get('mapping')
                    break

            target_node_info = self.executable_functions[next_node]
            target_input_schema = target_node_info.get("input_schema", {})

            #Apply mapping 
            current_input = self._apply_mapping(output, edge_mapping, target_input_schema)

            #Move to the next node
            current_node = next_node


    def execute_flow(self, initial_input: Any, conda_env: str):
        """
        Execute a flow from the database
        
        Returns:
            {
                "flow_id": int,
                "status": "completed" | "failed",
                "final_output": Any,
                "execution_trace": [
                    {"node": str, "input": Any, "output": Any},
                    ...
                ]
                "error": str(optional)
            }
        """
        try:
            self.execution_trace = []
            self.conda_env = conda_env

            #Prepare tools
            self._prepare_tools()

            #Execute from entry point
            entry_point = self.graph_config["entry_point"]
            final_output = self._execute_from_node(entry_point, initial_input)

            return {
                "flow_id": self.flow_id,
                "status": "completed",
                "final_output": final_output,
                "execution_trace": self.execution_trace
            }
        
        except Exception as e:
            logger.error(f"Flow execution failed: {e}")
            return {
                "flow_id": self.flow_id,
                "status": "failed",
                "final_output": None,
                "execution_trace": self.execution_trace,
                "error": str(e)
            }



    
    def resume_flow(self, flow_id: int, execution_trace: List[Dict], resume_input: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Resume flow from the last successful node trace
        Optinally provide a new input to the failed node

        Args:
            execution_trace: Execution trace from run
            resume_input: [Optional] Input to override for the failed node
        Returns:
            FlowExecutionResult dict (same as execution_flow)
        """
        try:
            # Restore the execution trace
            self.execution_trace = execution_trace.copy()

            # Find the node that failed
            failed_node = None
            failed_input = None

            for record in execution_trace:
                if record["status"] == "failed":
                    failed_node = record["node"]
                    failed_input = record["input"]
                    break
                
                if not failed_node:
                    raise ValueError("No failed node found")
                
                self.execution_trace = [r for r in self.execution_trace if r["status"] != "failed"]
                
                #Inputs into node
                if resume_input is not None:
                    input_data = resume_input
                else:
                    input_data = failed_input

                #Reload tools from the failed node onwards
                self._prepare_tools(start_node=failed_node)

                final_output = self._execute_from_node(failed_node, input_data)

                return {
                    "flow_id": self.flow_id,
                    "status": "completed",
                    "final_output": final_output,
                    "execution_trace": self.execution_trace
                }
            
        except Exception as e:
            logger.error(f"Flow resume failed: {e}")
            return {
                "flow_id": self.flow_id,
                "status": "failed",
                "final_output": None,
                "execution_trace": self.execution_trace,
                "error": str(e)
            }
        

