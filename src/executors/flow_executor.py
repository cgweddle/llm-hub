"""
Flow Executor
Executes database flows (DAGs of tool and agent nodes).
Records the full execution tree in the database.
"""

import sys
import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
from sqlalchemy.orm import Session
from src.database.database_setup import Flow, Tool, Execution
from src.database.database import create_execution, update_execution, get_execution_by_id
from src.executors.tool_executor import create_executable_function



logger = logging.getLogger(__name__)

class FlowExecutor:
    def __init__(self, session: Session, flow_id: int, user_id: int, llm_config: Optional[Dict] = None):
        self.session = session
        self.flow_id = flow_id
        self.user_id = user_id
        self.llm_config = llm_config or {"models": []}

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

        # DB execution tracking
        self.root_execution = None
        self._step_sequence = 0

    def _next_sequence(self) -> int:
        """Get and increment the step sequence counter."""
        seq = self._step_sequence
        self._step_sequence += 1
        return seq

    def _prepare_tools(self, start_node: Optional[int] = None):
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
            # Skip agent nodes — they delegate to AgentExecutor at runtime
            if node_info.get('node_type') == 'agent':
                continue

            # Skip trigger nodes — they provide initial input, not executable functions
            if node_info.get('node_type') == 'trigger':
                continue

            # Support both 'id' and 'tool_id' keys for backwards compatibility
            tool_id = node_info.get('id') or node_info.get('tool_id')

            # Skip if its not a node to reload
            if start_node and node_name not in nodes_to_reload:
                continue

            tool = self.session.query(Tool).filter(Tool.id == tool_id).first()
            if not tool:
                raise ValueError(f"Tool with ID {tool_id} not found in database")

            func = create_executable_function(tool, conda_env=self.conda_env)

            # Load LLM configuration if specified for this node
            node_llm_config = None
            model_name = node_info.get('model_name')
            if model_name:
                # Look up from the pre-loaded llm_config dict
                for m in self.llm_config.get("models", []):
                    if m.get("name") == model_name:
                        node_llm_config = m.copy()
                        node_llm_config['config_name'] = model_name
                        logger.info(f"Loaded LLM config '{model_name}' for node {node_name}")
                        break
                if not node_llm_config:
                    logger.warning(f"LLM config '{model_name}' not found for node {node_name}")

            self.tools_cache[tool_id] = tool
            self.executable_functions[node_name] = {
                "function": func,
                "tool": tool,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "llm_config": node_llm_config  # Store LLM config for tools that need it
            }

            if node_llm_config:
                logger.info(f"Prepared tool: {tool.name} for node {node_name} with LLM: {node_llm_config.get('name')}")
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
        tool = node_info["tool"]

        ## Set llm config for the specific node
        llm_config = node_info.get("llm_config")
        if llm_config:
            self._setup_llm_environment(llm_config=llm_config)

        logger.info(f"Executing node: {node_name}")

        # Create child execution record for this tool node
        step_exec = create_execution(
            self.session,
            parent_id=self.root_execution.id,
            user_id=self.user_id,
            tool_id=tool.id,
            execution_type='tool',
            node_id=node_name,
            name=tool.name,
            sequence=self._next_sequence(),
            input_data=input_data,
            status='running',
            started_at=datetime.now()
        )

        try:
            #Execute function
            output = func(**input_data)

            # Update DB record
            update_execution(self.session, step_exec.id,
                status='completed',
                output_data=output,
                completed_at=datetime.now()
            )

            # Keep in-memory trace for response compatibility
            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": output,
                "status": "success"
            })
            logger.info(f"Node {node_name} completed successfully")
            return output

        except Exception as e:
            update_execution(self.session, step_exec.id,
                status='failed',
                error_message=str(e),
                completed_at=datetime.now()
            )
            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": None,
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Node {node_name} failed: {e}")
            raise

    def _setup_llm_environment(self, llm_config: Dict) -> Dict[str, Optional[str]]:
        """
        Setup environment variables for calling the LLM from functions
        """
        provider = llm_config["provider"]
        model = llm_config["model"]
        api_key = llm_config.get("api_key")
        base_url = llm_config.get("base_url")
        config_name = llm_config.get("config_name")

        # Pass the config name for subprocess to look up in config.yaml
        if config_name:
            os.environ["LLMHUB_CONFIG_NAME"] = config_name
        os.environ["LLMHUB_MODEL_NAME"] = model
        if provider == "anthropic":
            if api_key:
              os.environ["ANTHROPIC_API_KEY"] = api_key
            if base_url:
                os.environ["ANTHROPIC_BASE_URL"] = base_url
        elif provider == "openai":
            if api_key:
              os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
        elif provider == "llmstudio":
            os.environ["OPENAI_API_KEY"] = api_key or "lm-studio"  # LM Studio needs a dummy key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
        elif provider=="azure":
            if api_key:
                os.environ["AZURE_API_KEY"] = api_key
            if base_url:
                os.environ["AZURE_API_BASE"] = base_url



    def _find_next_node(self, current_node: str) -> List[str]:
        """Find the next nodes in the flow"""
        next_nodes = []
        edges_config = self.graph_config["edges"]
        for edge in edges_config:
            if edge['from_node'] == current_node:
                next_nodes.append(edge['to_node'])
        return next_nodes

    def _apply_mapping(self, output_data: Any, mapping: Optional[Dict[str, str]], target_input_schema: Dict, base_input_values: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply mapping form source output to target input

        Args:
            output_data: Output from previous tool
            mapping: [Optional] Field mapping (dict) (or None)
            target_input_schema: Input schema of target tool
            base_input_values: [Optional] User-entered values for unconnected parameters

        Returns:
            Dict ready to be passed as **kwargs to target tool
        """

        # Start with base input values (user-entered values for unconnected params)
        result = base_input_values.copy() if base_input_values else {}

        if mapping is None:
            # Passthrough: merge output_data with base values (output_data takes precedence)
            if isinstance(output_data, dict):
                result.update(output_data)
            else:
                # If output is not a dict, we can't merge - just return output
                return output_data
        else:
            # Apply field mapping (overrides base values for connected params)
            for output_field, input_param in mapping.items():
                # Special case: empty string means "whole output"
                if output_field == "":
                    result[input_param] = output_data
                    logger.info(f"Mapping entire output to parameter '{input_param}'")
                # Normal case: map specific field to parameter
                elif output_field in output_data:
                    result[input_param] = output_data[output_field]
                else:
                    logger.warning(f"Output field '{output_field}' not found in output data")

        return result

    def _execute_agent_node(self, node_name: str, input_data: Any) -> str:
        """
        Execute an agent node by delegating to AgentExecutor.

        Agent nodes receive text input and produce text output.
        If input_data is a dict, it's serialized to a string for the agent.
        """
        from src.executors.agent_executor import AgentExecutor

        node_config = self.graph_config["nodes"][node_name]
        agent_id = node_config["id"]
        llm_provider = node_config.get("model_name", "")

        # Convert dict input to text for agent consumption
        if isinstance(input_data, dict):
            input_text = json.dumps(input_data, indent=2)
        else:
            input_text = str(input_data)

        logger.info(f"Executing agent node: {node_name} (agent_id={agent_id})")

        # Create child execution record for this agent node
        agent_exec = create_execution(
            self.session,
            parent_id=self.root_execution.id,
            user_id=self.user_id,
            agent_id=agent_id,
            execution_type='agent',
            node_id=node_name,
            name=node_config.get('name', f'Agent {agent_id}'),
            sequence=self._next_sequence(),
            input_data={"input": input_text},
            status='running',
            started_at=datetime.now()
        )

        try:
            executor = AgentExecutor(self.session, llm_config=self.llm_config)
            # Run the async method synchronously, passing parent execution for sub-recording
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(
                            asyncio.run,
                            executor.execute_agent_node(agent_id, input_text, self.session, llm_provider=llm_provider, parent_execution=agent_exec)
                        )
                        result = future.result()
                else:
                    result = loop.run_until_complete(
                        executor.execute_agent_node(agent_id, input_text, self.session, llm_provider=llm_provider, parent_execution=agent_exec)
                    )
            except RuntimeError:
                result = asyncio.run(
                    executor.execute_agent_node(agent_id, input_text, self.session, llm_provider=llm_provider, parent_execution=agent_exec)
                )

            update_execution(self.session, agent_exec.id,
                status='completed',
                output_data={"result": result},
                completed_at=datetime.now()
            )

            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": result,
                "status": "success"
            })
            logger.info(f"Agent node {node_name} completed successfully")
            return result

        except Exception as e:
            update_execution(self.session, agent_exec.id,
                status='failed',
                error_message=str(e),
                completed_at=datetime.now()
            )
            self.execution_trace.append({
                "node": node_name,
                "input": input_data,
                "output": None,
                "status": "failed",
                "error": str(e)
            })
            logger.error(f"Agent node {node_name} failed: {e}")
            raise

    def _is_agent_node(self, node_name: str) -> bool:
        """Check if a node is an agent node."""
        node_config = self.graph_config["nodes"].get(node_name, {})
        return node_config.get("node_type") == "agent"

    def _execute_from_node(self, start_node: str, initial_input: Any) -> Any:
        """
        Execute flow from a specific node.
        Supports both tool nodes and agent nodes.

        Args:
            start_node: Node name to start from
            initial_input: Input data for the start node
        Returns:
            Output from final node
        """
        current_node = start_node
        current_input = initial_input

        while True:
            node_config = self.graph_config["nodes"].get(current_node, {})

            # Trigger node: extract text value and pass to next node
            if node_config.get('node_type') == 'trigger':
                trigger_value = node_config.get('input_value', '')
                if trigger_value:
                    current_input = trigger_value

                # Record trigger as child execution
                create_execution(
                    self.session,
                    parent_id=self.root_execution.id,
                    user_id=self.user_id,
                    execution_type='trigger',
                    node_id=current_node,
                    name=node_config.get('name', 'Trigger'),
                    sequence=self._next_sequence(),
                    input_data=None,
                    output_data={"value": current_input},
                    status='completed',
                    started_at=datetime.now(),
                    completed_at=datetime.now()
                )

                self.execution_trace.append({
                    "node": current_node,
                    "input": None,
                    "output": current_input,
                    "status": "success"
                })

                next_nodes = self._find_next_node(current_node)
                if not next_nodes:
                    return current_input

                next_node = next_nodes[0]

                # Apply mapping from trigger to next node
                edge_mapping = None
                for edge in self.graph_config['edges']:
                    if edge['from_node'] == current_node and edge['to_node'] == next_node:
                        edge_mapping = edge.get('mapping')
                        break

                if edge_mapping:
                    result = {}
                    for output_field, input_param in edge_mapping.items():
                        result[input_param] = current_input
                    current_input = result
                elif isinstance(current_input, str):
                    # Auto-map: if next node expects named params, wrap in dict
                    next_node_info = self.executable_functions.get(next_node, {})
                    next_schema = next_node_info.get("input_schema", {})
                    if next_schema:
                        props = next_schema.get('properties', next_schema)
                        params = list(props.keys())
                        if len(params) == 1:
                            current_input = {params[0]: current_input}

                current_node = next_node
                continue

            # Execute node — dispatch based on node type
            if self._is_agent_node(current_node):
                output = self._execute_agent_node(current_node, current_input)
            else:
                output = self._execute_node(current_node, current_input)

            # Find next nodes
            next_nodes = self._find_next_node(current_node)

            # If there's not a next node, exit and return
            if not next_nodes:
                logger.info(f"Reached exit at node: {current_node}")
                return output

            # Just for sequential nodes
            next_node = next_nodes[0]
            if len(next_nodes) > 1:
                logger.warning(f"Multiple next nodes, just using first: {next_node}")

            # Prepare input for next node based on node types
            current_is_agent = self._is_agent_node(current_node)
            next_is_agent = self._is_agent_node(next_node)

            if current_is_agent and next_is_agent:
                # Agent→Agent: text passthrough
                current_input = output
            elif current_is_agent and not next_is_agent:
                # Agent→Tool: assign agent text output to mapped input parameters
                edge_mapping = None
                for edge in self.graph_config['edges']:
                    if edge['from_node'] == current_node and edge['to_node'] == next_node:
                        edge_mapping = edge.get('mapping')
                        break

                next_node_config = self.graph_config['nodes'].get(next_node, {})
                base_input_values = next_node_config.get('input_values', {})
                target_node_info = self.executable_functions.get(next_node, {})
                target_input_schema = target_node_info.get("input_schema", {})

                if edge_mapping:
                    # Use mapping: agent output is text, map it to specified param(s)
                    result = base_input_values.copy() if base_input_values else {}
                    for output_field, input_param in edge_mapping.items():
                        result[input_param] = output
                    current_input = result
                elif target_input_schema:
                    # Auto-detect: assign to first string parameter
                    result = base_input_values.copy() if base_input_values else {}
                    props = target_input_schema.get("properties", target_input_schema)
                    for param_name, param_info in props.items():
                        if param_name not in result:
                            result[param_name] = output
                            break
                    current_input = result
                else:
                    current_input = {"input": output}
            elif not current_is_agent and next_is_agent:
                # Tool→Agent: serialize tool output dict to text for agent
                # Mapping is ignored — agent gets text
                current_input = output
            else:
                # Tool→Tool: standard field-level mapping
                edge_mapping = None
                for edge in self.graph_config['edges']:
                    if edge['from_node'] == current_node and edge['to_node'] == next_node:
                        edge_mapping = edge.get('mapping')
                        break

                target_node_info = self.executable_functions[next_node]
                target_input_schema = target_node_info.get("input_schema", {})

                next_node_config = self.graph_config['nodes'].get(next_node, {})
                base_input_values = next_node_config.get('input_values', {})

                current_input = self._apply_mapping(output, edge_mapping, target_input_schema, base_input_values)

            # Move to the next node
            current_node = next_node

    def _generate_combined_flow_script(self, initial_input: Any) -> str:
        nodes_config = self.graph_config["nodes"]
        edges_config = self.graph_config["edges"]
        entry_point = self.graph_config["entry_point"]

        # Collect __future__ imports and tool nodes
        future_imports = []
        tool_code_blocks = []
        node_to_function = {}

        for node_id, node_info in nodes_config.items():
            tool_id = node_info.get('id')
            tool = self.session.query(Tool).filter(Tool.id==tool_id).first()

            # Separate __future__ imports from code
            code_lines = []
            for line in tool.script_code.split('\n'):
                if line.strip().startswith('from __future__'):
                    if line not in future_imports:
                        future_imports.append(line)
                else:
                    code_lines.append(line)
            tool_code_blocks.append('\n'.join(code_lines))
            node_to_function[node_id] = tool.main_function

        ## Build the entire script
        script_parts = []

        ## Put future imports at the top
        if future_imports:
            script_parts.extend(future_imports)

        ## Add the tool code
        script_parts.append('# Tool functions')
        for code_block in tool_code_blocks:
            script_parts.append(code_block)

        ## Flow execution
        script_parts.append('# Flow execution')
        script_parts.append(f'initial_input = {json.dumps(initial_input)}')

        execution_order = []
        current_node = entry_point
        visited=set()

        while current_node and current_node not in visited:
            execution_order.append(current_node)
            visited.add(current_node)

            # Find next node
            next_node = None
            for edge in edges_config:
                if edge["from_node"] == current_node:
                    next_node = edge["to_node"]
                    break
            current_node = next_node

        # Generate function calls
        for i, node_id in enumerate(execution_order):
            func_name = node_to_function[node_id]

            if i == 0:
                # For the first node
                script_parts.append(f"output_{i} = {func_name}(**initial_input)")
            else:
                prev_node = execution_order[i-1]
                edge = next((e for e in edges_config
                             if e['from_node']==prev_node and e['to_node']==node_id), None)

                if edge and edge.get('mapping'):
                    ## Apply the field mapping
                    mapping = edge['mapping']
                    mapped_args = ', '.join([f'{in_param}=output_{i-1}["{out_field}"]'
                                            for out_field, in_param in mapping.items()])
                    script_parts.append(f'output_{i} = {func_name}({mapped_args})')
                else:
                    # Pass entire output
                    script_parts.append(f'output_{i} = {func_name}(**output_{i-1})')

        script_parts.append('')
        script_parts.append(f'final_output = output_{len(execution_order)-1}')

        return '\n'.join(script_parts)


    def execute_flow(self, initial_input: Any, conda_env: str, execution_id: Optional[int] = None):
        """
        Execute a flow from the database.
        Creates a top-level Execution record and child records for each node.

        Args:
            initial_input: Trigger input for the flow
            conda_env: Optional conda environment path for tool nodes
            execution_id: Optional pre-existing Execution row ID. When provided
                (e.g. by a Celery task dispatched from the API), the flow
                reuses that row and transitions it to 'running' instead of
                creating a new one.

        Returns:
            {
                "flow_id": int,
                "execution_id": int,
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
            self._step_sequence = 0
            self.conda_env = conda_env

            if execution_id is not None:
                existing = get_execution_by_id(self.session, execution_id)
                if not existing:
                    raise ValueError(f"Execution {execution_id} not found")
                update_execution(
                    self.session,
                    execution_id,
                    status='running',
                    started_at=datetime.now(),
                )
                self.root_execution = existing
            else:
                self.root_execution = create_execution(
                    self.session,
                    user_id=self.user_id,
                    flow_id=self.flow_id,
                    execution_type='flow',
                    name=self.flow.name,
                    input_data=initial_input,
                    status='running',
                    started_at=datetime.now()
                )

            #Prepare tools
            self._prepare_tools()

            #Execute from entry point
            entry_point = self.graph_config["entry_point"]
            final_output = self._execute_from_node(entry_point, initial_input)

            # Mark execution as completed
            update_execution(self.session, self.root_execution.id,
                status='completed',
                output_data=final_output,
                completed_at=datetime.now()
            )

            return {
                "flow_id": self.flow_id,
                "execution_id": self.root_execution.id,
                "status": "completed",
                "final_output": final_output,
                "execution_trace": self.execution_trace
            }

        except Exception as e:
            logger.error(f"Flow execution failed: {e}")
            if self.root_execution:
                update_execution(self.session, self.root_execution.id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.now()
                )
            return {
                "flow_id": self.flow_id,
                "execution_id": self.root_execution.id if self.root_execution else None,
                "status": "failed",
                "final_output": None,
                "execution_trace": self.execution_trace,
                "error": str(e)
            }




    def resume_flow(self, flow_id: int, execution_trace: List[Dict], resume_input: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
        """
        Resume flow from the last successful node trace
        Optionally provide a new input to the failed node

        Args:
            execution_trace: Execution trace from run
            resume_input: [Optional] Input to override for the failed node
        Returns:
            FlowExecutionResult dict (same as execute_flow)
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

            # Create a new top-level execution for the resume
            self.root_execution = create_execution(
                self.session,
                user_id=self.user_id,
                flow_id=self.flow_id,
                execution_type='flow',
                name=f"{self.flow.name} (resumed)",
                input_data=input_data,
                status='running',
                started_at=datetime.now(),
                execution_metadata={"resumed_from_node": failed_node}
            )

            #Reload tools from the failed node onwards
            self._prepare_tools(start_node=failed_node)

            final_output = self._execute_from_node(failed_node, input_data)

            update_execution(self.session, self.root_execution.id,
                status='completed',
                output_data=final_output,
                completed_at=datetime.now()
            )

            return {
                "flow_id": self.flow_id,
                "execution_id": self.root_execution.id,
                "status": "completed",
                "final_output": final_output,
                "execution_trace": self.execution_trace
            }

        except Exception as e:
            logger.error(f"Flow resume failed: {e}")
            if self.root_execution:
                update_execution(self.session, self.root_execution.id,
                    status='failed',
                    error_message=str(e),
                    completed_at=datetime.now()
                )
            return {
                "flow_id": self.flow_id,
                "execution_id": self.root_execution.id if self.root_execution else None,
                "status": "failed",
                "final_output": None,
                "execution_trace": self.execution_trace,
                "error": str(e)
            }
