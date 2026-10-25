from pydantic import BaseModel, Field
from typing import Dict, List, Tuple 

## Data Structures ##

class Chain(BaseModel):
    chain_id: str
    #Nodes within chain
    nodes: List[str]
    #Other chains that are an input to chain
    inputs: List[str] = Field(default_factory=list)

    @property
    def head(self) -> str:
        return self.nodes[0]

    @property
    def tail(self) -> str:
        return self.nodes[-1]
    
## Decompose into linear chains

def decompose_into_chains(graph_config: dict) -> Tuple[List[Chain], Dict[str, str]]:
    """
    Split Directional Acyclic Graph into linear chains
    Chains don't have splitting - each node has 1 input and 1 output

    Returns:
        chains: list of Chain objects
        chain_by_node: map of node_id -> chain_id
    """
    nodes = graph_config["nodes"]
    edges = [e for e in (graph_config.get("edges", []) or [])]

    in_edges: Dict[str, List[dict]] = {n: [] for n in nodes}
    out_edges: Dict[str, List[dict]] = {n: [] for n in nodes}
    for e in edges:
        in_edges[e["to_node"]].append(e)
        out_edges[e["from_node"]].append(e)

    def is_chain_head(node_id: str) -> bool:
        """
        Detect if the node is the head of a chain
        True if:
            It has more than one predecessor (input from multiple nodes)
            OR
            Its predecessor has more than one output
        """
        preds = in_edges[node_id]
        #Check if it has more than one predecessor
        if len(preds) != 1:
            return True
        #Check if its a fork - its predecessor has more than one output path
        return len(out_edges[preds[0]["from_node"]]) > 1
    
    # Walk forward from each chain head, collecting linear successors
    chains: List[Chain] = []
    chain_by_node: Dict[str, str] = {}

    ## Loop through all head nodes, create chains with non-head-nodes
    ## Sort head nodes so final creation is deterministic
    for head in sorted(n for n in nodes if is_chain_head(n)):
        chain_nodes = [head]
        current = head
        while len(out_edges[current]) == 1:
            next = out_edges[current][0]["to_node"]
            if is_chain_head(next):
                break
            chain_nodes.append(next)
            current = next

        # Make the chain object
        chain = Chain(chain_id=f"chain_{len(chains)}", nodes=chain_nodes)
        chains.append(chain)
        for n in chain_nodes:
            chain_by_node[n] = chain.chain_id

    # Chain inputs (what chains feed int a chain)
    for chain in chains:
        for edge in in_edges[chain.head]:
            source_chain = chain_by_node[edge["from_node"]]
            if source_chain not in chain.inputs:
                chain.inputs.append(source_chain)

    return chains, chain_by_node
    

## Group chains into topological levals

def compute_chain_levels(chains: List[Chain]) -> List[List[Chain]]:
    """
    Use Kahn's algorithm to separate chains by level

    Chains at the same level can run in parallel processes
    """

    by_id = {c.chain_id: c for c in chains}
    in_degree = {c.chain_id: len(c.inputs) for c in chains}
    successors: Dict[str, List[str]] = {c.chain_id: [] for c in chains}
    for c in chains:
        for input_id in c.inputs:
            successors[input_id].append(c.chain_id)
    
    remaining = set(by_id.keys())
    levels: List[List[Chain]] = []
    while remaining:
        level_ids = sorted(cid for cid in remaining if in_degree[cid] == 0)
        if not level_ids:
            raise ValueError("Cycle detected at the chain level (graph is not a DAG).")
        levels.append([by_id[cid] for cid in level_ids])

        ## Remove chains that were added to levels
        for cid in level_ids:
            remaining.discard(cid)
            for s in successors[cid]:
                in_degree[s] -= 1
    return levels

## Create Python source file

_PARALLEL_HEADER = '''"""Generated parallel orchestrator"""
import asyncio
from src.runners.flow_runner import _run_stage, _resolve_node_input
'''

_SEQUENTIAL_HEADER = '''"""Generated sequential orchestrator"""
from src.runners.flow_runner import _run_stage, _resolve_node_input
'''

def emit_chain_function(chain: Chain) -> str:
    """
    Generate one async chain function

    Body:
        - First node: 
            Resolve its input from the function args with '_resolve_node_input(session, node_id, {chain_id: value, ...})`
        - Subsequent nodes:
            Take previous node's output directly
        - Return tail node's output
    """
    nodes = chain.nodes
    head = chain.head

    if not chain.inputs:
        signature = f"async def {chain.chain_id}(orchestrator, initial_input):"
        first_input_expr = "initial_input"
        docstring = f'    """Root chain. Nodes: {" -> ".join(nodes)}"""'
    else:
        params = ", ".join(f"input_from_{cid}" for cid in chain.inputs)
        signature = f"async def {chain.chain_id}(orchestrator, {params}):"
        if len(chain.inputs) == 1:
            # Single input, pass straight through the resolver
            cid = chain.inputs[0]
            first_input_expr = (
                f"_resolve_node_input(orchestrator, {repr(head)}, "
                f"{{{repr(cid)}: input_from_{cid}}})"
            )
            docstring = (
                f'    """Nodes: {" -> ".join(nodes)}\n'
                f'       Input from: {chain.inputs[0]}"""'
            )
        else:
            # Resolver combines multiple upstreams per edge mappings

            # Build mapping dict for script
            mapping_dict = "{" + ", ".join(
                f"{repr(cid)}: input_from_{cid}" for cid in chain.inputs
            ) + "}"
            first_input_expr = f"_resolve_node_input(orchestrator, {repr(head)}, {mapping_dict})"
            docstring = (
                f'    """Nodes: {" -> ".join(nodes)}\n'
                f'       Inputs from: {", ".join(chain.inputs)}"""'
            )

    lines = [signature, docstring]
    lines.append(f"    {_var(head)} = await _run_stage(orchestrator, {repr(head)}, {first_input_expr})")

    for prev, curr in zip(nodes, nodes[1:]):
        lines.append(f"    {_var(curr)} = await _run_stage(orchestrator, {repr(curr)}, {_var(prev)})")

    lines.append(f"    return {_var(chain.tail)}")
    return "\n".join(lines)


def emit_orchestrator_function(
    chains: List[Chain],
    levels: List[List[Chain]],
    parallel: bool = True,
) -> str:
    """ Generate top-level orchestration function
    
    Parallel Mode:
        - Single-chain levels: one `await chain_X(...)` function
        - Multi-chain levels: `await asyncio.gather(...)` over chains

    Non-parallel Mode:
        - Single-chain levels: one `await chain(...)` statement
        - Muti-chain levels: one `await chain_X(...)` statement per chain
    """
    lines = ["async def orchestrate(orchestrator, initial_input):"]
    lines.append(f'    """Top-level orchestrator, generated from graph config"""')

    for level in levels:
        if len(level) == 1 or not parallel:
            # Run serially - one await per chain
            for c in level:
                args = _format_chain_call_args(c)
                lines.append(f"    {c.chain_id}_out = await {c.chain_id}(orchestrator, {args})")
        else:
            # Mullti-chain in parallel mode
            outputs = ", ".join(f"{c.chain_id}_out" for c in level)
            calls = []
            for c in level:
                args = _format_chain_call_args(c)
                calls.append(f"        {c.chain_id}(orchestrator, {args})")
            lines.append(f"    {outputs} = await asyncio.gather(")
            lines.append(",\n".join(calls) + ",")
            lines.append("    )")

    # Return exit chain's output
    exit_chains = [c for c in chains if not any(c.chain_id in other.inputs for other in chains)]

    if len(exit_chains) == 1:
        lines.append(f"    return {exit_chains[0].chain_id}_out")
    else:
        ret = ", ".join(f"{c.chain_id}_out" for c in exit_chains)
        lines.append(f"    return ({ret})")
    return "\n".join(lines)

def _format_chain_call_args(chain: Chain) -> str:
    """Format positional arguments for a chain call"""
    if not chain.inputs:
        return "initial_input"
    return ", ".join(f"{cid}_out" for cid in chain.inputs) 

def _var(node_id: str) -> str:
    """Translate node_id into a valid Python variable name

    'node.1' -> node_1
    """
    safe = "".join(c if c.isalnum() else "_" for c in node_id)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return f"{safe}_out"


## Top-level function for generating entire graph code

def generate_orchestrator_code(graph_config: dict, parallel: bool = True) -> str:
    """Create python file that can run a flow

    Args:
    graph_config: 
        Flow's adjacency matrix
    parallel: 
        If True, use async.gather to run chains at the same level concurrrently
        Otherwise, run chains sequentially
    """

    chains, _ = decompose_into_chains(graph_config)
    levels = compute_chain_levels(chains)

    header = _PARALLEL_HEADER if parallel else _SEQUENTIAL_HEADER
    parts = [header.rstrip()]
    for chain in chains:
        parts.append(emit_chain_function(chain))
    parts.append(emit_orchestrator_function(chains, levels, parallel=parallel))
    return "\n\n\n".join(parts) + "\n"



