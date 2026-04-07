/**
 * Agent Graph Builder
 * Converts visual agent builder nodes/edges to graph_config for composed agents
 * Similar pattern to flowBuilder.ts but tailored for agent composition
 */

import type { Node as XYFlowNode, Edge as XYFlowEdge } from '@xyflow/svelte';
import type { AgentTypeKey } from './agentTemplates';

/**
 * Configuration for a single sub-agent in the composed agent
 */
export interface SubAgentConfig {
  agent_type: AgentTypeKey;
  name: string;
  description: string;
  system_prompt: string;
  user_prompt: string;
  llm_provider: string;
  tool_ids: number[];
  output_paths?: Record<string, string | { description: string; return_behavior: string }>;  // path_name → config for conditional routing
}

/**
 * Edge configuration in the agent graph
 */
export interface AgentEdge {
  from_node: string;
  to_node: string;
  is_loop: boolean;
  output_path?: string;  // Which output path this edge corresponds to
}

/**
 * Full graph configuration for composed agents
 */
export interface AgentGraphConfig {
  nodes: Record<string, SubAgentConfig>;
  edges: AgentEdge[];
  entry_point: string;
  exit_points: string[];
  max_loop_iterations?: number;
}

/**
 * Classify which edges are back edges (loops) using DFS from a known start node.
 * A back edge points to a node currently on the DFS recursion stack (an ancestor).
 * Returns a Set of edge indices that are back edges.
 */
function classifyBackEdges(
  startNodeId: string,
  nodeIds: string[],
  edges: { source: string; target: string }[]
): Set<number> {
  const adj = new Map<string, { target: string; edgeIndex: number }[]>();
  for (const id of nodeIds) {
    adj.set(id, []);
  }
  edges.forEach((edge, idx) => {
    const list = adj.get(edge.source);
    if (list) {
      list.push({ target: edge.target, edgeIndex: idx });
    }
  });

  const visited = new Set<string>();
  const onStack = new Set<string>();
  const backEdges = new Set<number>();

  function dfs(node: string) {
    visited.add(node);
    onStack.add(node);

    for (const { target, edgeIndex } of adj.get(node) || []) {
      if (onStack.has(target)) {
        backEdges.add(edgeIndex);
      } else if (!visited.has(target)) {
        dfs(target);
      }
    }

    onStack.delete(node);
  }

  // Start DFS from the known entry point
  dfs(startNodeId);

  // Visit any disconnected nodes (shouldn't happen in a valid graph, but safe)
  for (const id of nodeIds) {
    if (!visited.has(id)) {
      dfs(id);
    }
  }

  return backEdges;
}

/**
 * Convert visual agent builder representation to graph_config.
 * Uses the Start node to determine the entry point, and DFS to classify loop edges.
 */
export function buildAgentGraph(
  nodes: XYFlowNode[],
  edges: XYFlowEdge[]
): AgentGraphConfig {
  // Find and validate the Start node
  const startNode = nodes.find(n => n.type === 'startNode');
  if (!startNode) {
    throw new Error('No Start node found. The Start node determines which agent runs first.');
  }

  const startEdge = edges.find(e => e.source === startNode.id);
  if (!startEdge) {
    throw new Error('The Start node is not connected. Draw an edge from Start to the first agent.');
  }

  const entryPoint = startEdge.target;

  // Filter to agent nodes only (exclude Start node)
  const agentNodes = nodes.filter(n => n.type !== 'startNode');
  const agentEdges = edges.filter(e => e.source !== startNode.id);

  if (agentNodes.length === 0) {
    throw new Error('No agent nodes found. Add at least one agent to the canvas.');
  }

  // Verify the Start node points to an actual agent node
  if (!agentNodes.some(n => n.id === entryPoint)) {
    throw new Error('The Start node must connect to an agent node.');
  }

  // Build nodes config from agent nodes
  const nodesConfig: Record<string, SubAgentConfig> = {};

  agentNodes.forEach(node => {
    const subAgent: SubAgentConfig = {
      agent_type: node.data.agentType || 'react',
      name: node.data.name || 'Agent',
      description: node.data.description || '',
      system_prompt: node.data.system_prompt || '',
      user_prompt: node.data.user_prompt || '',
      llm_provider: node.data.llm_provider || '',
      tool_ids: node.data.tool_ids || []
    };

    if (node.data.output_paths && Object.keys(node.data.output_paths).length > 0) {
      subAgent.output_paths = node.data.output_paths;
    }

    nodesConfig[node.id] = subAgent;
  });

  // Classify back edges via DFS from the entry point
  const agentNodeIds = agentNodes.map(n => n.id);
  const backEdgeIndices = classifyBackEdges(
    entryPoint,
    agentNodeIds,
    agentEdges.map(e => ({ source: e.source, target: e.target }))
  );

  // Build edge configs with DFS-computed is_loop
  const edgesConfig: AgentEdge[] = agentEdges.map((edge, idx) => {
    const agentEdge: AgentEdge = {
      from_node: edge.source,
      to_node: edge.target,
      is_loop: backEdgeIndices.has(idx)
    };

    if (edge.sourceHandle) {
      agentEdge.output_path = edge.sourceHandle;
    }

    return agentEdge;
  });

  // Find exit points (agent nodes with no outgoing forward edges)
  const forwardEdges = edgesConfig.filter(e => !e.is_loop);
  const nodesWithOutgoing = new Set(forwardEdges.map(e => e.from_node));
  const exitPoints = agentNodes
    .filter(n => !nodesWithOutgoing.has(n.id))
    .map(n => n.id);

  if (exitPoints.length === 0) {
    throw new Error('No exit points found in agent graph');
  }

  return {
    nodes: nodesConfig,
    edges: edgesConfig,
    entry_point: entryPoint,
    exit_points: exitPoints
  };
}

/**
 * Build the top-level graph_config for an agent.
 * Returns AgentGraphConfig directly (no wrapper).
 */
export function buildAgentGraphConfig(
  nodes: XYFlowNode[],
  edges: XYFlowEdge[]
): AgentGraphConfig {
  return buildAgentGraph(nodes, edges);
}

/**
 * Validate that the agent graph is well-formed
 */
export function validateAgentGraph(
  nodes: XYFlowNode[],
  edges: XYFlowEdge[]
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Exclude Start node from agent validation
  const agentNodes = nodes.filter(n => n.type !== 'startNode');

  if (agentNodes.length === 0) {
    errors.push('No agent nodes found. Add at least one agent to the canvas.');
  }

  // Validate agent nodes have required data
  for (const node of agentNodes) {
    if (!node.data.agentType) {
      errors.push(`Node "${node.data.name || node.id}" has no agent type`);
    }
    if (!node.data.system_prompt) {
      errors.push(`Node "${node.data.name || node.id}" has no system prompt`);
    }
    if (!node.data.llm_provider) {
      errors.push(`Node "${node.data.name || node.id}" has no LLM provider assigned`);
    }
  }

  // Try building the graph to catch Start node and topology issues
  if (agentNodes.length > 0) {
    try {
      buildAgentGraph(nodes, edges);
    } catch (e) {
      errors.push((e as Error).message);
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}
