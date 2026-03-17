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
  system_prompt: string;
  user_prompt: string;
  llm_provider: string;
  tool_ids: number[];
}

/**
 * Edge configuration in the agent graph
 */
export interface AgentEdge {
  from_node: string;
  to_node: string;
  is_loop: boolean;  // True if this is a backward connection (loop)
}

/**
 * Full graph configuration for composed agents
 */
export interface AgentGraphConfig {
  nodes: Record<string, SubAgentConfig>;
  edges: AgentEdge[];
  entry_point: string;
  exit_points: string[];
}

/**
 * Convert visual agent builder representation to graph_config
 */
export function buildAgentGraph(
  nodes: XYFlowNode[],
  edges: XYFlowEdge[]
): AgentGraphConfig {
  // Build nodes config from agent builder nodes
  const nodesConfig: Record<string, SubAgentConfig> = {};

  nodes.forEach(node => {
    if (node.type === 'agentBuilderNode') {
      const subAgent: SubAgentConfig = {
        agent_type: node.data.agentType || 'react',
        name: node.data.name || 'Agent',
        system_prompt: node.data.systemPrompt || '',
        user_prompt: node.data.userPrompt || '',
        llm_provider: node.data.llmProvider || '',
        tool_ids: node.data.assignedTools || []
      };

      nodesConfig[node.id] = subAgent;
    }
  });

  // Build edges, detecting loops based on node positions
  const edgesConfig: AgentEdge[] = edges.map(edge => {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);

    // Detect loop: if target is positioned to the left of source, it's a loop
    const isLoop = !!(sourceNode && targetNode &&
      targetNode.position.x < sourceNode.position.x);

    return {
      from_node: edge.source,
      to_node: edge.target,
      is_loop: isLoop
    };
  });

  // Find entry point (node with no incoming edges, excluding loops)
  const forwardEdges = edgesConfig.filter(e => !e.is_loop);
  const nodesWithIncoming = new Set(forwardEdges.map(e => e.to_node));
  const agentNodes = nodes.filter(n => n.type === 'agentBuilderNode');
  const entryNodes = agentNodes.filter(n => !nodesWithIncoming.has(n.id));

  if (entryNodes.length === 0) {
    throw new Error('No entry point found - all agent nodes have incoming edges');
  }
  if (entryNodes.length > 1) {
    throw new Error(`Multiple entry points found: ${entryNodes.map(n => n.data.name || n.id).join(', ')}. The composed agent must have exactly one starting node.`);
  }

  const entryPoint = entryNodes[0].id;

  // Find exit points (nodes with no outgoing forward edges)
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

  // Check we have at least one node
  const agentNodes = nodes.filter(n => n.type === 'agentBuilderNode');
  if (agentNodes.length === 0) {
    errors.push('No agent nodes found. Add at least one agent to the canvas.');
  }

  // Check all nodes have required data
  for (const node of agentNodes) {
    if (!node.data.agentType) {
      errors.push(`Node "${node.data.name || node.id}" has no agent type`);
    }
    if (!node.data.systemPrompt) {
      errors.push(`Node "${node.data.name || node.id}" has no system prompt`);
    }
    if (!node.data.llmProvider) {
      errors.push(`Node "${node.data.name || node.id}" has no LLM provider assigned`);
    }
  }

  // Try building the graph to catch entry/exit point issues
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
