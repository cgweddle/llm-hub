/**
 * Flow Builder
 * Converts @xyflow visual representation to graph_config
 */

import type { Node as XYFlowNode, Edge as XYFlowEdge } from '@xyflow/svelte';
import type { GraphConfig, NodeConfig, EdgeMapping } from './api';
import type { Tool } from './api';

/**
 * Convert @xyflow visual nodes/edges to graph_config
 */
export function buildEnhancedGraphConfig(
  nodes: XYFlowNode[],
  edges: XYFlowEdge[],
  tools: Tool[]
): GraphConfig {
  // Build nodes config from tool nodes
  const nodesConfig: Record<string, NodeConfig> = {};

  nodes.forEach(node => {
    if (node.type === 'triggerNode') {
      // Trigger node — entry point with user-provided text
      nodesConfig[node.id] = {
        node_type: 'trigger',
        id: 0,
        name: node.data.name || 'Text Input',
        input_value: node.data.triggerValue || ''
      };
    } else if (node.type === 'toolNode' && node.data.isAgent && node.data.agentId) {
      nodesConfig[node.id] = {
        node_type: 'agent',
        id: node.data.agentId,
        name: node.data.name
      };
    } else if (node.type === 'toolNode' && node.data.toolId) {
      const nodeConfig: NodeConfig = {
        node_type: 'tool',
        id: node.data.toolId,
        name: node.data.name
      };

      if (node.data.parameterValues && Object.keys(node.data.parameterValues).length > 0) {
        nodeConfig.input_values = node.data.parameterValues;
      }

      nodesConfig[node.id] = nodeConfig;
    }
  });

  // Build edges with mapping
  const edgesConfig: EdgeMapping[] = edges.map(edge => {
    const sourceNode = nodes.find(n => n.id === edge.source);
    const targetNode = nodes.find(n => n.id === edge.target);

    // Determine mapping from edge handles or tool schemas
    const mapping = determineMappingFromEdge(edge, sourceNode, targetNode, tools);

    return {
      from_node: edge.source,
      to_node: edge.target,
      mapping: mapping  // undefined or Record<string, string>
    };
  });

  // Find entry point (node with no incoming edges)
  // Include tool, agent, and trigger nodes
  const nodesWithIncoming = new Set(edges.map(e => e.target));
  const allFlowNodes = nodes.filter(n =>
    (n.type === 'toolNode' && (n.data.toolId || n.data.agentId)) ||
    n.type === 'triggerNode'
  );
  const entryNodes = allFlowNodes.filter(n => !nodesWithIncoming.has(n.id));

  if (entryNodes.length === 0) {
    throw new Error('No entry point found in flow - all nodes have incoming edges');
  }
  if (entryNodes.length > 1) {
    throw new Error(`Multiple entry points found in flow: ${entryNodes.map(n => n.data.name).join(', ')}. Flow must have exactly one starting node.`);
  }

  const entryPoint = entryNodes[0].id;

  // Find exit points (nodes with no outgoing edges, excluding triggers)
  const nodesWithOutgoing = new Set(edges.map(e => e.source));
  const exitPoints = allFlowNodes
    .filter(n => n.type !== 'triggerNode')
    .filter(n => !nodesWithOutgoing.has(n.id))
    .map(n => n.id);

  if (exitPoints.length === 0) {
    throw new Error('No exit points found in flow');
  }

  return {
    nodes: nodesConfig,
    edges: edgesConfig,
    entry_point: entryPoint,
    exit_points: exitPoints
  };
}

/**
 * Determine output->input mapping from edge handles
 * Returns undefined for passthrough, or a mapping object
 */
function determineMappingFromEdge(
  edge: XYFlowEdge,
  sourceNode: XYFlowNode | undefined,
  targetNode: XYFlowNode | undefined,
  tools: Tool[]
): Record<string, string> | undefined {
  // If edge has explicit handle info, use it. xyflow reports a handle with
  // id="" as either "" or null depending on the code path — treat both as
  // "the generic whole-output/whole-input handle".
  if (edge.sourceHandle != null || edge.targetHandle != null) {
    const sourceHandle = edge.sourceHandle ?? "";
    const targetHandle = edge.targetHandle ?? "";

    // Empty targetHandle = generic whole-input target (agent nodes, tools
    // without an input schema). A named sourceHandle still selects one
    // field from an expanded dict output: store {field: ""} so the backend
    // feeds that field (not the whole dict) to the node.
    if (targetHandle === "") {
      if (sourceHandle !== "") {
        return { [sourceHandle]: "" };
      }
      return undefined; // whole output → whole input: plain passthrough
    }

    // Whole output → Specific parameter (e.g., "" → "inputs")
    // Use empty string as key to signal "whole output"
    if (sourceHandle === "") {
      return { "": targetHandle };
    }

    // Specific field → Specific parameter (e.g., "result" → "input_text")
    return { [sourceHandle]: targetHandle };
  }

  // Fallback: auto-detect mapping by name matching
  return autoDetectMapping(sourceNode, targetNode);
}

/**
 * Auto-detect mapping by matching field names (fallback)
 */
function autoDetectMapping(
  sourceNode: XYFlowNode | undefined,
  targetNode: XYFlowNode | undefined
): Record<string, string> | undefined {
  // If either node doesn't have schemas, passthrough
  if (!sourceNode?.data.output_schema || !targetNode?.data.input_schema) {
    return undefined;
  }

  const outputSchema = sourceNode.data.output_schema;
  const inputSchema = targetNode.data.input_schema;

  // Build mapping by matching field names
  const mapping: Record<string, string> = {};

  // If output is an object with properties
  if (outputSchema.type === "object" && outputSchema.properties) {
    Object.keys(outputSchema.properties).forEach(outputField => {
      // Try to find matching input param
      Object.keys(inputSchema).forEach(inputParam => {
        // Match by exact name or similar name
        if (inputParam === outputField || inputParam.includes(outputField)) {
          mapping[outputField] = inputParam;
        }
      });
    });
  }

  // If no mappings found, use passthrough (undefined)
  return Object.keys(mapping).length > 0 ? mapping : undefined;
}
