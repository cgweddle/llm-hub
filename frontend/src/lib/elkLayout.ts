/**
 * ELK Auto-Layout for SvelteFlow graphs
 * Uses elkjs to automatically position nodes with proper port/handle routing
 */

import ELK, { type ElkNode, type ElkExtendedEdge } from 'elkjs/lib/elk.bundled.js';
import type { Node, Edge } from '@xyflow/svelte';

const elk = new ELK();

// ELK layout options
const elkOptions = {
  'elk.algorithm': 'layered',
  'elk.layered.spacing.nodeNodeBetweenLayers': '200',  // Increased horizontal spacing between layers
  'elk.spacing.nodeNode': '150',                        // Increased spacing between nodes in same layer
  'elk.direction': 'RIGHT',
  'elk.padding': '[top=50,left=50,bottom=50,right=50]', // Add padding around the graph
  'elk.edgeRouting': 'ORTHOGONAL',                      // Better edge routing
};

export interface LayoutOptions {
  nodeWidth?: number;
  nodeHeight?: number;
}

/**
 * Auto-layout nodes using ELK with port information
 */
export async function autoLayoutNodes(
  nodes: Node[],
  edges: Edge[],
  options: LayoutOptions = {}
): Promise<Node[]> {
  const { nodeWidth = 300, nodeHeight = 200 } = options;  // Increased default dimensions

  // Build ELK graph
  const graph: ElkNode = {
    id: 'root',
    layoutOptions: elkOptions,
    children: nodes.map((node) => {
      // Extract ports from node data
      const ports = buildPortsForNode(node);

      return {
        id: node.id,
        width: nodeWidth,
        height: nodeHeight,
        ports: ports
      };
    }),
    edges: edges.map((edge) => {
      const elkEdge: ElkExtendedEdge = {
        id: edge.id,
        sources: [edge.sourceHandle ? `${edge.source}_${edge.sourceHandle}` : edge.source],
        targets: [edge.targetHandle ? `${edge.target}_${edge.targetHandle}` : edge.target]
      };

      return elkEdge;
    })
  };

  // Run layout
  const layoutedGraph = await elk.layout(graph);

  // Apply positions back to nodes
  const layoutedNodes = nodes.map((node) => {
    const elkNode = layoutedGraph.children?.find((n) => n.id === node.id);

    if (elkNode?.x !== undefined && elkNode?.y !== undefined) {
      return {
        ...node,
        position: {
          x: elkNode.x,
          y: elkNode.y
        }
      };
    }

    return node;
  });

  return layoutedNodes;
}

/**
 * Build ELK ports from node's input/output schemas
 */
function buildPortsForNode(node: Node) {
  const ports = [];

  // Input ports (left side)
  if (node.data.input_schema) {
    const inputParams = extractInputParams(node.data.input_schema);
    inputParams.forEach((paramName, index) => {
      ports.push({
        id: `${node.id}_input-${paramName}`,
        properties: {
          side: 'WEST',
          index: index
        }
      });
    });
  }

  // Output ports (right side)
  if (node.data.output_schema) {
    const outputProps = extractOutputProps(node.data.output_schema);
    outputProps.forEach((propName, index) => {
      ports.push({
        id: `${node.id}_output-${propName}`,
        properties: {
          side: 'EAST',
          index: index
        }
      });
    });
  } else {
    // Default single output port
    ports.push({
      id: `${node.id}_output`,
      properties: {
        side: 'EAST',
        index: 0
      }
    });
  }

  return ports;
}

/**
 * Extract input parameter names from schema
 */
function extractInputParams(inputSchema: any): string[] {
  if (!inputSchema || typeof inputSchema !== 'object') {
    return [];
  }

  // Check for properties field (JSON Schema style)
  if (inputSchema.properties) {
    return Object.keys(inputSchema.properties);
  }

  // Fallback: use all keys
  return Object.keys(inputSchema);
}

/**
 * Extract output property names from schema
 */
function extractOutputProps(outputSchema: any): string[] {
  if (!outputSchema || typeof outputSchema !== 'object') {
    return [];
  }

  // Check if it's a dictionary output
  if (outputSchema.properties) {
    return Object.keys(outputSchema.properties);
  }

  return [];
}
