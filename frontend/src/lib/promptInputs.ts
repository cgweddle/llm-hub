/**
 * Named-input helpers mirroring src/utils/prompt_template.py and
 * src/validate/agent_graph_inputs.py.
 * Ported line-by-line so frontend and backend agree on input names, edge
 * target resolution, and graph validation.
 */

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

export interface PromptInputGraphEdge {
  from_node: string;
  to_node: string;
  is_loop?: boolean;
  output_path?: string;
  target_input?: string;
  mapping?: Record<string, string>;
}

export interface PromptInputGraphNode {
  user_prompt?: string;
  output_paths?: Record<string, any>;
}

export interface PromptInputGraphConfig {
  nodes?: Record<string, PromptInputGraphNode>;
  entry_point?: string;
  edges?: PromptInputGraphEdge[];
}

/**
 * List a user prompt's named inputs: ordered, de-duplicated, skipping
 * {{escaped}} braces and the reserved {message_history} placeholder.
 *
 * An empty (or whitespace-only) prompt implicitly means "{input}",
 * so it returns ["input"]. A non-empty prompt with zero
 * {name} placeholders returns [].
 */
export function extractPromptInputs(userPrompt: string | undefined | null): string[] {
  if (!userPrompt || !userPrompt.trim()) {
    return ['input'];
  }

  const names: string[] = [];
  const seen = new Set<string>();
  let i = 0;
  const n = userPrompt.length;
  while (i < n) {
    if (userPrompt[i] === '{') {
      if (i + 1 < n && userPrompt[i + 1] === '{') {
        i += 2;
        continue;
      }
      const close = userPrompt.indexOf('}', i + 1);
      if (close === -1) {
        i += 1;
        continue;
      }
      const name = userPrompt.slice(i + 1, close);
      if (name !== 'message_history' && NAME_RE.test(name) && !seen.has(name)) {
        seen.add(name);
        names.push(name);
      }
      i = close + 1;
    } else {
      i += 1;
    }
  }
  return names;
}

/**
 * The agent's flow-level inputs: the entry sub-agent's inputs
 * that no internal edge feeds. These are the handles shown on the agent
 * node on the flow canvas.
 */
export function agentExternalInputs(graphConfig: PromptInputGraphConfig | undefined | null): string[] {
  const nodesConfig = graphConfig?.nodes || {};
  const entryPoint = graphConfig?.entry_point;
  const entryConfig = (entryPoint && nodesConfig[entryPoint]) || {};
  const entryInputs = extractPromptInputs(entryConfig.user_prompt || '');

  const fed = new Set(
    (graphConfig?.edges || [])
      .filter(edge => edge.to_node === entryPoint)
      .map(edge => edge.target_input || 'input')
  );

  return entryInputs.filter(name => !fed.has(name));
}

/**
 * Flow-edge mapping shape -> the agent input it targets. Reuses the
 * tool-edge `mapping` shape ({out_field_or_"": input_name}); missing
 * mapping, or a mapping whose value is "", targets "input".
 */
export function agentTargetInput(edge: { mapping?: Record<string, string> } | undefined | null): string {
  const mapping = edge?.mapping;
  if (!mapping || Object.keys(mapping).length === 0) {
    return 'input';
  }
  const value = Object.values(mapping)[0];
  return value || 'input';
}

function nodeInputsMap(nodesConfig: Record<string, PromptInputGraphNode>): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  for (const [nodeId, cfg] of Object.entries(nodesConfig)) {
    result[nodeId] = extractPromptInputs(cfg.user_prompt || '');
  }
  return result;
}

function buildForwardIn(
  nodesConfig: Record<string, PromptInputGraphNode>,
  edges: PromptInputGraphEdge[]
): Record<string, PromptInputGraphEdge[]> {
  const forwardIn: Record<string, PromptInputGraphEdge[]> = {};
  for (const n of Object.keys(nodesConfig)) forwardIn[n] = [];
  for (const edge of edges) {
    if (edge.to_node in forwardIn && !edge.is_loop) {
      forwardIn[edge.to_node].push(edge);
    }
  }
  return forwardIn;
}

/**
 * component_id: node -> component id, component_size: id -> size.
 * Computed over ALL edges (forward + loop) — mirrors
 * agent_graph_engine.strongly_connected, used here only for the
 * exclusivity check's "inside a loop" test (component size > 1).
 */
function stronglyConnected(
  nodesConfig: Record<string, PromptInputGraphNode>,
  edges: PromptInputGraphEdge[]
): { componentId: Record<string, number>; componentSize: Record<number, number> } {
  const adjacency: Record<string, string[]> = {};
  for (const n of Object.keys(nodesConfig)) adjacency[n] = [];
  for (const edge of edges) {
    if (edge.from_node in adjacency && edge.to_node in adjacency) {
      adjacency[edge.from_node].push(edge.to_node);
    }
  }

  function reachable(start: string): Set<string> {
    const seen = new Set<string>([start]);
    const stack = [start];
    while (stack.length) {
      const n = stack.pop()!;
      for (const nxt of adjacency[n] || []) {
        if (!seen.has(nxt)) {
          seen.add(nxt);
          stack.push(nxt);
        }
      }
    }
    return seen;
  }

  const reach: Record<string, Set<string>> = {};
  for (const n of Object.keys(nodesConfig)) reach[n] = reachable(n);

  const componentId: Record<string, number> = {};
  let nextId = 0;
  for (const n of Object.keys(nodesConfig)) {
    if (n in componentId) continue;
    const component = [...reach[n]].filter(m => reach[m].has(n));
    for (const m of component) componentId[m] = nextId;
    nextId += 1;
  }

  const componentSize: Record<number, number> = {};
  for (const cid of Object.values(componentId)) {
    componentSize[cid] = (componentSize[cid] || 0) + 1;
  }
  return { componentId, componentSize };
}

function computeLoopNodes(
  nodesConfig: Record<string, PromptInputGraphNode>,
  edges: PromptInputGraphEdge[]
): Set<string> {
  const { componentId, componentSize } = stronglyConnected(nodesConfig, edges);
  const loopNodes = new Set<string>();
  for (const [n, cid] of Object.entries(componentId)) {
    if (componentSize[cid] > 1) loopNodes.add(n);
  }
  return loopNodes;
}

type EdgeReqFn = (edge: PromptInputGraphEdge) => Record<string, string>;

function computeEdgeReq(
  nodesConfig: Record<string, PromptInputGraphNode>,
  edges: PromptInputGraphEdge[]
): EdgeReqFn {
  const forwardIn = buildForwardIn(nodesConfig, edges);
  const nodeReqCache = new Map<string, Record<string, string>>();
  const edgeReqCache = new Map<PromptInputGraphEdge, Record<string, string>>();

  function nodeReq(nodeId: string | undefined): Record<string, string> {
    if (!nodeId) return {};
    if (nodeReqCache.has(nodeId)) return nodeReqCache.get(nodeId)!;
    const incoming = forwardIn[nodeId] || [];
    let result: Record<string, string>;
    if (incoming.length === 0) {
      result = {};
    } else {
      const reqs = incoming.map(e => edgeReq(e));
      result = { ...reqs[0] };
      for (const r of reqs.slice(1)) {
        for (const k of Object.keys(result)) {
          if (!(k in r) || r[k] !== result[k]) delete result[k];
        }
      }
    }
    nodeReqCache.set(nodeId, result);
    return result;
  }

  function edgeReq(edge: PromptInputGraphEdge): Record<string, string> {
    if (edgeReqCache.has(edge)) return edgeReqCache.get(edge)!;
    const base = { ...nodeReq(edge.from_node) };
    if (edge.output_path) {
      base[edge.from_node] = edge.output_path;
    }
    edgeReqCache.set(edge, base);
    return base;
  }

  return edgeReq;
}

function edgesExclusive(
  e1: PromptInputGraphEdge,
  e2: PromptInputGraphEdge,
  edgeReq: EdgeReqFn,
  loopNodes: Set<string>
): boolean {
  const r1 = edgeReq(e1);
  const r2 = edgeReq(e2);
  for (const [d, v] of Object.entries(r1)) {
    if (d in r2 && r2[d] !== v && !loopNodes.has(d)) return true;
  }
  return false;
}

/**
 * Validates ONE agent's own graph_config (the structure used both by a
 * standalone agent record and by a flow's agent_graphs[node_id]). Returns a
 * list of human-readable problem strings; [] means valid. Mirrors
 * src/validate/agent_graph_inputs.py::validate_agent_graph.
 */
export function validateAgentGraphInputs(graphConfig: PromptInputGraphConfig | undefined | null): string[] {
  const problems: string[] = [];
  const nodesConfig = graphConfig?.nodes || {};
  const edges = graphConfig?.edges || [];

  const nodeInputs = nodeInputsMap(nodesConfig);

  for (const [nodeId, inputs] of Object.entries(nodeInputs)) {
    if (inputs.length === 0) {
      problems.push(
        `Sub-agent '${nodeId}' has no inputs — its user_prompt has no {name} placeholders.`
      );
    }
  }

  for (const edge of edges) {
    if (!(edge.to_node in nodesConfig)) continue;
    const target = edge.target_input || 'input';
    if (!(nodeInputs[edge.to_node] || []).includes(target)) {
      problems.push(
        `Edge ${edge.from_node} -> ${edge.to_node} targets unknown input '${target}' on sub-agent '${edge.to_node}'.`
      );
    }
  }

  const edgeKeyCounts = new Map<string, { count: number; fromNode: string; outputPath: string | undefined; toNode: string; targetInput: string }>();
  for (const edge of edges) {
    const targetInput = edge.target_input || 'input';
    const key = `${edge.from_node}\u0000${edge.output_path ?? ''}\u0000${edge.to_node}\u0000${targetInput}`;
    const existing = edgeKeyCounts.get(key);
    if (existing) {
      existing.count += 1;
    } else {
      edgeKeyCounts.set(key, {
        count: 1,
        fromNode: edge.from_node,
        outputPath: edge.output_path,
        toNode: edge.to_node,
        targetInput
      });
    }
  }
  for (const { count, fromNode, outputPath, toNode, targetInput } of edgeKeyCounts.values()) {
    if (count > 1) {
      problems.push(
        `Edges ${fromNode} -> ${toNode} (output_path=${outputPath === undefined ? 'None' : `'${outputPath}'`}, ` +
        `target_input='${targetInput}') are identical duplicates.`
      );
    }
  }

  const edgeReq = computeEdgeReq(nodesConfig, edges);
  const loopNodes = computeLoopNodes(nodesConfig, edges);

  const groups = new Map<string, { toNode: string; target: string; edges: PromptInputGraphEdge[] }>();
  for (const edge of edges) {
    const target = edge.target_input || 'input';
    const key = `${edge.to_node}\u0000${target}`;
    const group = groups.get(key);
    if (group) {
      group.edges.push(edge);
    } else {
      groups.set(key, { toNode: edge.to_node, target, edges: [edge] });
    }
  }

  for (const { toNode, target, edges: groupEdges } of groups.values()) {
    if (groupEdges.length < 2) continue;
    for (let i = 0; i < groupEdges.length; i++) {
      for (let j = i + 1; j < groupEdges.length; j++) {
        const e1 = groupEdges[i];
        const e2 = groupEdges[j];
        if (!edgesExclusive(e1, e2, edgeReq, loopNodes)) {
          problems.push(
            `'${toNode}.${target}' is fed by ${e1.from_node}->${toNode} and ${e2.from_node}->${toNode}, ` +
            'which can both fire. Use separate inputs.'
          );
        }
      }
    }
  }

  for (const [nodeId, nodeCfg] of Object.entries(nodesConfig)) {
    for (const [pathName, pathCfg] of Object.entries(nodeCfg.output_paths || {})) {
      if (typeof pathCfg !== 'object' || pathCfg === null) continue;
      if ((pathCfg as any).return_behavior !== 'previous_output') continue;
      const returnInput = (pathCfg as any).return_input || 'input';
      if (!(nodeInputs[nodeId] || []).includes(returnInput)) {
        problems.push(
          `Output path '${pathName}' on sub-agent '${nodeId}' has return_input '${returnInput}', ` +
          "which is not one of its own inputs."
        );
      }
    }
  }

  return problems;
}

/**
 * Validates the FLOW-canvas edges feeding every agent node named in
 * agentGraphsByNodeId (node_id -> that agent's own graph_config, used to
 * compute agentExternalInputs). Mirrors
 * src/validate/agent_graph_inputs.py::validate_flow_agent_edges.
 */
export function validateFlowAgentEdges(
  flowGraphConfig: { edges?: PromptInputGraphEdge[] } | undefined | null,
  agentGraphsByNodeId: Record<string, PromptInputGraphConfig>
): string[] {
  const problems: string[] = [];
  const edges = flowGraphConfig?.edges || [];

  for (const [agentNodeId, agentGraph] of Object.entries(agentGraphsByNodeId)) {
    const external = new Set(agentExternalInputs(agentGraph));
    const incoming = edges.filter(edge => edge.to_node === agentNodeId && !edge.is_loop);

    const seen: Record<string, PromptInputGraphEdge[]> = {};
    for (const edge of incoming) {
      const target = agentTargetInput(edge);
      if (!external.has(target)) {
        problems.push(
          `Flow edge from '${edge.from_node}' targets unknown input '${target}' on agent node '${agentNodeId}'.`
        );
        continue;
      }
      (seen[target] = seen[target] || []).push(edge);
    }

    for (const [target, groupEdges] of Object.entries(seen)) {
      if (groupEdges.length > 1) {
        const sources = groupEdges.map(e => e.from_node || '?').join(', ');
        problems.push(
          `Agent node '${agentNodeId}' input '${target}' is fed by multiple flow edges (${sources}); ` +
          "flow edges have no output paths so this can't be made exclusive."
        );
      }
    }
  }

  return problems;
}
