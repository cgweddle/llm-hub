<script lang="ts">
  import { createEventDispatcher, setContext } from 'svelte';
  import { writable } from 'svelte/store';
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    Position,
    type Node,
    type Edge,
    MarkerType,
    ConnectionLineType
  } from '@xyflow/svelte';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";

  import ToolNode from './ToolNode.svelte';
  import StartNode from './StartNode.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import LLMProvidersPanel from './LLMProvidersPanel.svelte';

  import { getAgentTemplatesList, getAgentTemplate, type AgentTypeKey } from '$lib/agentTemplates';
  import { buildAgentGraphConfig, validateAgentGraph } from '$lib/agentGraphBuilder';
  import { autoLayoutNodes } from '$lib/elkLayout';
  import { createAgent, updateAgent, loadLLMProvidersConfig, type Agent, type AgentCreateData, type Evaluation } from '$lib/api';
  import type { LLMProvider } from '$lib/store';
  import type { Viewport } from '@xyflow/svelte';

  import type { AgentTemplate } from '$lib/agentTemplates';

  const dispatch = createEventDispatcher<{
    back: void;
    agentCreated: Agent;
    agentUpdated: Agent;
    configureNewAgent: { template: AgentTemplate };
  }>();

  // Props
  export let agents: Agent[] = [];
  export let userId: number = 1;
  export let evaluations: Evaluation[] = [];

  // Sidebar section collapse state
  let sectionsExpanded: Record<string, boolean> = {
    newAgent: false,
    availableAgents: false,
    evaluations: false
  };

  function toggleSection(section: string) {
    sectionsExpanded[section] = !sectionsExpanded[section];
  }

  // Provide llmProviders context for ToolNode
  const llmProvidersStore = writable<LLMProvider[]>([]);
  setContext('llmProviders', llmProvidersStore);

  // Node and edge types for SvelteFlow
  const nodeTypes = {
    toolNode: ToolNode,
    startNode: StartNode
  };

  const edgeTypes = {
    floating: FloatingEdge
  };

  // The Start node is always present on the canvas
  const START_NODE_ID = 'start';
  function createStartNode(): Node {
    return {
      id: START_NODE_ID,
      type: 'startNode',
      data: { label: 'Start' },
      position: { x: 50, y: 150 },
      deletable: false,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };
  }

  // Canvas state
  let agentNodes: Node[] = [createStartNode()];
  let agentEdges: Edge[] = [];
  let agentViewport: Viewport = { x: 0, y: 0, zoom: 1 };

  // LLM provider state
  let selectedLLMProvider: LLMProvider | null = null;
  let llmProviders: LLMProvider[] = [];

  // Sync llmProviders into the context store for ToolNode
  $: llmProvidersStore.set(llmProviders);

  // Save dialog state
  let loadedAgentId: number | null = null;
  let showSaveAgentDialog = false;
  let composedAgentName = '';
  let composedAgentDescription = '';
  let maxLoopIterations = 5;
  let isSavingAgent = false;
  let topLevelEvalIds: number[] = [];

  function toggleTopLevelEval(evalId: number) {
    if (topLevelEvalIds.includes(evalId)) {
      topLevelEvalIds = topLevelEvalIds.filter(id => id !== evalId);
    } else {
      topLevelEvalIds = [...topLevelEvalIds, evalId];
    }
  }


  // Toast state
  let showToast = false;
  let toastMessage = '';
  let toastSuccess = false;

  function showToastMessage(message: string, success: boolean) {
    toastMessage = message;
    toastSuccess = success;
    showToast = true;
    setTimeout(() => { showToast = false; }, success ? 3000 : 5000);
  }

  /**
   * Add an agent node to the canvas from a template type
   */
  function addAgentNode(agentType: AgentTypeKey, position: { x: number; y: number }) {
    const template = getAgentTemplate(agentType);

    const newNode: Node = {
      id: String(Date.now()),
      type: 'toolNode',
      data: {
        label: template.name,
        handles: ['a'],
        isAgent: true,
        agentType,
        name: template.name,
        description: template.description,
        system_prompt: template.defaultSystemPrompt,
        llm_provider: '',
        tool_ids: [],
        eval_ids: [],
        graph_config: {},
        script_code: '',
        main_function: '',
        input_schema: null,
        output_schema: null,
        runtimeLLM: null
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };

    agentNodes = [...agentNodes, newNode];
  }

  /**
   * Add a fully configured agent node to the canvas (called from parent after modal confirmation)
   */
  export function addConfiguredAgentNode(nodeData: {
    name: string;
    description: string;
    system_prompt: string;
    user_prompt?: string;
    llm_provider: string;
    tool_ids: number[];
    eval_ids: number[];
    output_paths?: Record<string, string | { description: string; return_behavior: string }>;
    agentType: AgentTypeKey;
  }) {
    // Calculate a default position based on existing nodes
    const xOffset = agentNodes.length * 250;
    const position = { x: 100 + xOffset, y: 150 };

    const template = getAgentTemplate(nodeData.agentType);

    const newNode: Node = {
      id: String(Date.now()),
      type: 'toolNode',
      data: {
        label: nodeData.name,
        handles: ['a'],
        isAgent: true,
        agentType: nodeData.agentType,
        name: nodeData.name,
        description: nodeData.description,
        system_prompt: nodeData.system_prompt,
        user_prompt: nodeData.user_prompt || '',
        llm_provider: nodeData.llm_provider,
        tool_ids: nodeData.tool_ids,
        eval_ids: nodeData.eval_ids,
        output_paths: nodeData.output_paths,
        graph_config: {},
        script_code: '',
        main_function: '',
        input_schema: null,
        output_schema: null,
        runtimeLLM: null
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };

    agentNodes = [...agentNodes, newNode];
  }

  /**
   * Update a node's data on the canvas (called from FullscreenNodeModal for unsaved nodes)
   */
  export function updateNodeData(nodeId: string, updatedData: any) {
    agentNodes = agentNodes.map(n =>
      n.id === nodeId ? { ...n, data: { ...n.data, ...updatedData } } : n
    );
  }

  /**
   * Add a node from an existing agent's configuration
   */
  function addExistingAgentNode(agentId: number, position: { x: number; y: number }) {
    const agent = agents.find(a => a.id === agentId);
    if (!agent) return;

    const entryKey = agent.graph_config.entry_point || Object.keys(agent.graph_config.nodes)[0];
    const nodeConfig = agent.graph_config.nodes[entryKey];

    const newNode: Node = {
      id: String(Date.now()),
      type: 'toolNode',
      data: {
        label: agent.name,
        handles: ['a'],
        isAgent: true,
        agentId: agent.id,
        agentType: nodeConfig?.agent_type || 'pydanticai',
        name: agent.name,
        description: agent.description || '',
        system_prompt: nodeConfig?.system_prompt || '',
        llm_provider: nodeConfig?.llm_provider || '',
        tool_ids: nodeConfig?.tool_ids || [],
        eval_ids: nodeConfig?.eval_ids || [],
        output_paths: nodeConfig?.output_paths || undefined,
        graph_config: agent.graph_config,
        script_code: '',
        main_function: '',
        input_schema: null,
        output_schema: agent.output_schema || null,
        runtimeLLM: null
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };

    agentNodes = [...agentNodes, newNode];
  }

  /**
   * Load an existing complex agent onto the canvas.
   * Reconstructs XYFlow nodes/edges from graph_config, then auto-layouts with ELK.
   */
  export async function loadAgent(agent: Agent) {
    loadedAgentId = agent.id;
    try {
      const config = agent.graph_config;
      if (!config?.nodes) {
        showToastMessage('Agent has no graph configuration', false);
        return;
      }

      // 1. Create Start node + agent nodes
      const newNodes: Node[] = [createStartNode()];

      for (const [nodeId, nodeConfig] of Object.entries(config.nodes) as [string, any][]) {
        newNodes.push({
          id: nodeId,
          type: 'toolNode',
          data: {
            label: nodeConfig.name || nodeId,
            handles: ['a'],
            isAgent: true,
            agentType: nodeConfig.agent_type || 'pydanticai',
            name: nodeConfig.name || nodeId,
            description: nodeConfig.description || '',
            system_prompt: nodeConfig.system_prompt || '',
            user_prompt: nodeConfig.user_prompt || '',
            llm_provider: nodeConfig.llm_provider || '',
            tool_ids: nodeConfig.tool_ids || [],
            eval_ids: nodeConfig.eval_ids || [],
            output_paths: nodeConfig.output_paths || undefined,
            graph_config: {},
            script_code: '',
            main_function: '',
            input_schema: null,
            output_schema: null,
            runtimeLLM: null
          },
          position: { x: 0, y: 0 },
          sourcePosition: Position.Right,
          targetPosition: Position.Left
        });
      }

      // 2. Create edges (without sourceHandle initially, for ELK)
      const newEdges: Edge[] = [];

      if (config.entry_point) {
        newEdges.push({
          id: `start-${config.entry_point}`,
          source: START_NODE_ID,
          target: config.entry_point,
          type: 'floating',
          markerEnd: { type: MarkerType.ArrowClosed }
        });
      }

      for (const edgeConfig of config.edges || []) {
        newEdges.push({
          id: `${edgeConfig.from_node}-${edgeConfig.to_node}`,
          source: edgeConfig.from_node,
          target: edgeConfig.to_node,
          type: 'floating',
          markerEnd: { type: MarkerType.ArrowClosed }
        });
      }

      // 3. Auto-layout with ELK
      const layoutedNodes = await autoLayoutNodes(newNodes, newEdges);

      // 4. Restore sourceHandle from output_path after layout
      for (const edgeConfig of config.edges || []) {
        if (edgeConfig.output_path) {
          const edge = newEdges.find(
            e => e.source === edgeConfig.from_node && e.target === edgeConfig.to_node
          );
          if (edge) {
            edge.sourceHandle = edgeConfig.output_path;
          }
        }
      }

      // 5. Apply to canvas
      agentNodes = layoutedNodes;
      agentEdges = newEdges;

      // 6. Pre-fill save dialog metadata
      composedAgentName = agent.name || '';
      composedAgentDescription = agent.description || '';
      maxLoopIterations = config.max_loop_iterations || 5;
      topLevelEvalIds = config.eval_ids || [];

      showToastMessage(`Agent "${agent.name}" loaded successfully!`, true);
    } catch (error) {
      showToastMessage(`Failed to load agent: ${error}`, false);
    }
  }

  /**
   * Compute drop position from a drag event
   */
  function getDropPosition(event: DragEvent): { x: number; y: number } {
    const container = event.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left - agentViewport.x) / agentViewport.zoom,
      y: (event.clientY - rect.top - agentViewport.y) / agentViewport.zoom
    };
  }

  /**
   * Handle drop on canvas — supports existing agents and tools (templates use click-to-configure)
   */
  function handleCanvasDrop(event: DragEvent) {
    event.preventDefault();
    const position = getDropPosition(event);

    const existingAgentId = event.dataTransfer?.getData('existing-agent-id');
    if (existingAgentId) {
      addExistingAgentNode(Number(existingAgentId), position);
      return;
    }
  }

  /**
   * Handle connection between nodes
   */
  function onConnect(params: any) {
    const newEdge: Edge = {
      id: `${params.source}-${params.target}`,
      source: params.source,
      sourceHandle: params.sourceHandle || undefined,
      target: params.target,
      type: 'floating',
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: '#10b981'
      }
    };

    agentEdges = [...agentEdges, newEdge];
  }


  /**
   * Save the composed agent
   */
  async function saveComposedAgent() {
    // Validate the graph
    const validation = validateAgentGraph(agentNodes, agentEdges);
    if (!validation.valid) {
      showToastMessage(validation.errors.join('; '), false);
      return;
    }

    if (!composedAgentName.trim()) {
      showToastMessage('Please enter a name for the composed agent', false);
      return;
    }

    try {
      isSavingAgent = true;

      const graphConfig = buildAgentGraphConfig(agentNodes, agentEdges);
      graphConfig.max_loop_iterations = maxLoopIterations;
      if (topLevelEvalIds.length > 0) {
        graphConfig.eval_ids = topLevelEvalIds;
      }

      let savedAgent: Agent;
      const agentName = composedAgentName.trim();
      const agentDescription = composedAgentDescription.trim() || `Composed agent with ${agentNodes.length} sub-agents`;

      if (loadedAgentId) {
        savedAgent = await updateAgent(loadedAgentId, {
          name: agentName,
          description: agentDescription,
          graph_config: graphConfig
        });
        showToastMessage(`Agent "${savedAgent.name}" updated successfully!`, true);
        dispatch('agentUpdated', savedAgent);
      } else {
        const agentData: AgentCreateData = {
          name: agentName,
          description: agentDescription,
          graph_config: graphConfig
        };
        savedAgent = await createAgent(userId, agentData);
        showToastMessage(`Composed agent "${savedAgent.name}" created successfully!`, true);
        dispatch('agentCreated', savedAgent);
      }

      // Reset state
      showSaveAgentDialog = false;
      loadedAgentId = null;
      composedAgentName = '';
      composedAgentDescription = '';
      topLevelEvalIds = [];
      agentNodes = [createStartNode()];
      agentEdges = [];

      dispatch('back');

    } catch (error) {
      showToastMessage(`Failed to save composed agent: ${error}`, false);
    } finally {
      isSavingAgent = false;
    }
  }

  /**
   * Clear the canvas
   */
  function clearCanvas() {
    const agentCount = agentNodes.filter(n => n.type !== 'startNode').length;
    if (agentCount === 0) return;
    if (!confirm('Clear all agent nodes? This cannot be undone.')) return;
    loadedAgentId = null;
    topLevelEvalIds = [];
    agentNodes = [createStartNode()];
    agentEdges = [];
  }

  function handleBack() {
    dispatch('back');
  }
</script>

<!-- Toast -->
{#if showToast}
  <div class="toast" class:success={toastSuccess} class:error={!toastSuccess}>
    <span>{toastMessage}</span>
    <button class="toast-close" onclick={() => showToast = false}>×</button>
  </div>
{/if}

<div class="agent-builder-container">
  <!-- Sidebar -->
  <div class="agent-builder-sidebar">
    <div class="sidebar-header">
      <Button size="sm" onclick={handleBack} class="w-full mb-3" variant="outline">
        {#snippet children()}
          ← Back to Flow Builder
        {/snippet}
      </Button>
      <h3 class="sidebar-title">Agent Builder</h3>
    </div>

    <!-- New Agent Section -->
    <div class="sidebar-section">
      <button class="section-header" onclick={() => toggleSection('newAgent')}>
        <span class="section-chevron" class:expanded={sectionsExpanded.newAgent}>&#9656;</span>
        <h4>New Agent</h4>
        <span class="section-count">{getAgentTemplatesList().length}</span>
      </button>
      {#if sectionsExpanded.newAgent}
        <div class="section-content">
          <div class="hint">Click to configure and add</div>
          {#each getAgentTemplatesList() as template}
            <button
              class="agent-type-item"
              onclick={() => dispatch('configureNewAgent', { template })}
              style="border-left-color: {template.color};"
              type="button"
            >
              <span class="agent-type-icon">{template.icon}</span>
              <div class="agent-type-info">
                <span class="agent-type-name">{template.name}</span>
                <span class="agent-type-desc">{template.description}</span>
              </div>
            </button>
          {/each}
        </div>
      {/if}
    </div>

    <!-- Available Agents Section -->
    <div class="sidebar-section">
      <button class="section-header" onclick={() => toggleSection('availableAgents')}>
        <span class="section-chevron" class:expanded={sectionsExpanded.availableAgents}>&#9656;</span>
        <h4>Available Agents</h4>
        <span class="section-count">{agents.length}</span>
      </button>
      {#if sectionsExpanded.availableAgents}
        <div class="section-content">
          {#if agents.length === 0}
            <div class="hint">No agents created yet</div>
          {:else}
            <div class="hint">Drag to canvas</div>
            {#each agents as agent}
              <div
                class="available-agent-item"
                draggable="true"
                ondragstart={(event) => event.dataTransfer?.setData('existing-agent-id', String(agent.id))}
              >
                <span class="available-agent-name">{agent.name}</span>
                {#if agent.description}
                  <span class="available-agent-desc">{agent.description}</span>
                {/if}
              </div>
            {/each}
          {/if}
        </div>
      {/if}
    </div>

    <LLMProvidersPanel bind:selectedProvider={selectedLLMProvider} bind:providers={llmProviders} />

    <!-- Agent-Level Evaluations -->
    <div class="sidebar-section">
      <button class="section-header" onclick={() => toggleSection('evaluations')}>
        <span class="section-chevron" class:expanded={sectionsExpanded.evaluations}>&#9656;</span>
        <h4>Evaluations</h4>
      </button>
      {#if sectionsExpanded.evaluations}
        <div class="section-content">
          {#if evaluations.length === 0}
            <div class="hint">No evaluations defined</div>
          {:else}
            {#each evaluations as evaluation}
              <label class="eval-item">
                <input
                  type="checkbox"
                  checked={topLevelEvalIds.includes(evaluation.id)}
                  onchange={() => toggleTopLevelEval(evaluation.id)}
                />
                <span class="eval-item-name">{evaluation.name}</span>
              </label>
            {/each}
          {/if}
        </div>
      {/if}
    </div>

    <div class="actions-section">
      <Button size="sm" onclick={() => showSaveAgentDialog = true} class="w-full mb-2 bg-purple-600 hover:bg-purple-700" disabled={agentNodes.filter(n => n.type !== 'startNode').length === 0}>
        {#snippet children()}
          {loadedAgentId ? 'Update Agent' : 'Save Complex Agent'}
        {/snippet}
      </Button>
      <Button size="sm" onclick={clearCanvas} class="w-full" variant="outline" disabled={agentNodes.filter(n => n.type !== 'startNode').length === 0}>
        {#snippet children()}
          Clear Canvas
        {/snippet}
      </Button>
    </div>
  </div>

  <!-- Canvas -->
  <div
    class="agent-builder-canvas"
    role="application"
    ondragover={(event) => event.preventDefault()}
    ondrop={handleCanvasDrop}
  >
    <div class="canvas-header">
      <h2>Visual Agent Builder</h2>
      <span class="node-count">{agentNodes.filter(n => n.type !== 'startNode').length} agent{agentNodes.filter(n => n.type !== 'startNode').length !== 1 ? 's' : ''}</span>
    </div>

    <SvelteFlow
      bind:nodes={agentNodes}
      {nodeTypes}
      bind:edges={agentEdges}
      {edgeTypes}
      defaultEdgeOptions={{
        style: 'stroke-width: 3; stroke: #10b981;',
        type: 'floating',
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: '#10b981'
        }
      }}
      connectionLineType={ConnectionLineType.Straight}
      connectionLineStyle="stroke: #10b981; stroke-width: 3;"
      style="background: #0f0f23"
      fitView
      bind:viewport={agentViewport}
      onconnect={onConnect}
    >
      <Background bgColor="#0f0f23" gap={20} />
      <Controls />
      <MiniMap nodeColor={() => '#007acc'} />
    </SvelteFlow>
  </div>
</div>

<!-- Save Agent Dialog -->
{#if showSaveAgentDialog}
  <div class="dialog-overlay" onclick={() => showSaveAgentDialog = false}>
    <div class="dialog-content" onclick={(e) => e.stopPropagation()}>
      <div class="dialog-header">
        <h3>{loadedAgentId ? 'Update Agent' : 'Save Complex Agent'}</h3>
        <button class="dialog-close" onclick={() => showSaveAgentDialog = false}>×</button>
      </div>

      <div class="dialog-body">
        <div class="form-field">
          <Label for="composedAgentName">Agent Name</Label>
          <Input id="composedAgentName" bind:value={composedAgentName} placeholder="My Complex Agent" />
        </div>

        <div class="form-field">
          <Label for="composedAgentDesc">Description</Label>
          <Input id="composedAgentDesc" bind:value={composedAgentDescription} placeholder="Describe what this agent does..." />
        </div>

        {#if agentEdges.filter(e => e.source !== START_NODE_ID).length > agentNodes.filter(n => n.type !== 'startNode').length - 1}
          <div class="form-field">
            <Label for="maxLoopIter">Max Loop Iterations</Label>
            <Input id="maxLoopIter" type="number" bind:value={maxLoopIterations} min={1} max={20} />
          </div>
        {/if}

        <div class="agent-summary">
          <span class="summary-label">Contains:</span>
          <span class="summary-value">{agentNodes.filter(n => n.type !== 'startNode').length} sub-agent{agentNodes.filter(n => n.type !== 'startNode').length !== 1 ? 's' : ''}, {agentEdges.filter(e => e.source !== START_NODE_ID).length} connection{agentEdges.filter(e => e.source !== START_NODE_ID).length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      <div class="dialog-footer">
        <Button variant="outline" onclick={() => showSaveAgentDialog = false}>Cancel</Button>
        <Button onclick={saveComposedAgent} disabled={isSavingAgent || !composedAgentName.trim()} class="bg-purple-600 hover:bg-purple-700">
          {isSavingAgent ? 'Saving...' : loadedAgentId ? 'Update' : 'Save'}
        </Button>
      </div>
    </div>
  </div>
{/if}

<style>
  .agent-builder-container {
    display: flex;
    height: 100%;
    width: 100%;
  }

  /* Sidebar */
  .agent-builder-sidebar {
    width: 220px;
    background: #f0f0f0;
    padding: 10px;
    border-right: 1px solid #ccc;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .sidebar-header {
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
    margin-bottom: 15px;
  }

  .sidebar-title {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: #a855f7;
    text-align: center;
  }

  .sidebar-section {
    margin-bottom: 2px;
    border-bottom: 1px solid #ccc;
  }

  .section-header {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    padding: 8px 4px;
    background: none;
    border: none;
    cursor: pointer;
    text-align: left;
    transition: background 0.15s;
  }

  .section-header:hover {
    background: #e5e5e5;
  }

  .section-header h4 {
    margin: 0;
    font-size: 13px;
    font-weight: 600;
    color: #333;
    flex: 1;
  }

  .section-chevron {
    font-size: 12px;
    color: #666;
    transition: transform 0.2s;
    display: inline-block;
  }

  .section-chevron.expanded {
    transform: rotate(90deg);
  }

  .section-count {
    font-size: 11px;
    color: #888;
    background: #e0e0e0;
    padding: 1px 6px;
    border-radius: 8px;
  }

  .section-content {
    padding: 0 4px 10px 4px;
  }

  .available-agent-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 6px 10px;
    margin: 4px 0;
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-left: 3px solid #a855f7;
    border-radius: 4px;
    cursor: grab;
    transition: all 0.2s;
  }

  .available-agent-item:hover {
    background: #f0edf5;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .available-agent-item:active {
    cursor: grabbing;
  }

  .available-agent-name {
    font-size: 12px;
    font-weight: 600;
    color: #333;
  }

  .available-agent-desc {
    font-size: 10px;
    color: #666;
    line-height: 1.3;
  }

  .hint {
    font-size: 11px;
    color: #888;
    margin-bottom: 8px;
    font-style: italic;
  }

  .agent-type-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px;
    margin: 6px 0;
    background: white;
    border: 1px solid #ccc;
    border-left: 4px solid;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
    width: 100%;
    text-align: left;
    font: inherit;
  }

  .agent-type-item:hover {
    background: #f5f5f5;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .agent-type-item:active {
    background: #ebebeb;
  }

  .agent-type-icon {
    font-size: 20px;
    line-height: 1;
  }

  .agent-type-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }

  .agent-type-name {
    font-size: 13px;
    font-weight: 600;
    color: #333;
  }

  .agent-type-desc {
    font-size: 10px;
    color: #666;
    line-height: 1.3;
  }

  .eval-item {
    padding: 5px 8px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .eval-item:hover {
    background: #e8f4f8;
  }

  .eval-item-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12px;
  }

  .actions-section {
    margin-top: auto;
    padding-top: 15px;
  }

  /* Canvas */
  .agent-builder-canvas {
    flex: 1;
    position: relative;
    background: #0f0f23;
  }

  .canvas-header {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 10;
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 10px 20px;
    background: rgba(168, 85, 247, 0.9);
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
  }

  .canvas-header h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: white;
  }

  .node-count {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.8);
    background: rgba(0, 0, 0, 0.2);
    padding: 4px 10px;
    border-radius: 12px;
  }

  /* Toast */
  .toast {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 10001;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 14px;
    font-weight: 500;
    min-width: 300px;
    max-width: 500px;
  }

  .toast.success {
    background-color: #10b981;
    color: white;
  }

  .toast.error {
    background-color: #ef4444;
    color: white;
  }

  .toast-close {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    margin-left: auto;
  }

  /* Dialog */
  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
  }

  .dialog-content {
    background: white;
    border-radius: 8px;
    min-width: 400px;
    max-width: 500px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
    border: 2px solid #a855f7;
  }

  .dialog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    background: #a855f7;
    color: white;
    border-radius: 6px 6px 0 0;
  }

  .dialog-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }

  .dialog-close {
    background: none;
    border: none;
    color: white;
    font-size: 24px;
    cursor: pointer;
  }

  .dialog-body {
    padding: 20px;
  }

  .form-field {
    margin-bottom: 16px;
  }

  .agent-summary {
    padding: 12px;
    background: #f5f0ff;
    border: 1px solid #e0d4f5;
    border-radius: 6px;
    display: flex;
    gap: 8px;
  }

  .summary-label {
    font-size: 13px;
    color: #666;
  }

  .summary-value {
    font-size: 13px;
    font-weight: 600;
    color: #a855f7;
  }

  .dialog-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid #e5e7eb;
    background-color: #f9fafb;
    border-radius: 0 0 6px 6px;
  }
</style>
