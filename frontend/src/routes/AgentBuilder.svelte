<script lang="ts">
  import { createEventDispatcher } from 'svelte';
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

  import AgentBuilderNode from './AgentBuilderNode.svelte';
  import AgentConfigModal from './AgentConfigModal.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import LLMProvidersPanel from './LLMProvidersPanel.svelte';

  import { getAgentTemplatesList, getAgentTemplate, type AgentTypeKey } from '$lib/agentTemplates';
  import { buildAgentGraphConfig, validateAgentGraph } from '$lib/agentGraphBuilder';
  import { createAgent, type Tool, type Agent, type AgentCreateData } from '$lib/api';
  import type { LLMProvider } from '$lib/store';
  import type { Viewport } from '@xyflow/svelte';

  const dispatch = createEventDispatcher<{
    back: void;
    agentCreated: Agent;
  }>();

  // Props
  export let tools: Tool[] = [];
  export let userId: number = 1;

  // Node and edge types for SvelteFlow
  const nodeTypes = {
    agentBuilderNode: AgentBuilderNode
  };

  const edgeTypes = {
    floating: FloatingEdge
  };

  // Canvas state
  let agentNodes: Node[] = [];
  let agentEdges: Edge[] = [];
  let agentViewport: Viewport = { x: 0, y: 0, zoom: 1 };

  // LLM provider state
  let selectedLLMProvider: LLMProvider | null = null;
  let llmProviders: LLMProvider[] = [];

  // Save dialog state
  let showSaveAgentDialog = false;
  let composedAgentName = '';
  let composedAgentDescription = '';
  let isSavingAgent = false;

  // Agent Config Modal state
  let showAgentConfigModal = false;
  let configNodeId = '';
  let configAgentType: AgentTypeKey = 'react';
  let configNodeName = '';
  let configNodePrompt = '';
  let configNodeTools: number[] = [];
  let configNodeLLM = '';
  let configNodeUserPrompt = '';

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
   * Add an agent node to the canvas
   */
  function addAgentNode(agentType: AgentTypeKey, position: { x: number; y: number }) {
    const template = getAgentTemplate(agentType);

    const newNode: Node = {
      id: String(Date.now()),
      type: 'agentBuilderNode',
      data: {
        agentType,
        name: template.name,
        systemPrompt: template.defaultSystemPrompt,
        userPrompt: '',
        assignedTools: [],
        llmProvider: ''
      },
      position,
      sourcePosition: Position.Bottom,
      targetPosition: Position.Top
    };

    agentNodes = [...agentNodes, newNode];
  }

  /**
   * Handle drop on canvas
   */
  function handleCanvasDrop(event: DragEvent) {
    event.preventDefault();
    const agentType = event.dataTransfer?.getData('agent-type');
    if (!agentType) return;

    const container = event.currentTarget as HTMLElement;
    const rect = container.getBoundingClientRect();

    const position = {
      x: (event.clientX - rect.left - agentViewport.x) / agentViewport.zoom,
      y: (event.clientY - rect.top - agentViewport.y) / agentViewport.zoom
    };

    addAgentNode(agentType as AgentTypeKey, position);
  }

  /**
   * Handle connection between nodes
   */
  function onConnect(params: any) {
    const newEdge: Edge = {
      id: `${params.source}-${params.target}`,
      source: params.source,
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
   * Open config modal for an agent node
   */
  function openAgentNodeConfig(nodeId: string) {
    const node = agentNodes.find(n => n.id === nodeId);
    if (!node) return;

    configNodeId = nodeId;
    configAgentType = node.data.agentType || 'react';
    configNodeName = node.data.name || '';
    configNodePrompt = node.data.systemPrompt || '';
    configNodeUserPrompt = node.data.userPrompt || '';
    configNodeTools = node.data.assignedTools || [];
    configNodeLLM = node.data.llmProvider || '';
    showAgentConfigModal = true;
  }

  /**
   * Save agent node config from modal
   */
  function handleAgentConfigSave(event: CustomEvent<{
    nodeId: string;
    name: string;
    systemPrompt: string;
    userPrompt: string;
    assignedTools: number[];
    llmProvider: string;
  }>) {
    const { nodeId, name, systemPrompt, userPrompt, assignedTools, llmProvider } = event.detail;

    agentNodes = agentNodes.map(node => {
      if (node.id === nodeId) {
        return {
          ...node,
          data: {
            ...node.data,
            name,
            systemPrompt,
            userPrompt,
            assignedTools,
            llmProvider
          }
        };
      }
      return node;
    });

    showAgentConfigModal = false;
  }

  /**
   * Handle configure event from AgentBuilderNode
   */
  function handleConfigureAgentEvent(event: CustomEvent<{ nodeId: string; data: any }>) {
    openAgentNodeConfig(event.detail.nodeId);
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

      const agentData: AgentCreateData = {
        name: composedAgentName.trim(),
        description: composedAgentDescription.trim() || `Composed agent with ${agentNodes.length} sub-agents`,
        graph_config: graphConfig
      };

      const createdAgent = await createAgent(userId, agentData);

      showToastMessage(`Composed agent "${createdAgent.name}" created successfully!`, true);

      // Reset state
      showSaveAgentDialog = false;
      composedAgentName = '';
      composedAgentDescription = '';
      agentNodes = [];
      agentEdges = [];

      // Notify parent
      dispatch('agentCreated', createdAgent);
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
    if (agentNodes.length === 0) return;
    if (!confirm('Clear all agent nodes? This cannot be undone.')) return;
    agentNodes = [];
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

    <div class="agent-types-section">
      <h4>Agent Types</h4>
      <div class="hint">Drag to canvas</div>
      {#each getAgentTemplatesList() as template}
        <div
          class="agent-type-item"
          draggable="true"
          ondragstart={(event) => event.dataTransfer?.setData('agent-type', template.type)}
          style="border-left-color: {template.color};"
        >
          <span class="agent-type-icon">{template.icon}</span>
          <div class="agent-type-info">
            <span class="agent-type-name">{template.name}</span>
            <span class="agent-type-desc">{template.description}</span>
          </div>
        </div>
      {/each}
    </div>

    <div class="tools-section">
      <h4>Available Tools</h4>
      <div class="hint">Assign tools via node config</div>
      {#each tools as tool}
        <div class="tool-item">
          <span>{tool.name}</span>
        </div>
      {/each}
    </div>

    <LLMProvidersPanel bind:selectedProvider={selectedLLMProvider} bind:providers={llmProviders} />

    <div class="actions-section">
      <Button size="sm" onclick={() => showSaveAgentDialog = true} class="w-full mb-2 bg-purple-600 hover:bg-purple-700" disabled={agentNodes.length === 0}>
        {#snippet children()}
          Save Complex Agent
        {/snippet}
      </Button>
      <Button size="sm" onclick={clearCanvas} class="w-full" variant="outline" disabled={agentNodes.length === 0}>
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
    onconfigureAgent={handleConfigureAgentEvent}
  >
    <div class="canvas-header">
      <h2>Visual Agent Builder</h2>
      <span class="node-count">{agentNodes.length} agent{agentNodes.length !== 1 ? 's' : ''}</span>
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
      <MiniMap nodeColor={(node) => {
        const template = getAgentTemplate(node.data?.agentType || 'react');
        return template.color;
      }} />
    </SvelteFlow>
  </div>
</div>

<!-- Save Agent Dialog -->
{#if showSaveAgentDialog}
  <div class="dialog-overlay" onclick={() => showSaveAgentDialog = false}>
    <div class="dialog-content" onclick={(e) => e.stopPropagation()}>
      <div class="dialog-header">
        <h3>Save Complex Agent</h3>
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

        <div class="agent-summary">
          <span class="summary-label">Contains:</span>
          <span class="summary-value">{agentNodes.length} sub-agent{agentNodes.length !== 1 ? 's' : ''}, {agentEdges.length} connection{agentEdges.length !== 1 ? 's' : ''}</span>
        </div>
      </div>

      <div class="dialog-footer">
        <Button variant="outline" onclick={() => showSaveAgentDialog = false}>Cancel</Button>
        <Button onclick={saveComposedAgent} disabled={isSavingAgent || !composedAgentName.trim()} class="bg-purple-600 hover:bg-purple-700">
          {isSavingAgent ? 'Saving...' : 'Save Agent'}
        </Button>
      </div>
    </div>
  </div>
{/if}

<!-- Agent Config Modal -->
<AgentConfigModal
  bind:open={showAgentConfigModal}
  nodeId={configNodeId}
  agentType={configAgentType}
  initialName={configNodeName}
  initialSystemPrompt={configNodePrompt}
  initialUserPrompt={configNodeUserPrompt}
  initialAssignedTools={configNodeTools}
  initialLLMProvider={configNodeLLM}
  {tools}
  {llmProviders}
  {selectedLLMProvider}
  on:save={handleAgentConfigSave}
  on:close={() => showAgentConfigModal = false}
/>

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

  .agent-types-section,
  .tools-section {
    margin-bottom: 15px;
    padding-bottom: 15px;
    border-bottom: 1px solid #ccc;
  }

  .agent-types-section h4,
  .tools-section h4 {
    margin: 0 0 8px 0;
    font-size: 13px;
    font-weight: 600;
    color: #333;
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
    cursor: grab;
    transition: all 0.2s;
  }

  .agent-type-item:hover {
    background: #f5f5f5;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }

  .agent-type-item:active {
    cursor: grabbing;
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

  .tool-item {
    padding: 6px 10px;
    margin: 4px 0;
    background: #f8f8f8;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    font-size: 12px;
    color: #555;
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
