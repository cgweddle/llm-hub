<script lang="ts">
  import { writable } from 'svelte/store';
  import {
    SvelteFlow,
    Background,
    Controls,
    MiniMap,
    Position,
    type Node,
    type Edge,
    addEdge,
    ConnectionLineType,
    MarkerType,
    type Viewport
  } from '@xyflow/svelte';

  import ColorSelectorNode from './ColorSelectorNode.svelte';
  import ToolNode from './ToolNode.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import CondaEnvironmentsPanel from './CondaEnvironmentsPanel.svelte';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import { validateTwoTools, createFlow, executeFlow, getFlowDetails, type ValidationResult, type Tool, type Agent, type FlowCreateRequest, type Flow as FlowType } from '../lib/api';
  import { buildEnhancedGraphConfig } from '$lib/flowBuilder';
  import { autoLayoutNodes } from '$lib/elkLayout';
  import '@xyflow/svelte/dist/style.css';
  import type { PageData } from './$types';

  export let data: PageData;

  // Track viewport for coordinate conversion
  let viewport: Viewport = { x: 0, y: 0, zoom: 1 };

  // Validation state
  let validationMessage = '';
  let showValidationToast = false;
  let validationSuccess = false;

  // Flow save state
  let flowName = '';
  let flowDescription = '';
  let showSaveDialog = false;
  let isSaving = false;

  // Conda environment state
  let selectedCondaEnv: string | null = null;

  const nodeTypes = {
    selectorNode: ColorSelectorNode,
    toolNode: ToolNode
  };

  const edgeTypes = {
    floating: FloatingEdge
  };

  const defaultEdgeOptions = {
    style: 'stroke-width: 3; stroke: black;',
    type: 'floating',
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: 'black'
    }
  };

  const connectionLineStyle = 'stroke: black; stroke-width: 3;';

  // Simple working example
  let nodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'Input' },
      position: { x: 100, y: 100 },
      sourcePosition: Position.Right
    },
    {
      id: '2',
      type: 'selectorNode',
      data: { color: writable('#ff0000'), handles: ['a', 'b'] },
      position: { x: 400, y: 100 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    },
    {
      id: '3',
      type: 'output',
      data: { label: 'Output' },
      position: { x: 700, y: 100 },
      targetPosition: Position.Left
    }
  ];

  let edges: Edge[] = [];

  // Use tools and agents from database instead of hardcoded nodes
  $: availableTools = data.tools.map(tool => tool.name);
  $: availableAgents = data.agents.map(agent => agent.name);
  $: availableFlows = data.flows;

  function addNode(nodeName: string, position: { x: number; y: number }) {
    // Find the tool from the database
    const tool = data.tools.find((t: Tool) => t.name === nodeName);

    const newNode: Node = {
      id: String(Date.now()),
      type: nodeName === 'Color Picker' ? 'selectorNode' : 'toolNode',
      data: {
        label: nodeName,
        handles: ['a'],
        toolId: tool?.id, // Store tool ID for validation
        // Pass full tool data for ToolNode
        name: tool?.name || nodeName,
        description: tool?.description || '',
        script_code: tool?.script_code || '',
        input_schema: tool?.input_schema || null,
        output_schema: tool?.output_schema || null
      },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };
    nodes = [...nodes, newNode];
  }

  async function onConnect(params) {
    try {
      console.log('Connecting:', params);

      // Get source and target nodes
      const sourceNode = nodes.find(n => n.id === params.source);
      const targetNode = nodes.find(n => n.id === params.target);

      // If both nodes have tool IDs, validate compatibility
      if (sourceNode?.data?.toolId && targetNode?.data?.toolId) {
        const validation = await validateTwoTools(
          sourceNode.data.toolId,
          targetNode.data.toolId
        );

        if (!validation.compatible) {
          // Show error toast
          validationSuccess = false;
          validationMessage = `Incompatible tools: ${validation.issues.join(', ')}`;
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 5000);
          return; // Don't create the connection
        } else {
          // Show success toast
          validationSuccess = true;
          validationMessage = 'Tools are compatible!';
          showValidationToast = true;
          setTimeout(() => { showValidationToast = false; }, 3000);
        }
      }

      // Create the edge
      edges = addEdge(params, edges);
      console.log('Edges after connection:', edges);
    } catch (error) {
      console.error('Error creating edge:', error);
      validationSuccess = false;
      validationMessage = `Connection error: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Save the current visual flow to database
   */
  async function saveFlow() {
    try {
      isSaving = true;

      // Convert visual flow to graph_config
      const graphConfig = buildEnhancedGraphConfig(nodes, edges, data.tools);

      // Create flow request
      const flowData: FlowCreateRequest = {
        name: flowName,
        description: flowDescription,
        graph_config: graphConfig,
        is_public: false,
        user_id: data.user?.id || 1,  // Use user ID or default to 1
        conda_env: selectedCondaEnv || undefined  // Store conda env as separate field
      };

      // Send to backend
      const createdFlow = await createFlow(flowData);

      // Show success
      validationSuccess = true;
      validationMessage = `Flow "${createdFlow.name}" saved successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

      // Reset and close dialog
      showSaveDialog = false;
      flowName = '';
      flowDescription = '';

    } catch (error) {
      // Show error
      validationSuccess = false;
      validationMessage = `Failed to save flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    } finally {
      isSaving = false;
    }
  }

  /**
   * Load a saved flow and recreate its nodes and edges
   */
  async function loadFlow(flowId: number) {
    try {
      // Fetch flow details with graph_config
      const flow = await getFlowDetails(flowId);

      if (!flow.graph_config) {
        throw new Error('Flow has no graph_config');
      }

      const graphConfig = flow.graph_config;

      // Clear existing nodes and edges
      nodes = [];
      edges = [];

      // Recreate nodes from graph_config
      const nodeMap = new Map<string, Node>();

      for (const [nodeId, nodeConfig] of Object.entries(graphConfig.nodes)) {
        if (nodeConfig.node_type === 'tool') {
          // Find the tool from data.tools
          const tool = data.tools.find((t: Tool) => t.id === nodeConfig.id);

          if (tool) {
            const newNode: Node = {
              id: nodeId,
              type: 'toolNode',
              data: {
                label: tool.name,
                handles: ['a'],
                toolId: tool.id,
                name: tool.name,
                description: tool.description,
                script_code: tool.script_code,
                input_schema: tool.input_schema,
                output_schema: tool.output_schema
              },
              position: { x: 100 + Math.random() * 400, y: 100 + Math.random() * 300 },
              sourcePosition: Position.Right,
              targetPosition: Position.Left
            };
            nodeMap.set(nodeId, newNode);
          }
        }
        // TODO: Add agent node support
      }

      // Recreate edges from graph_config with handle information
      const newEdges: Edge[] = [];
      for (const edgeConfig of graphConfig.edges) {
        if (nodeMap.has(edgeConfig.from_node) && nodeMap.has(edgeConfig.to_node)) {
          const edge: Edge = {
            id: `${edgeConfig.from_node}-${edgeConfig.to_node}`,
            source: edgeConfig.from_node,
            target: edgeConfig.to_node,
            ...defaultEdgeOptions
          };

          // Add sourceHandle and targetHandle if mapping exists
          if (edgeConfig.mapping && Object.keys(edgeConfig.mapping).length > 0) {
            // For now, use the first mapping entry
            // TODO: Handle multiple output→input mappings (might need multiple edges)
            const [outputField, inputParam] = Object.entries(edgeConfig.mapping)[0];
            edge.sourceHandle = `output-${outputField}`;
            edge.targetHandle = `input-${inputParam}`;
          }

          newEdges.push(edge);
        }
      }

      // Auto-layout the nodes using ELK
      const layoutedNodes = await autoLayoutNodes(Array.from(nodeMap.values()), newEdges);
      nodes = layoutedNodes;
      edges = newEdges;

      // Set the conda environment from the loaded flow
      selectedCondaEnv = flow.conda_env;

      // Show success
      validationSuccess = true;
      validationMessage = `Flow "${flow.name}" loaded successfully!`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 3000);

    } catch (error) {
      validationSuccess = false;
      validationMessage = `Failed to load flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }

  /**
   * Execute a saved flow
   */
  async function runFlow(flowId: number, initialInput: Record<string, any>) {
    try {
      const result = await executeFlow(flowId, initialInput, selectedCondaEnv);

      if (result.status === 'completed') {
        console.log('✓ Flow completed successfully');
        console.log('Final output:', result.final_output);
        console.log('Execution trace:', result.execution_trace);

        // Show success
        validationSuccess = true;
        validationMessage = `Flow completed! Output: ${JSON.stringify(result.final_output)}`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 5000);
      } else {
        console.error('✗ Flow failed:', result.error);
        validationSuccess = false;
        validationMessage = `Flow failed: ${result.error}`;
        showValidationToast = true;
        setTimeout(() => { showValidationToast = false; }, 5000);
      }

    } catch (error) {
      console.error('Flow execution error:', error);
      validationSuccess = false;
      validationMessage = `Failed to execute flow: ${error}`;
      showValidationToast = true;
      setTimeout(() => { showValidationToast = false; }, 5000);
    }
  }
</script>

<div class="app-container">
  <!-- Validation Toast -->
  {#if showValidationToast}
    <div class="validation-toast" class:success={validationSuccess} class:error={!validationSuccess}>
      <span class="toast-icon">{validationSuccess ? '✓' : '✗'}</span>
      <span class="toast-message">{validationMessage}</span>
      <button class="toast-close" on:click={() => showValidationToast = false}>×</button>
    </div>
  {/if}

  <div class="node-window">
    <div class="user-section">
      {#if data.user}
        <div class="current-user">
          <strong>{data.user.username}</strong>
          <small>{data.user.email}</small>
        </div>
        <form method="POST" action="/logout" class="mt-2">
          <Button type="submit" variant="outline" class="w-full" size="sm">
            Logout
          </Button>
        </form>
      {:else}
        <div class="space-y-2">
          <Button class="w-full" onclick={() => window.location.href = '/login'}>
            Sign In
          </Button>
          <Button variant="outline" class="w-full" onclick={() => window.location.href = '/register'}>
            Register
          </Button>
        </div>
      {/if}
    </div>

    <CondaEnvironmentsPanel bind:selectedEnv={selectedCondaEnv} />

    <div class="flows-section">
      <h4>Available Flows</h4>
      <Button class="w-full mb-2" size="sm" on:click={() => showSaveDialog = true}>
        Create New Flow
      </Button>
      {#each availableFlows as flow}
        <div
          class="flow-item"
          on:click={() => loadFlow(flow.id)}
        >
          {flow.name}
        </div>
      {/each}
    </div>

    <h4>Available Tools</h4>
    {#each availableTools as tool}
      <div
        class="draggable-node"
        draggable="true"
        on:dragstart={(event) => event.dataTransfer?.setData('text/plain', tool)}
      >
        {tool}
      </div>
    {/each}

    <h4>Available Agents</h4>
    {#each availableAgents as agent}
      <div
        class="draggable-node"
        draggable="true"
        on:dragstart={(event) => event.dataTransfer?.setData('text/plain', agent)}
      >
        {agent}
      </div>
    {/each}
  </div>

  <div
    class="flow-container"
    role="application"
    on:dragover={(event) => event.preventDefault()}
    on:drop={(event) => {
      event.preventDefault();
      const nodeName = event.dataTransfer?.getData('text/plain');
      if (!nodeName) return;

      // Get the flow container bounds
      const flowContainer = event.currentTarget;
      const rect = flowContainer.getBoundingClientRect();

      // Convert screen coordinates to flow coordinates using viewport
      const position = {
        x: (event.clientX - rect.left - viewport.x) / viewport.zoom,
        y: (event.clientY - rect.top - viewport.y) / viewport.zoom
      };

      addNode(nodeName, position);
    }}
  >
    <!-- Flow Controls -->
    <div class="flow-controls">
      <Button on:click={() => showSaveDialog = true}>
        Save Flow
      </Button>
    </div>

    <SvelteFlow
      bind:nodes
      {nodeTypes}
      bind:edges
      {edgeTypes}
      {defaultEdgeOptions}
      connectionLineType={ConnectionLineType.Straight}
      {connectionLineStyle}
      style="background: #1A192B"
      fitView
      bind:viewport
      on:connect={onConnect}
    >
      <Background />
      <Controls />
      <MiniMap />
    </SvelteFlow>
  </div>

  <!-- Save Flow Dialog -->
  {#if showSaveDialog}
    <div class="dialog-overlay" on:click={() => showSaveDialog = false}>
      <div class="dialog-content" on:click={(e) => e.stopPropagation()}>
        <div class="dialog-header">
          <h3>Save Flow</h3>
          <button class="dialog-close" on:click={() => showSaveDialog = false}>×</button>
        </div>

        <div class="dialog-body">
          <div class="form-field">
            <Label for="flowName">Flow Name</Label>
            <Input id="flowName" bind:value={flowName} placeholder="My Data Pipeline" />
          </div>

          <div class="form-field">
            <Label for="flowDesc">Description</Label>
            <Input id="flowDesc" bind:value={flowDescription} placeholder="Describe what this flow does..." />
          </div>
        </div>

        <div class="dialog-footer">
          <Button variant="outline" on:click={() => showSaveDialog = false}>Cancel</Button>
          <Button on:click={saveFlow} disabled={isSaving || !flowName}>
            {isSaving ? 'Saving...' : 'Save'}
          </Button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .app-container {
    display: flex;
    height: 100vh;
    position: relative;
  }

  .node-window {
    width: 200px;
    background: #f0f0f0;
    padding: 10px;
    border-right: 1px solid #ccc;
  }

  .user-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .current-user {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 8px;
    background: white;
    border-radius: 4px;
    font-size: 12px;
  }

  .current-user strong {
    color: #333;
  }

  .current-user small {
    color: #666;
  }

  .flows-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .flow-item {
    padding: 5px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .flow-item:hover {
    background: #e8f4f8;
  }

  .draggable-node {
    padding: 5px;
    margin: 5px 0;
    background: white;
    border: 1px solid #ccc;
    cursor: grab;
  }

  .flow-container {
    flex-grow: 1;
    position: relative;
    padding: 20px;
  }

  /* Validation Toast */
  .validation-toast {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    font-size: 14px;
    font-weight: 500;
    animation: slideIn 0.3s ease-out;
    min-width: 300px;
    max-width: 500px;
  }

  .validation-toast.success {
    background-color: #10b981;
    color: white;
    border: 2px solid #059669;
  }

  .validation-toast.error {
    background-color: #ef4444;
    color: white;
    border: 2px solid #dc2626;
  }

  .toast-icon {
    font-size: 20px;
    font-weight: bold;
  }

  .toast-message {
    flex: 1;
    line-height: 1.4;
  }

  .toast-close {
    background: none;
    border: none;
    color: white;
    font-size: 24px;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0.8;
    transition: opacity 0.2s;
  }

  .toast-close:hover {
    opacity: 1;
  }

  @keyframes slideIn {
    from {
      transform: translateX(400px);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }

  /* Flow Controls */
  .flow-controls {
    position: absolute;
    top: 20px;
    left: 20px;
    z-index: 10;
  }

  /* Dialog Overlay */
  .dialog-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1001;
  }

  .dialog-content {
    background: white;
    border-radius: 8px;
    padding: 0;
    min-width: 400px;
    max-width: 500px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
  }

  .dialog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 1px solid #e5e7eb;
  }

  .dialog-header h3 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
    color: #111827;
  }

  .dialog-close {
    background: none;
    border: none;
    font-size: 28px;
    color: #6b7280;
    cursor: pointer;
    padding: 0;
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: background-color 0.2s;
  }

  .dialog-close:hover {
    background-color: #f3f4f6;
    color: #111827;
  }

  .dialog-body {
    padding: 24px;
  }

  .form-field {
    margin-bottom: 16px;
  }

  .form-field:last-child {
    margin-bottom: 0;
  }

  .dialog-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 24px;
    border-top: 1px solid #e5e7eb;
    background-color: #f9fafb;
    border-bottom-left-radius: 8px;
    border-bottom-right-radius: 8px;
  }
</style>