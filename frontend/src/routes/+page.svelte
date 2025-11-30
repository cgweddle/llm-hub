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
    MarkerType
  } from '@xyflow/svelte';

  import ColorSelectorNode from './ColorSelectorNode.svelte';
  import ToolNode from './ToolNode.svelte';
  import FloatingEdge from './FloatingEdge.svelte';
  import { Button } from "$lib/components/ui/button";
  import { validateTwoTools, type ValidationResult, type Tool } from '../lib/api';
  import '@xyflow/svelte/dist/style.css';
  import type { PageData } from './$types';

  export let data: PageData;

  // Validation state
  let validationMessage = '';
  let showValidationToast = false;
  let validationSuccess = false;

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

  // Use tools from database instead of hardcoded nodes
  $: availableNodes = data.tools.map(tool => tool.name);

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
        script_code: tool?.script_code || ''
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

    <h4>Available Nodes</h4>
    {#each availableNodes as node}
      <div
        class="draggable-node"
        draggable="true"
        role="button"
        tabindex="0"
        on:dragstart={(event) => event.dataTransfer.setData('text/plain', node)}
        on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.dataTransfer.setData('text/plain', node); } }}
      >
        {node}
      </div>
    {/each}
  </div>

  <div 
    class="flow-container" 
    role="application" 
    on:dragover={(event) => event.preventDefault()} 
    on:drop={(event) => {
      event.preventDefault();
      const nodeName = event.dataTransfer.getData('text/plain');
      const boundingRect = event.currentTarget.getBoundingClientRect();
      const position = {
        x: event.clientX - boundingRect.left - 50,
        y: event.clientY - boundingRect.top - 25,
      };
      addNode(nodeName, position);
    }}
  >
    <SvelteFlow 
      {nodes} 
      {nodeTypes} 
      {edges} 
      {edgeTypes} 
      {defaultEdgeOptions} 
      connectionLineType={ConnectionLineType.Straight} 
      {connectionLineStyle} 
      style="background: #1A192B" 
      fitView 
      on:connect={onConnect}
    >
      <Background />
      <Controls />
      <MiniMap />
    </SvelteFlow>
  </div>
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
</style>