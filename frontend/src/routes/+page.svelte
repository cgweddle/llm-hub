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
  import FloatingEdge from './FloatingEdge.svelte';
  import '@xyflow/svelte/dist/style.css';

  const nodeTypes = {
    selectorNode: ColorSelectorNode
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

  let availableNodes = ['Color Picker', 'Text Node', 'Decision Node'];

  function addNode(nodeName: string, position: { x: number; y: number }) {
    const newNode: Node = {
      id: String(Date.now()),
      type: nodeName === 'Color Picker' ? 'selectorNode' : 'default',
      data: { label: nodeName, handles: ['a'] },
      position,
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    };
    nodes = [...nodes, newNode];
  }

  function onConnect(params) {
    try {
      console.log('Connecting:', params);
      edges = addEdge(params, edges);
      console.log('Edges after connection:', edges);
    } catch (error) {
      console.error('Error creating edge:', error);
    }
  }
</script>

<div class="app-container">
  <div class="node-window">
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
  }

  .node-window {
    width: 200px;
    background: #f0f0f0;
    padding: 10px;
    border-right: 1px solid #ccc;
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
</style>