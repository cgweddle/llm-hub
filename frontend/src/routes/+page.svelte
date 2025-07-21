
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
 
    useUpdateNodeInternals
 
  } from '@xyflow/svelte';
 
  import ColorSelectorNode from './ColorSelectorNode.svelte';
 
  import FloatingEdge from './FloatingEdge.svelte';
 
  import '@xyflow/svelte/dist/style.css';
 
  const nodeTypes = {
    selectorNode: ColorSelectorNode
  };
 
  const bgColor = writable('#1A192B');
 
  const initialNodes: Node[] = [
    {
      id: '1',
      type: 'input',
      data: { label: 'An input node', handles: ['a'] },
      position: { x: 0, y: 50 },
      sourcePosition: Position.Right
    },
    {
      id: '2',
      type: 'selectorNode',
      data: { color: bgColor, handles: ['a', 'b'] },
      style: 'border: 1px solid #777; padding: 10px;',
      position: { x: 300, y: 50 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left
    },
    {
      id: '3',
      type: 'output',
      data: { label: 'Output A', handles: ['a'] },
      position: { x: 650, y: 25 },
      targetPosition: Position.Left
    },
    {
      id: '4',
      type: 'output',
      data: { label: 'Output B', handles: ['a'] },
      position: { x: 650, y: 100 },
      targetPosition: Position.Left
    }
  ];
 
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
 
  const initialEdges: Edge[] = [];
 
  const nodes = writable<Node[]>(initialNodes);
  const edges = writable<Edge[]>(initialEdges);
 
  //const updateNodeInternals = useUpdateNodeInternals()
 
  let availableNodes = ['Color Picker', 'Text Node', 'Decision Node'];
 
  function addNode(nodeName: string, position: { x: number; y: number }) {
    nodes.update((existingNodes) => {
      const newNode: Node = {
        id: String(existingNodes.length + 1),
        type: nodeName === 'Color Picker' ? 'selectorNode' : 'default',
        data: { label: nodeName, handles: ['a'] },
        position,
        sourcePosition: Position.Right,
        targetPosition: Position.Left
      };
      return [...existingNodes, newNode];
    });
  }
 
  function onConnect(params) {
    edges.update((existingEdges) => addEdge(params, existingEdges));
   
    nodes.update((existingNodes) => {
      return existingNodes.map(node => {
        if (node.id === params.target) {
          const newHandle = `h${node.data.handles.length + 1}`;
          return { ...node, data: { ...node.data, handles: [...node.data.handles, newHandle] } };
        }
        return node;
      });
    });
  }
</script>
 
<div class="app-container">
  <div class="node-window">
    <h4>Available Nodes</h4>
    {#each availableNodes as node}
      <div class="draggable-node" draggable="true" on:dragstart={(event) => event.dataTransfer.setData('text/plain', node)}>
        {node}
      </div>
    {/each}
  </div>
 
  <div class="flow-container" on:dragover={(event) => event.preventDefault()} on:drop={(event) => {
    event.preventDefault();
    const nodeName = event.dataTransfer.getData('text/plain');
    const boundingRect = event.currentTarget.getBoundingClientRect();
    const position = {
      x: event.clientX - boundingRect.left - 50,
      y: event.clientY - boundingRect.top - 25,
    };
    addNode(nodeName, position);
  }}>
    <SvelteFlow {nodes} {nodeTypes} {edges} {edgeTypes} {defaultEdgeOptions} connectionLineType={ConnectionLineType.Straight} {connectionLineStyle} style="background: {$bgColor}" fitView on:connect={onConnect}>
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
 
 
 
ColorSelectorNode
<script lang="ts">
  import { Handle, Position, useConnection, type NodeProps } from '@xyflow/svelte';
 
  type $$Props = NodeProps;
  export let id: NodeProps['id'];
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];
 
  let { color, handles } = data;
  let expanded = false;
 
  const connection = useConnection();
  let isConnecting = false;
  let isTarget = false;
 
  $: isConnecting = !!$connection.startHandle?.nodeId;
  $: isTarget = !!$connection.startHandle && $connection.startHandle?.nodeId !== id;
  $: label = isTarget ? 'Drop here' : 'Drag to connect';
</script>
 
<!-- Target Handle (Fixed) -->
<Handle type="target" position={Position.Left} style="background: #555;" {isConnectable} />
 
<div class="customNode" on:click={() => (expanded = !expanded)}>
  <div
    class="customNodeBody"
    style:border-style={isTarget ? 'dashed' : 'solid'}
    style:background-color={isTarget ? '#ffcce3' : '#ccd9f6'}
  >
    {#if !isConnecting}
      <Handle class="customHandle" position={Position.Right} type="source" style="z-index: 1;" />
    {/if}
    <Handle class="customHandle" position={Position.Left} type="target" isConnectableStart={false} />
    {label}
  </div>
</div>
 
{#if expanded}
  <div class="node-container">
    <div>Custom Color Picker Node: <strong>{$color}</strong></div>
    <input
      class="nodrag"
      type="color"
      on:input={(event) => {
        $color = event.currentTarget.value;
      }}
      value={$color}
    />
  </div>
{/if}
 
<!-- Dynamically Generated Source Handles -->
{#each handles as handleId, index}
  <Handle
    type="source"
    position={Position.Right}
    id={handleId}
    style="top: {index * 20 + 10}px; background: #555;"
    {isConnectable}
  />
{/each}
 
<style>
  .customNode {
    width: 150px;
    height: 80px;
    position: relative;
    cursor: move;
  }
 
  .customNodeBody {
    width: 100%;
    height: 100%;
    border: 3px solid black;
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    display: flex;
    justify-content: center;
    align-items: center;
    font-weight: bold;
    cursor: default; /* Prevents dragging the body */
  }
 
  .customNode:before {
    content: '';
    position: absolute;
    top: -10px;
    left: 50%;
    height: 20px;
    width: 40px;
    transform: translate(-50%, 0);
    background: #d6d5e6;
    z-index: 1000;
    line-height: 1;
    border-radius: 4px;
    color: #fff;
    font-size: 9px;
    border: 2px solid #222138;
  }
 
  .node-container {
    padding: 10px;
  }
 
  /* The handles should be along the borders */
  :global(div.customHandle) {
    width: 20px;
    height: 20px;
    background: blue;
    position: absolute;
    border-radius: 50%;
    opacity: 0.7;
  }
 
  /* Position the handles along the borders */
  .customNodeBody > .customHandle.position-left {
    left: -10px;
    top: 50%;
    transform: translateY(-50%);
  }
 
  .customNodeBody > .customHandle.position-right {
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
  }
 
  .customNodeBody > .customHandle.position-top {
    top: -10px;
    left: 50%;
    transform: translateX(-50%);
  }
 
  .customNodeBody > .customHandle.position-bottom {
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
  }
 
  /* Ensure that only the inside of the node is draggable */
  .customNodeBody {
    pointer-events: auto;
  }
</style>
 
 
 
FloatingEdge
<svelte:options immutable />
 
<script lang="ts">
  import { getStraightPath, useInternalNode, type EdgeProps } from '@xyflow/svelte';
 
  import { getEdgeParams } from './utils';
 
  type $$Props = EdgeProps;
 
  export let source: EdgeProps['source'];
  export let target: EdgeProps['target'];
  export let markerEnd: EdgeProps['markerEnd'] = undefined;
  export let style: EdgeProps['style'] = undefined;
  export let id: EdgeProps['id'];
 
  $: sourceNode = useInternalNode(source);
  $: targetNode = useInternalNode(target);
 
  let edgePath: string | undefined;
 
  $: {
    if ($sourceNode && $targetNode) {
      const edgeParams = getEdgeParams($sourceNode, $targetNode);
      edgePath = getStraightPath({
        sourceX: edgeParams.sx,
        sourceY: edgeParams.sy,
        targetX: edgeParams.tx,
        targetY: edgeParams.ty
      })[0];
    } else {
      edgePath = undefined;
    }
  }
</script>
 
<path {id} marker-end={markerEnd} d={edgePath} {style} />