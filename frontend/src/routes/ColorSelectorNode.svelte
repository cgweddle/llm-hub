<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];

  let { color, handles } = data;
  let expanded = false;
</script>

<div class="customNode">
  <div class="customNodeBody">
    <div class="node-header">
      <span class="node-title">Color Picker</span>
      <button 
        class="expand-button"
        on:click={() => (expanded = !expanded)}
        on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { expanded = !expanded; } }}
        aria-label={expanded ? 'Collapse node' : 'Expand node'}
      >
        {expanded ? '−' : '+'}
      </button>
    </div>
    <div class="node-content">
      {#if expanded}
        <div class="expanded-content">
          <div class="color-display">
            <span>Color: </span>
            <span class="color-value" style="color: {$color}">{$color}</span>
          </div>
          <input
            class="nodrag color-input"
            type="color"
            on:input={(event) => {
              $color = event.currentTarget.value;
            }}
            value={$color}
          />
        </div>
      {:else}
        <div class="collapsed-content">
          <div class="color-preview" style="background-color: {$color}"></div>
          <span class="color-text">Click to expand</span>
        </div>
      {/if}
    </div>
  </div>
</div>

<!-- Target Handle -->
<Handle type="target" position={Position.Left} style="background: #555;" {isConnectable} />

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
    width: 200px;
    min-height: 100px;
    position: relative;
    cursor: move;
  }

  .customNodeBody {
    width: 100%;
    min-height: 100px;
    border: 3px solid #4a90e2;
    position: relative;
    overflow: hidden;
    border-radius: 12px;
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
  }

  .customNodeBody:hover {
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px 8px;
    border-bottom: 1px solid #dee2e6;
    background: #fff;
    border-radius: 9px 9px 0 0;
  }

  .node-title {
    font-weight: 600;
    color: #495057;
    font-size: 14px;
  }

  .expand-button {
    width: 24px;
    height: 24px;
    border: none;
    background: #4a90e2;
    color: white;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.2s ease;
    line-height: 1;
  }

  .expand-button:hover {
    background: #357abd;
    transform: scale(1.1);
  }

  .expand-button:active {
    transform: scale(0.95);
  }

  .node-content {
    padding: 12px 16px;
  }

  .expanded-content {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .color-display {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: #6c757d;
  }

  .color-value {
    font-weight: 600;
    font-family: 'Courier New', monospace;
  }

  .color-input {
    width: 100%;
    height: 32px;
    border: 2px solid #dee2e6;
    border-radius: 6px;
    cursor: pointer;
    transition: border-color 0.2s ease;
  }

  .color-input:hover {
    border-color: #4a90e2;
  }

  .collapsed-content {
    display: flex;
    align-items: center;
    gap: 12px;
    justify-content: center;
  }

  .color-preview {
    width: 24px;
    height: 24px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  .color-text {
    font-size: 12px;
    color: #6c757d;
    font-style: italic;
  }

  /* The handles should be along the borders */
  :global(div.customHandle) {
    width: 20px;
    height: 20px;
    background: #4a90e2;
    position: absolute;
    border-radius: 50%;
    opacity: 0.8;
    border: 2px solid #fff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  }

  /* Ensure that only the inside of the node is draggable */
  .customNodeBody {
    pointer-events: auto;
  }

  /* Prevent text selection on the expand button */
  .expand-button {
    user-select: none;
    -webkit-user-select: none;
  }
</style>
