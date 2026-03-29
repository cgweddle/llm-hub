<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];
  export let id: string;

  let triggerValue: string;
  $: triggerValue = data.triggerValue || '';

  function handleInput(event: Event) {
    const target = event.target as HTMLTextAreaElement;
    data.triggerValue = target.value;
    triggerValue = target.value;
  }
</script>

<div class="triggerNode">
  <div class="triggerNodeBody">
    <div class="node-header">
      <span class="node-title">{data.name || 'Text Input'}</span>
    </div>
    <div class="node-content">
      <textarea
        class="trigger-textarea"
        placeholder="Enter input text..."
        value={triggerValue}
        on:input={handleInput}
        rows="4"
      ></textarea>
    </div>
  </div>
</div>

<!-- Single output handle -->
<div class="output-handle-wrapper" style="top: 60px;">
  <span class="handle-label-outside output-label">Output</span>
  <Handle
    type="source"
    position={Position.Right}
    id=""
    style="background: #0e7a0d; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
    {isConnectable}
  />
</div>

<style>
  .triggerNode {
    width: 280px;
    min-height: 100px;
    position: relative;
    cursor: move;
  }

  .triggerNodeBody {
    width: 100%;
    min-height: 100px;
    border: 3px solid #e67e22;
    position: relative;
    overflow: visible;
    border-radius: 8px;
    background: #1e1e1e;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
  }

  .triggerNodeBody:hover {
    box-shadow: 0 6px 16px rgba(230, 126, 34, 0.4);
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2d2d30;
    background: #252526;
    border-radius: 5px 5px 0 0;
  }

  .node-title {
    font-weight: 600;
    color: #e67e22;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .node-content {
    padding: 12px;
  }

  .trigger-textarea {
    width: 100%;
    background: #2d2d30;
    color: #cccccc;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 8px;
    font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    resize: vertical;
    outline: none;
    line-height: 1.4;
    box-sizing: border-box;
  }

  .trigger-textarea:focus {
    border-color: #e67e22;
    box-shadow: 0 0 0 2px rgba(230, 126, 34, 0.3);
  }

  .trigger-textarea::placeholder {
    color: #6b6b6b;
  }

  /* Output handle wrapper and labels */
  .output-handle-wrapper {
    position: absolute;
    right: 0;
    transform: translateX(100%);
    display: flex;
    align-items: center;
    pointer-events: all;
  }

  .output-label {
    font-size: 11px;
    font-weight: 600;
    color: #ffffff;
    background: #0e7a0d;
    padding: 4px 10px;
    border-radius: 0 4px 4px 0;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(14, 122, 13, 0.3);
    transition: all 0.2s ease;
    border-top: none;
    border-bottom: none;
    border-right: none;
    border-left: 3px solid #0e7a0d;
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .output-label:hover {
    background: #0c6b0c;
    box-shadow: 0 3px 8px rgba(14, 122, 13, 0.5);
  }
</style>
