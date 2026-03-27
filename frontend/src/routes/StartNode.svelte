<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';

  type $$Props = Omit<NodeProps, 'id'>;
  export let isConnectable: $$Props['isConnectable'];
</script>

<div class="start-node">
  <span class="start-label">Start</span>
  <Handle
    type="source"
    position={Position.Right}
    id="start"
    style="background: #10b981; border: 2px solid black; width: 12px; height: 12px; border-radius: 50%;"
    {isConnectable}
  />
</div>

<style>
  .start-node {
    background: #10b981;
    color: white;
    padding: 8px 18px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 13px;
    border: 2px solid #059669;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.4);
    display: flex;
    align-items: center;
    min-width: 60px;
    justify-content: center;
  }

  .start-label {
    user-select: none;
  }
</style>
