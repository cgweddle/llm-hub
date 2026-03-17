<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { getAgentTemplate, type AgentTypeKey } from '$lib/agentTemplates';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];
  export let id: string;

  // Extract data with reactivity
  let agentType: AgentTypeKey;
  let name: string;
  let systemPrompt: string;
  let userPrompt: string;
  let assignedTools: number[];
  let llmProvider: string;

  $: ({
    agentType = 'react',
    name = 'Agent',
    systemPrompt = '',
    userPrompt = '',
    assignedTools = [],
    llmProvider = ''
  } = data);

  // Get template for styling
  $: template = getAgentTemplate(agentType);

  // Create display for assigned tools
  $: toolCount = assignedTools?.length || 0;

  function handleConfigClick(event: Event) {
    event.stopPropagation();
    // Dispatch event to open config modal
    const customEvent = new CustomEvent('configureAgent', {
      detail: { nodeId: id, data },
      bubbles: true,
      composed: true
    });
    (event.target as HTMLElement).dispatchEvent(customEvent);
  }
</script>

<div
  class="agent-builder-node"
  style="--agent-color: {template.color}; --agent-border: {template.borderColor};"
>
  <!-- Header with type color -->
  <div class="node-header">
    <span class="node-icon">{template.icon}</span>
    <span class="node-name">{name}</span>
    <button
      class="config-button"
      on:click={handleConfigClick}
      on:keydown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleConfigClick(e); }}
      aria-label="Configure agent"
      title="Configure agent"
    >
      ⚙️
    </button>
  </div>

  <!-- Body with type indicator and tools -->
  <div class="node-body">
    <div class="type-badge" style="background: {template.color};">
      {template.name}
    </div>

    {#if userPrompt}
      <div class="user-prompt-preview" title={userPrompt}>
        {userPrompt.length > 60 ? userPrompt.slice(0, 60) + '...' : userPrompt}
      </div>
    {/if}

    {#if toolCount > 0}
      <div class="tools-section">
        <span class="tools-label">Tools:</span>
        <span class="tools-count">{toolCount} assigned</span>
      </div>
    {:else}
      <div class="tools-section empty">
        <span class="tools-label">No tools assigned</span>
      </div>
    {/if}

    {#if llmProvider}
      <div class="llm-indicator">
        <span class="llm-icon">🤖</span>
        <span class="llm-name">{llmProvider}</span>
      </div>
    {/if}
  </div>
</div>

<!-- Input Handle (top) -->
<Handle
  type="target"
  position={Position.Top}
  id="input"
  style="background: #007acc; border: 2px solid black; width: 12px; height: 12px; border-radius: 50%;"
  {isConnectable}
/>

<!-- Output Handle (bottom, colored by agent type) -->
<Handle
  type="source"
  position={Position.Bottom}
  id="output"
  style="background: {template.color}; border: 2px solid {template.borderColor}; width: 12px; height: 12px; border-radius: 50%;"
  {isConnectable}
/>

<style>
  .agent-builder-node {
    width: 240px;
    min-height: 100px;
    border: 3px solid var(--agent-border);
    border-radius: 10px;
    background: #1e1e1e;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    overflow: hidden;
    cursor: move;
    transition: box-shadow 0.2s ease;
  }

  .agent-builder-node:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 20px color-mix(in srgb, var(--agent-color) 30%, transparent);
  }

  .node-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: var(--agent-color);
    color: white;
  }

  .node-icon {
    font-size: 18px;
    line-height: 1;
  }

  .node-name {
    flex: 1;
    font-size: 14px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .config-button {
    width: 26px;
    height: 26px;
    border: none;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    transition: background 0.2s ease;
  }

  .config-button:hover {
    background: rgba(255, 255, 255, 0.35);
  }

  .node-body {
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .type-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    align-self: flex-start;
  }

  .user-prompt-preview {
    font-size: 11px;
    font-style: italic;
    color: #a0a0a0;
    line-height: 1.3;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tools-section {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: #b0b0b0;
  }

  .tools-section.empty {
    color: #707070;
    font-style: italic;
  }

  .tools-label {
    font-weight: 500;
  }

  .tools-count {
    color: #10b981;
    font-weight: 600;
  }

  .llm-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: #2d2d30;
    border-radius: 4px;
    font-size: 11px;
    color: #9333ea;
  }

  .llm-icon {
    font-size: 12px;
  }

  .llm-name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
