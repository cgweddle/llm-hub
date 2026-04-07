<script lang="ts">
  import { onMount } from 'svelte';
  import { 
    availableAgents, 
    availableTools, 
    availableFlows, 
    isLoadingAgents, 
    isLoadingTools, 
    isLoadingFlows,
    error,
    currentUser
  } from '../lib/store';
  import { fetchAvailableAgents, fetchAvailableTools, fetchAvailableFlows } from '../lib/api';
  import type { Agent, Tool, Flow } from '../lib/api';

  let activeTab: 'agents' | 'tools' | 'flows' = 'agents';
  let searchTerm = '';

  // Computed values for filtered items
  $: filteredAgents = $availableAgents.filter(agent => 
    agent.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    agent.description.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  $: filteredTools = $availableTools.filter(tool => 
    tool.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    tool.description.toLowerCase().includes(searchTerm.toLowerCase())
  );
  
  $: filteredFlows = $availableFlows.filter(flow => 
    flow.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    flow.description.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Load data when component mounts or user changes
  $: if ($currentUser) {
    loadAllData();
  }

  async function loadAllData() {
    if (!$currentUser) return;
    
    try {
      // Load all data in parallel
      const [agents, tools, flows] = await Promise.all([
        fetchAvailableAgents($currentUser.id),
        fetchAvailableTools($currentUser.id),
        fetchAvailableFlows($currentUser.id)
      ]);
      
      availableAgents.set(agents);
      availableTools.set(tools);
      availableFlows.set(flows);
    } catch (err) {
      error.set('Failed to load data');
      console.error('Error loading data:', err);
    }
  }

  function handleDragStart(event: DragEvent, item: Agent | Tool | Flow, type: string) {
    if (event.dataTransfer) {
      event.dataTransfer.setData('text/plain', JSON.stringify({ ...item, itemType: type }));
    }
  }

  function getItemIcon(type: string): string {
    switch (type) {
      case 'agents': return '🤖';
      case 'tools': return '🔧';
      case 'flows': return '🔄';
      default: return '📦';
    }
  }

  function getItemTypeLabel(type: string): string {
    switch (type) {
      case 'agents': return 'Agent';
      case 'tools': return 'Tool';
      case 'flows': return 'Flow';
      default: return 'Item';
    }
  }
</script>

<div class="panel">
  <div class="panel-header">
    <h3>Available Items</h3>
    <div class="search-box">
      <input 
        type="text" 
        placeholder="Search..." 
        bind:value={searchTerm}
        class="search-input"
      />
    </div>
  </div>

  <div class="tabs">
    <button 
      class="tab-button" 
      class:active={activeTab === 'agents'}
      on:click={() => activeTab = 'agents'}
    >
      🤖 Agents
    </button>
    <button 
      class="tab-button" 
      class:active={activeTab === 'tools'}
      on:click={() => activeTab = 'tools'}
    >
      🔧 Tools
    </button>
    <button 
      class="tab-button" 
      class:active={activeTab === 'flows'}
      on:click={() => activeTab = 'flows'}
    >
      🔄 Flows
    </button>
  </div>

  <div class="content">
    {#if $error}
      <div class="error-message">
        {$error}
        <button on:click={() => error.set(null)}>×</button>
      </div>
    {/if}

    {#if activeTab === 'agents'}
      <div class="items-section">
        <h4>Available Agents ({filteredAgents.length})</h4>
        {#if $isLoadingAgents}
          <div class="loading">Loading agents...</div>
        {:else if filteredAgents.length === 0}
          <div class="empty-state">No agents available</div>
        {:else}
          <div class="items-list">
            {#each filteredAgents as agent (agent.id)}
              <div 
                class="draggable-item agent-item"
                draggable="true"
                on:dragstart={(event) => handleDragStart(event, agent, 'agent')}
                title={agent.description}
              >
                <div class="item-header">
                  <span class="item-icon">{getItemIcon('agents')}</span>
                  <span class="item-name">{agent.name}</span>
                  {#if agent.is_public}
                    <span class="public-badge">Public</span>
                  {/if}
                </div>
                <div class="item-description">{agent.description}</div>
                <div class="item-meta">
                  <span class="item-type">{agent.graph_config && Object.keys(agent.graph_config.nodes || {}).length > 1 ? 'Multi-Agent' : (agent.graph_config?.nodes?.[agent.graph_config?.entry_point]?.agent_type || 'agent')}</span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {:else if activeTab === 'tools'}
      <div class="items-section">
        <h4>Available Tools ({filteredTools.length})</h4>
        {#if $isLoadingTools}
          <div class="loading">Loading tools...</div>
        {:else if filteredTools.length === 0}
          <div class="empty-state">No tools available</div>
        {:else}
          <div class="items-list">
            {#each filteredTools as tool (tool.id)}
              <div 
                class="draggable-item tool-item"
                draggable="true"
                on:dragstart={(event) => handleDragStart(event, tool, 'tool')}
                title={tool.description}
              >
                <div class="item-header">
                  <span class="item-icon">{getItemIcon('tools')}</span>
                  <span class="item-name">{tool.name}</span>
                  {#if tool.is_public}
                    <span class="public-badge">Public</span>
                  {/if}
                </div>
                <div class="item-description">{tool.description}</div>
                <div class="item-meta">
                  <span class="item-type">{tool.tool_type}</span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {:else if activeTab === 'flows'}
      <div class="items-section">
        <h4>Available Flows ({filteredFlows.length})</h4>
        {#if $isLoadingFlows}
          <div class="loading">Loading flows...</div>
        {:else if filteredFlows.length === 0}
          <div class="empty-state">No flows available</div>
        {:else}
          <div class="items-list">
            {#each filteredFlows as flow (flow.id)}
              <div 
                class="draggable-item flow-item"
                draggable="true"
                on:dragstart={(event) => handleDragStart(event, flow, 'flow')}
                title={flow.description}
              >
                <div class="item-header">
                  <span class="item-icon">{getItemIcon('flows')}</span>
                  <span class="item-name">{flow.name}</span>
                  {#if flow.is_public}
                    <span class="public-badge">Public</span>
                  {/if}
                </div>
                <div class="item-description">{flow.description}</div>
                <div class="item-meta">
                  <span class="item-type">Workflow</span>
                </div>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>
</div>

<style>
  .panel {
    width: 300px;
    height: 100vh;
    background: #f8f9fa;
    border-right: 1px solid #dee2e6;
    display: flex;
    flex-direction: column;
  }

  .panel-header {
    padding: 16px;
    border-bottom: 1px solid #dee2e6;
    background: white;
  }

  .panel-header h3 {
    margin: 0 0 12px 0;
    color: #495057;
    font-size: 18px;
  }

  .search-box {
    width: 100%;
  }

  .search-input {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid #ced4da;
    border-radius: 4px;
    font-size: 14px;
  }

  .search-input:focus {
    outline: none;
    border-color: #007bff;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }

  .tabs {
    display: flex;
    background: white;
    border-bottom: 1px solid #dee2e6;
  }

  .tab-button {
    flex: 1;
    padding: 12px 8px;
    border: none;
    background: transparent;
    cursor: pointer;
    font-size: 12px;
    color: #6c757d;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
  }

  .tab-button:hover {
    background: #f8f9fa;
    color: #495057;
  }

  .tab-button.active {
    color: #007bff;
    border-bottom-color: #007bff;
    background: #f8f9fa;
  }

  .content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .items-section h4 {
    margin: 0 0 12px 0;
    color: #495057;
    font-size: 14px;
    font-weight: 600;
  }

  .loading, .empty-state {
    text-align: center;
    color: #6c757d;
    padding: 20px;
    font-style: italic;
  }

  .error-message {
    background: #f8d7da;
    color: #721c24;
    padding: 12px;
    border-radius: 4px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .error-message button {
    background: none;
    border: none;
    color: #721c24;
    cursor: pointer;
    font-size: 18px;
  }

  .items-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .draggable-item {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    padding: 12px;
    cursor: grab;
    transition: all 0.2s;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }

  .draggable-item:hover {
    border-color: #007bff;
    box-shadow: 0 2px 6px rgba(0, 123, 255, 0.15);
    transform: translateY(-1px);
  }

  .draggable-item:active {
    cursor: grabbing;
  }

  .item-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
  }

  .item-icon {
    font-size: 16px;
  }

  .item-name {
    font-weight: 600;
    color: #495057;
    flex: 1;
  }

  .public-badge {
    background: #28a745;
    color: white;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    font-weight: 500;
  }

  .item-description {
    color: #6c757d;
    font-size: 12px;
    margin-bottom: 6px;
    line-height: 1.4;
  }

  .item-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .item-type {
    background: #e9ecef;
    color: #495057;
    font-size: 10px;
    padding: 2px 6px;
    border-radius: 10px;
    font-weight: 500;
  }

  .agent-item {
    border-left: 3px solid #007bff;
  }

  .tool-item {
    border-left: 3px solid #28a745;
  }

  .flow-item {
    border-left: 3px solid #ffc107;
  }
</style>
