<script lang="ts">
  import { Handle, Position, type NodeProps, useUpdateNodeInternals } from '@xyflow/svelte';
  import { fullscreenNode } from '$lib/stores/fullscreenNode';
  import type { LLMProvider } from '$lib/store';
  import { getContext } from 'svelte';
  import type { Writable } from 'svelte/store';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];
  export let id: string;

  // Reactive destructuring so variables update when data prop changes
  let name: string, description: string, script_code: string, main_function: string;
  let handles: string[], toolId: number, input_schema: any, output_schema: any, runtimeLLM: any;
  let output_paths: Record<string, string> | undefined;
  $: ({ name, description, script_code, main_function, handles = ['a'], toolId, input_schema, output_schema, runtimeLLM, output_paths } = data);

  // Get LLM providers from parent context (passed from +page.svelte)
  const llmProvidersStore = getContext<Writable<LLMProvider[]>>('llmProviders');
  $: llmProviders = llmProvidersStore ? $llmProvidersStore : [];

  // Get the function to update node internals when handles change
  const updateNodeInternals = useUpdateNodeInternals();

  // Extract input parameter names from the input schema
  let inputParameters: string[] = [];
  $: {
    if (input_schema && typeof input_schema === 'object') {
      // Assuming input_schema has a "properties" field like JSON Schema
      if (input_schema.properties) {
        inputParameters = Object.keys(input_schema.properties);
      } else {
        // Fallback: use all keys from input_schema
        inputParameters = Object.keys(input_schema);
      }
    } else {
      inputParameters = [];
    }
  }

  // Extract output properties if the output is a dictionary
  let outputIsDictionary = false;
  let outputProperties: Array<{key: string, type: string}> = [];
  let outputExpanded = false;

  $: {
    if (output_schema && typeof output_schema === 'object' && output_schema.properties) {
      // Has properties defined - treat as expandable dictionary
      outputIsDictionary = true;
      outputProperties = Object.entries(output_schema.properties).map(([key, value]: [string, any]) => ({
        key,
        type: value.type || 'any'
      }));
    } else {
      outputIsDictionary = false;
      outputProperties = [];
    }
  }

  // Compute output path entries for agent nodes with conditional routing
  let outputPathEntries: Array<[string, string]> = [];
  $: {
    if (data.isAgent && output_paths && typeof output_paths === 'object') {
      outputPathEntries = Object.entries(output_paths);
    } else {
      outputPathEntries = [];
    }
  }

  // Update node internals when expansion state or output paths change
  $: if (outputExpanded !== undefined || outputPathEntries.length >= 0) {
    updateNodeInternals(id);
  }

  // Track custom parameter values
  let parameterValues: { [key: string]: string } = data.parameterValues || {};
  let editingParameter: string | null = null;
  let tempValue: string = '';

  // Update node data when parameterValues change
  $: if (data) {
    data.parameterValues = parameterValues;
  }

  function handleParameterClick(paramName: string) {
    editingParameter = paramName;
    tempValue = parameterValues[paramName] || '';
  }

  function saveParameterValue(paramName: string) {
    if (tempValue.trim()) {
      parameterValues[paramName] = tempValue;
    } else {
      delete parameterValues[paramName];
    }
    editingParameter = null;
    tempValue = '';
  }

  function cancelEdit() {
    editingParameter = null;
    tempValue = '';
  }

  function openFullscreen() {
    if (data.isAgent) {
      fullscreenNode.open({
        nodeId: id,
        nodeType: 'agent',
        data: {
          name,
          description,
          agentId: data.agentId,
          system_prompt: data.system_prompt || '',
          llm_provider: data.llm_provider || '',
          tool_ids: data.tool_ids || [],
          graph_config: data.graph_config || {},
          output_paths: data.output_paths || {},
          output_schema,
          id
        }
      });
    } else {
      fullscreenNode.open({
        nodeId: id,
        nodeType: 'tool',
        data: {
          name,
          description,
          script_code,
          main_function,
          toolId,
          input_schema,
          output_schema,
          id
        }
      });
    }
  }

  // Truncate description for collapsed view
  function truncateText(text: string, maxLength: number = 80): string {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }

  // LLM configuration
  let showLLMDropdown = false;
  let selectedLLM: LLMProvider | null = runtimeLLM || null;

  // Keep selectedLLM in sync with runtimeLLM (when flow is loaded)
  $: selectedLLM = runtimeLLM || null;

  function toggleLLMDropdown(event: Event) {
    event.stopPropagation();
    showLLMDropdown = !showLLMDropdown;
  }

  function selectLLM(provider: LLMProvider | null, event: Event) {
    event.stopPropagation();
    selectedLLM = provider;
    data.runtimeLLM = provider;
    showLLMDropdown = false;
  }

  function closeLLMDropdown() {
    showLLMDropdown = false;
  }

  // Close dropdown when clicking outside
  function handleOutsideClick(event: MouseEvent) {
    if (showLLMDropdown) {
      const target = event.target as HTMLElement;
      if (!target.closest('.llm-config-container')) {
        closeLLMDropdown();
      }
    }
  }
</script>

<svelte:window on:click={handleOutsideClick} />

<div class="toolNode">
  <div class="toolNodeBody">
    <div class="node-header">
      <span class="node-title">{name}</span>
      <div class="node-actions">
        <!-- LLM Configuration Button -->
        <div class="llm-config-container">
          <button
            class="llm-button {selectedLLM ? 'has-llm' : ''}"
            on:click={toggleLLMDropdown}
            on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { toggleLLMDropdown(event); } }}
            aria-label="Configure LLM"
            title={selectedLLM ? `LLM: ${selectedLLM.name}` : 'Attach LLM'}
          >
            🤖
          </button>

          {#if showLLMDropdown}
            <div class="llm-dropdown">
              <div class="llm-dropdown-header">Attach LLM</div>
              <button
                class="llm-dropdown-item {!selectedLLM ? 'selected' : ''}"
                on:click={(e) => selectLLM(null, e)}
              >
                <span class="llm-item-icon">⭘</span>
                <span>None</span>
              </button>
              {#each llmProviders as provider}
                <button
                  class="llm-dropdown-item {selectedLLM?.name === provider.name ? 'selected' : ''}"
                  on:click={(e) => selectLLM(provider, e)}
                >
                  <span class="llm-item-icon">🤖</span>
                  <div class="llm-item-info">
                    <span class="llm-item-name">{provider.name}</span>
                    <span class="llm-item-model">{provider.model}</span>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        </div>

        <button
          class="expand-button"
          on:click={openFullscreen}
          on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { openFullscreen(); } }}
          aria-label="Expand node fullscreen"
        >
          +
        </button>
      </div>
    </div>
    <div class="node-content">
      <div class="collapsed-content">
        <div class="description-preview">
          {truncateText(description)}
        </div>
      </div>
    </div>
  </div>
</div>

<!-- Input Handles (Target) - one for each input parameter -->
{#if inputParameters.length > 0}
  {#each inputParameters as paramName, index}
    <div class="input-handle-wrapper" style="top: {60 + index * 35}px;">
      {#if editingParameter === paramName}
        <div class="parameter-input-container">
          <input
            type="text"
            class="parameter-input"
            bind:value={tempValue}
            on:keydown={(e) => {
              if (e.key === 'Enter') saveParameterValue(paramName);
              if (e.key === 'Escape') cancelEdit();
            }}
            on:blur={() => saveParameterValue(paramName)}
            placeholder="Enter value..."
            autofocus
          />
        </div>
      {:else}
        <button
          class="handle-label-outside"
          on:click|stopPropagation={() => handleParameterClick(paramName)}
          title="Click to set custom value"
        >
          <div class="param-content">
            <span class="param-name">{paramName}</span>
            {#if parameterValues[paramName]}
              <span class="param-value">{parameterValues[paramName]}</span>
            {/if}
          </div>
        </button>
      {/if}
      <Handle
        type="target"
        position={Position.Left}
        id={paramName}
        style="background: #007acc; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
        {isConnectable}
      />
    </div>
  {/each}
{:else}
  <!-- Default single target handle if no input schema -->
  <Handle type="target" position={Position.Left} id="" style="background: #007acc; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;" {isConnectable} />
{/if}

<!-- Output Handles (Source) -->
{#if outputPathEntries.length > 0}
  <!-- Agent nodes with output paths: one handle per path -->
  {#each outputPathEntries as [pathName, pathDescription], index}
    <div class="output-handle-wrapper" style="top: {60 + index * 35}px;">
      <span class="handle-label-outside output-label" title={pathDescription}>
        {pathName}
      </span>
      <Handle
        type="source"
        position={Position.Right}
        id={pathName}
        style="background: #0e7a0d; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
        {isConnectable}
      />
    </div>
  {/each}
{:else if outputIsDictionary}
  <!-- Dictionary output with expand/collapse -->
  <div class="output-handle-wrapper" style="top: 60px;">
    <button
      class="handle-label-outside output-label"
      on:click|stopPropagation={() => outputExpanded = !outputExpanded}
      title={outputExpanded ? 'Click to collapse' : 'Click to expand properties'}
    >
      <span class="expander-icon">{outputExpanded ? '∨' : '→'}</span>
      <span>Output</span>
    </button>
    <Handle
      type="source"
      position={Position.Right}
      id=""
      style="background: #0e7a0d; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
      isConnectable={!outputExpanded && isConnectable}
    />
  </div>

  {#if outputExpanded}
    {#each outputProperties as prop, index}
      <div class="output-handle-wrapper" style="top: {95 + index * 35}px;">
        <span class="handle-label-outside output-label property-label">
          {prop.key}
          <span class="property-type">({prop.type})</span>
        </span>
        <Handle
          type="source"
          position={Position.Right}
          id={prop.key}
          style="background: #0e7a0d; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
          {isConnectable}
        />
      </div>
    {/each}
  {/if}
{:else}
  <!-- Single output handle for non-dictionary outputs -->
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
{/if}

<style>
  .toolNode {
    width: 280px;
    min-height: 100px;
    position: relative;
    cursor: move;
  }

  .toolNodeBody {
    width: 100%;
    min-height: 100px;
    border: 3px solid #007acc;
    position: relative;
    overflow: visible;
    border-radius: 8px;
    background: #1e1e1e;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
  }

  .toolNodeBody:hover {
    box-shadow: 0 6px 16px rgba(0, 122, 204, 0.4);
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2d2d30;
    background: #252526;
    overflow: visible;
  }

  .node-title {
    font-weight: 600;
    color: #cccccc;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .node-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  /* LLM Configuration Button */
  .llm-config-container {
    position: relative;
  }

  .llm-button {
    width: 24px;
    height: 24px;
    border: none;
    background: #6b6b6b;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    transition: all 0.2s ease;
    line-height: 1;
  }

  .llm-button:hover {
    background: #8b8b8b;
    transform: scale(1.05);
  }

  .llm-button.has-llm {
    background: #9333ea;
    box-shadow: 0 0 8px rgba(147, 51, 234, 0.5);
  }

  .llm-button.has-llm:hover {
    background: #7e22ce;
  }

  .llm-button:active {
    transform: scale(0.95);
  }

  /* LLM Dropdown */
  .llm-dropdown {
    position: absolute;
    top: calc(100% + 8px);
    right: 0;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8);
    min-width: 220px;
    max-width: 280px;
    z-index: 10000;
    animation: slideDown 0.2s ease-out;
    max-height: 300px;
    overflow-y: auto;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .llm-dropdown-header {
    padding: 8px 12px;
    background: #252526;
    border-bottom: 1px solid #3e3e42;
    font-size: 11px;
    font-weight: 600;
    color: #9333ea;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .llm-dropdown-item {
    width: 100%;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    background: transparent;
    border: none;
    color: #cccccc;
    font-size: 13px;
    cursor: pointer;
    transition: background 0.15s ease;
    text-align: left;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .llm-dropdown-item:hover {
    background: #3e3e42;
  }

  .llm-dropdown-item.selected {
    background: #9333ea;
    color: white;
  }

  .llm-dropdown-item.selected:hover {
    background: #7e22ce;
  }

  .llm-item-icon {
    font-size: 14px;
    width: 18px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .llm-item-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    flex: 1;
    min-width: 0;
  }

  .llm-item-name {
    font-weight: 500;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .llm-item-model {
    font-size: 11px;
    color: #a0a0a0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .llm-dropdown-item.selected .llm-item-model {
    color: #e0e0e0;
  }

  .expand-button {
    width: 24px;
    height: 24px;
    border: none;
    background: #007acc;
    color: white;
    border-radius: 4px;
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
    background: #005a9e;
    transform: scale(1.05);
  }

  .expand-button:active {
    transform: scale(0.95);
  }

  .node-content {
    padding: 0;
  }

  .collapsed-content {
    padding: 12px 16px;
  }

  .description-preview {
    font-size: 12px;
    color: #cccccc;
    line-height: 1.4;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .expand-button {
    user-select: none;
    -webkit-user-select: none;
  }

  /* Input handle wrapper and labels */
  .input-handle-wrapper {
    position: absolute;
    left: 0;
    transform: translateX(-100%);
    display: flex;
    align-items: center;
    height: 20px;
    pointer-events: all;
  }

  .handle-label-outside {
    font-size: 11px;
    font-weight: 600;
    color: #ffffff;
    background: #007acc;
    padding: 4px 10px;
    border-radius: 4px 0 0 4px;
    cursor: pointer;
    box-shadow: 0 2px 6px rgba(0, 122, 204, 0.3);
    transition: all 0.2s ease;
    border-top: none;
    border-bottom: none;
    border-left: none;
    border-right: 3px solid #007acc;
    display: flex;
    align-items: flex-start;
  }

  .handle-label-outside:hover {
    background: #005a9e;
    box-shadow: 0 3px 8px rgba(0, 122, 204, 0.5);
  }

  .param-content {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-start;
  }

  .param-name {
    white-space: nowrap;
    font-weight: 600;
  }

  .param-value {
    font-size: 10px;
    font-weight: 400;
    color: #e0e0e0;
    white-space: nowrap;
    font-style: italic;
  }

  .parameter-input-container {
    display: flex;
    align-items: center;
  }

  .parameter-input {
    font-size: 11px;
    padding: 4px 8px;
    border: 2px solid #007acc;
    border-radius: 4px 0 0 4px;
    background: #1e1e1e;
    color: #ffffff;
    outline: none;
    min-width: 120px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .parameter-input:focus {
    border-color: #005a9e;
    box-shadow: 0 0 0 2px rgba(0, 122, 204, 0.3);
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

  .expander-icon {
    font-size: 10px;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
  }

  .property-label {
    background: #0e7a0d;
    border-left-color: #0e7a0d;
    padding-left: 20px;
    gap: 4px;
  }

  .property-type {
    font-size: 9px;
    font-weight: 400;
    color: #d0d0d0;
    font-style: italic;
  }
</style>
