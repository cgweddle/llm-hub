<script lang="ts">
  import { Handle, Position, type NodeProps, useUpdateNodeInternals } from '@xyflow/svelte';
  import { onMount, onDestroy } from 'svelte';
  import { EditorView, basicSetup } from 'codemirror';
  import { python } from '@codemirror/lang-python';
  import { oneDark } from '@codemirror/theme-one-dark';
  import { EditorState } from '@codemirror/state';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];
  export let id: string;

  let { name, description, script_code, handles = ['a'], toolId, input_schema, output_schema } = data;

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

  // Update node internals when expansion state changes
  $: if (outputExpanded !== undefined) {
    updateNodeInternals(id);
  }

  // Track custom parameter values
  let parameterValues: { [key: string]: string } = {};
  let editingParameter: string | null = null;
  let tempValue: string = '';

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

  let expanded = false;
  let isEditing = false;
  let isSaving = false;
  let saveMessage = '';
  let saveError = false;

  let editorContainer: HTMLDivElement;
  let editorView: EditorView | null = null;
  let editedCode = script_code || '';

  function initEditor() {
    if (!editorContainer || editorView) return;

    const startState = EditorState.create({
      doc: isEditing ? editedCode : script_code,
      extensions: [
        basicSetup,
        python(),
        oneDark,
        EditorView.lineWrapping,
        EditorState.readOnly.of(!isEditing), // Read-only when not editing
        EditorView.editable.of(isEditing), // Not editable when not editing
        EditorView.updateListener.of((update) => {
          if (update.docChanged && isEditing) {
            editedCode = update.state.doc.toString();
          }
        })
      ]
    });

    editorView = new EditorView({
      state: startState,
      parent: editorContainer
    });
  }

  function destroyEditor() {
    if (editorView) {
      editorView.destroy();
      editorView = null;
    }
  }

  function updateEditorMode() {
    if (!editorView) return;

    // Update the editor's state based on editing mode
    const newState = EditorState.create({
      doc: isEditing ? editedCode : script_code,
      extensions: [
        basicSetup,
        python(),
        oneDark,
        EditorView.lineWrapping,
        EditorState.readOnly.of(!isEditing),
        EditorView.editable.of(isEditing),
        EditorView.updateListener.of((update) => {
          if (update.docChanged && isEditing) {
            editedCode = update.state.doc.toString();
          }
        })
      ]
    });

    editorView.setState(newState);
  }

  // Initialize editor when expanded
  $: if (expanded && editorContainer && !editorView && script_code) {
    initEditor();
  }

  // Update editor mode when switching between edit/view
  $: if (isEditing !== undefined && editorView) {
    updateEditorMode();
  }

  async function handleSave() {
    if (!toolId) {
      saveError = true;
      saveMessage = 'Cannot save: Tool ID not found';
      setTimeout(() => { saveMessage = ''; }, 3000);
      return;
    }

    isSaving = true;
    saveError = false;

    try {
      const response = await fetch(`http://localhost:8000/tools/${toolId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          script_code: editedCode
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to update tool: ${response.statusText}`);
      }

      // Update the local data
      script_code = editedCode;
      data.script_code = editedCode;

      saveError = false;
      saveMessage = 'Saved successfully!';
      isEditing = false;

      setTimeout(() => { saveMessage = ''; }, 3000);
    } catch (error) {
      console.error('Error saving code:', error);
      saveError = true;
      saveMessage = `Error: ${error}`;
      setTimeout(() => { saveMessage = ''; }, 5000);
    } finally {
      isSaving = false;
    }
  }

  function handleCancel() {
    editedCode = script_code || '';
    if (editorView) {
      editorView.dispatch({
        changes: {
          from: 0,
          to: editorView.state.doc.length,
          insert: editedCode
        }
      });
    }
    isEditing = false;
  }

  // Cleanup on component destroy
  onDestroy(() => {
    destroyEditor();
  });

  // Truncate description for collapsed view
  function truncateText(text: string, maxLength: number = 80): string {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  }

  // Prevent SvelteFlow zoom when scrolling in the code/content area
  function handleWheel(event: WheelEvent) {
    event.stopPropagation();
  }
</script>

<div class="toolNode" class:expanded={expanded}>
  <div class="toolNodeBody">
    <div class="node-header">
      <span class="node-title">{name}</span>
      <button
        class="expand-button"
        on:click={() => (expanded = !expanded)}
        on:keydown={(event) => { if (event.key === 'Enter' || event.key === ' ') { expanded = !expanded; } }}
        aria-label={expanded ? 'Collapse node' : 'Expand node'}
      >
        {expanded ? '−' : '+'}
      </button>
    </div>
    <div class="node-content" on:wheel={handleWheel}>
      {#if expanded}
        <div class="expanded-content">
          <div class="section">
            <div class="section-label">Description</div>
            <p class="description-text">{description || 'No description available'}</p>
          </div>

          {#if script_code || isEditing}
            <div class="section">
              <div class="section-header">
                <div class="section-label">Script Code</div>
                <div class="action-buttons">
                  {#if !isEditing}
                    <button class="edit-button" on:click={() => { isEditing = true; editedCode = script_code || ''; }}>
                      Edit
                    </button>
                  {:else}
                    <button class="save-button" on:click={handleSave} disabled={isSaving}>
                      {isSaving ? 'Saving...' : 'Save'}
                    </button>
                    <button class="cancel-button" on:click={handleCancel} disabled={isSaving}>
                      Cancel
                    </button>
                  {/if}
                </div>
              </div>

              {#if saveMessage}
                <div class="save-message" class:error={saveError} class:success={!saveError}>
                  {saveMessage}
                </div>
              {/if}

              <div class="code-container">
                <div bind:this={editorContainer} class="editor-container"></div>
              </div>
            </div>
          {:else}
            <div class="section">
              <div class="no-code">No script code available</div>
            </div>
          {/if}
        </div>
      {:else}
        <div class="collapsed-content">
          <div class="description-preview">
            {truncateText(description)}
          </div>
        </div>
      {/if}
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
        id={`input-${paramName}`}
        style="background: #007acc; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;"
        {isConnectable}
      />
    </div>
  {/each}
{:else}
  <!-- Default single target handle if no input schema -->
  <Handle type="target" position={Position.Left} style="background: #007acc; border: 2px solid black; width: 10px; height: 10px; border-radius: 50%;" {isConnectable} />
{/if}

<!-- Output Handles (Source) -->
{#if outputIsDictionary}
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
      id="output"
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
          id={`output-${prop.key}`}
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
      id="output"
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
    transition: width 0.3s ease;
  }

  .toolNode.expanded {
    width: 85vw;
    max-width: 1400px;
  }

  .toolNodeBody {
    width: 100%;
    min-height: 100px;
    border: 3px solid #007acc;
    position: relative;
    overflow: hidden;
    border-radius: 8px;
    background: #1e1e1e;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    transition: all 0.3s ease;
  }

  .toolNodeBody:hover {
    box-shadow: 0 6px 16px rgba(0, 122, 204, 0.4);
    transform: translateY(-2px);
  }

  .node-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid #2d2d30;
    background: #252526;
  }

  .node-title {
    font-weight: 600;
    color: #cccccc;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
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

  .expanded-content {
    display: flex;
    flex-direction: column;
    max-height: 70vh;
    overflow-y: auto;
  }

  .section {
    padding: 12px 16px;
    border-bottom: 1px solid #2d2d30;
  }

  .section:last-child {
    border-bottom: none;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }

  .section-label {
    font-size: 11px;
    font-weight: 600;
    color: #007acc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .action-buttons {
    display: flex;
    gap: 8px;
  }

  .edit-button,
  .save-button,
  .cancel-button {
    padding: 4px 12px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 11px;
    font-weight: 500;
    transition: all 0.2s ease;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .edit-button {
    background: #007acc;
    color: white;
  }

  .edit-button:hover {
    background: #005a9e;
  }

  .save-button {
    background: #0e7a0d;
    color: white;
  }

  .save-button:hover:not(:disabled) {
    background: #0c6b0c;
  }

  .save-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .cancel-button {
    background: #5a5a5a;
    color: white;
  }

  .cancel-button:hover:not(:disabled) {
    background: #484848;
  }

  .cancel-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .save-message {
    padding: 8px 12px;
    border-radius: 4px;
    font-size: 12px;
    margin-bottom: 8px;
    animation: slideDown 0.3s ease;
  }

  .save-message.success {
    background: #0e7a0d;
    color: white;
  }

  .save-message.error {
    background: #c42b1c;
    color: white;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }

  .description-text {
    font-size: 12px;
    color: #cccccc;
    line-height: 1.5;
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .code-container {
    background: #1e1e1e;
    border-radius: 4px;
    overflow-x: auto;
    max-height: 60vh;
    overflow-y: auto;
    border: 1px solid #2d2d30;
  }

  .editor-container {
    min-height: 200px;
    font-size: 13px;
  }

  .editor-container :global(.cm-editor) {
    height: 100%;
  }

  .editor-container :global(.cm-scroller) {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 13px;
  }

  /* Read-only mode styling - remove cursor */
  .editor-container :global(.cm-editor.cm-focused) {
    outline: none;
  }

  .no-code {
    color: #858585;
    font-size: 12px;
    font-style: italic;
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

  /* Custom scrollbar for VS Code feel */
  .expanded-content::-webkit-scrollbar,
  .code-container::-webkit-scrollbar {
    width: 10px;
  }

  .expanded-content::-webkit-scrollbar-track,
  .code-container::-webkit-scrollbar-track {
    background: #1e1e1e;
  }

  .expanded-content::-webkit-scrollbar-thumb,
  .code-container::-webkit-scrollbar-thumb {
    background: #424242;
    border-radius: 5px;
  }

  .expanded-content::-webkit-scrollbar-thumb:hover,
  .code-container::-webkit-scrollbar-thumb:hover {
    background: #4e4e4e;
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
