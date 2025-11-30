<script lang="ts">
  import { Handle, Position, type NodeProps } from '@xyflow/svelte';
  import { onMount, onDestroy } from 'svelte';
  import { EditorView, basicSetup } from 'codemirror';
  import { python } from '@codemirror/lang-python';
  import { oneDark } from '@codemirror/theme-one-dark';
  import { EditorState } from '@codemirror/state';

  type $$Props = Omit<NodeProps, 'id'>;
  export let data: $$Props['data'];
  export let isConnectable: $$Props['isConnectable'];

  let { name, description, script_code, handles = ['a'], toolId } = data;
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
  $: {
    // Reference isEditing to make this reactive to its changes
    const editMode = isEditing;
    if (editorView) {
      updateEditorMode();
    }
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

<!-- Target Handle -->
<Handle type="target" position={Position.Left} style="background: #007acc;" {isConnectable} />

<!-- Dynamically Generated Source Handles -->
{#each handles as handleId, index}
  <Handle
    type="source"
    position={Position.Right}
    id={handleId}
    style="top: {index * 20 + 10}px; background: #007acc;"
    {isConnectable}
  />
{/each}

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
</style>
