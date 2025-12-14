<script lang="ts">
  import { fullscreenNode } from '$lib/stores/fullscreenNode';
  import { EditorView, basicSetup } from 'codemirror';
  import { python } from '@codemirror/lang-python';
  import { oneDark } from '@codemirror/theme-one-dark';
  import { EditorState } from '@codemirror/state';
  import { onDestroy } from 'svelte';

  $: nodeData = $fullscreenNode;

  // Tool node state
  let isEditing = false;
  let isSaving = false;
  let saveMessage = '';
  let saveError = false;
  let editorContainer: HTMLDivElement;
  let editorView: EditorView | null = null;
  let editedCode = '';

  // Watch for node data changes and initialize editor
  $: if (nodeData?.nodeType === 'tool' && editorContainer && nodeData.data.script_code) {
    editedCode = nodeData.data.script_code;
    initEditor();
  }

  function initEditor() {
    destroyEditor(); // Clean up any existing editor

    if (!editorContainer) return;

    const startState = EditorState.create({
      doc: isEditing ? editedCode : nodeData?.data.script_code || '',
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
    if (!editorView || !nodeData) return;

    const newState = EditorState.create({
      doc: isEditing ? editedCode : nodeData.data.script_code || '',
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

  $: if (isEditing !== undefined && editorView) {
    updateEditorMode();
  }

  async function handleSave() {
    if (!nodeData?.data.toolId) {
      saveError = true;
      saveMessage = 'Cannot save: Tool ID not found';
      setTimeout(() => { saveMessage = ''; }, 3000);
      return;
    }

    isSaving = true;
    saveError = false;

    try {
      const response = await fetch(`http://localhost:8000/tools/${nodeData.data.toolId}`, {
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

      // Update the node data
      if (nodeData) {
        nodeData.data.script_code = editedCode;
      }

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
    editedCode = nodeData?.data.script_code || '';
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

  function handleClose() {
    destroyEditor();
    isEditing = false;
    saveMessage = '';
    fullscreenNode.close();
  }

  // Cleanup on component destroy
  onDestroy(() => {
    destroyEditor();
  });
</script>

{#if nodeData}
  <div class="modal-overlay" on:click={handleClose}>
    <div class="modal-content" on:click={(e) => e.stopPropagation()}>
      <div class="modal-header">
        <h2 class="modal-title">
          {#if nodeData.nodeType === 'tool'}
            {nodeData.data.name}
          {:else if nodeData.nodeType === 'expandable'}
            {nodeData.data.label}
          {:else if nodeData.nodeType === 'colorSelector'}
            Color Picker
          {/if}
        </h2>
        <button class="close-button" on:click={handleClose}>×</button>
      </div>

      <div class="modal-body">
        {#if nodeData.nodeType === 'tool'}
          <!-- Tool Node Content -->
          <div class="section">
            <div class="section-label">Description</div>
            <p class="description-text">{nodeData.data.description || 'No description available'}</p>
          </div>

          {#if nodeData.data.script_code || isEditing}
            <div class="section">
              <div class="section-header">
                <div class="section-label">Script Code</div>
                <div class="action-buttons">
                  {#if !isEditing}
                    <button class="edit-button" on:click={() => { isEditing = true; editedCode = nodeData.data.script_code || ''; }}>
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
        {:else if nodeData.nodeType === 'expandable'}
          <!-- Expandable Node Content -->
          <div class="section">
            <div class="node-details">
              <p>This is a {nodeData.data.label.toLowerCase()} node.</p>
              <p>You can configure its properties here.</p>
            </div>
            <div class="node-actions">
              <button class="action-button primary">Configure</button>
              <button class="action-button secondary">Settings</button>
            </div>
          </div>
        {:else if nodeData.nodeType === 'colorSelector'}
          <!-- Color Selector Node Content -->
          <div class="section">
            <div class="color-display">
              <span>Color: </span>
              <span class="color-value" style="color: {$nodeData.data.color}">{$nodeData.data.color}</span>
            </div>
            <input
              class="color-input"
              type="color"
              on:input={(event) => {
                if (nodeData?.data.color) {
                  nodeData.data.color.set(event.currentTarget.value);
                }
              }}
              value={$nodeData.data.color}
            />
          </div>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.85);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: fadeIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from {
      opacity: 0;
    }
    to {
      opacity: 1;
    }
  }

  .modal-content {
    background: #1e1e1e;
    border-radius: 8px;
    width: 95vw;
    height: 95vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    animation: slideUp 0.3s ease-out;
  }

  @keyframes slideUp {
    from {
      transform: translateY(50px);
      opacity: 0;
    }
    to {
      transform: translateY(0);
      opacity: 1;
    }
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 24px;
    border-bottom: 2px solid #2d2d30;
    background: #252526;
  }

  .modal-title {
    margin: 0;
    font-size: 24px;
    font-weight: 600;
    color: #cccccc;
  }

  .close-button {
    background: none;
    border: none;
    font-size: 36px;
    color: #cccccc;
    cursor: pointer;
    padding: 0;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .close-button:hover {
    background-color: #3e3e42;
    color: #ffffff;
  }

  .modal-body {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  .section {
    margin-bottom: 24px;
    padding: 20px;
    background: #252526;
    border-radius: 8px;
    border: 1px solid #2d2d30;
  }

  .section:last-child {
    margin-bottom: 0;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  .section-label {
    font-size: 13px;
    font-weight: 600;
    color: #007acc;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
  }

  .description-text {
    font-size: 14px;
    color: #cccccc;
    line-height: 1.6;
    margin: 0;
  }

  .action-buttons {
    display: flex;
    gap: 12px;
  }

  .edit-button,
  .save-button,
  .cancel-button {
    padding: 8px 16px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.2s ease;
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
    padding: 12px 16px;
    border-radius: 4px;
    font-size: 14px;
    margin-bottom: 12px;
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

  .code-container {
    background: #1e1e1e;
    border-radius: 4px;
    overflow: auto;
    max-height: 60vh;
    border: 1px solid #2d2d30;
  }

  .editor-container {
    min-height: 400px;
    font-size: 14px;
  }

  .editor-container :global(.cm-editor) {
    height: 100%;
  }

  .editor-container :global(.cm-scroller) {
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 14px;
  }

  .no-code {
    color: #858585;
    font-size: 14px;
    font-style: italic;
  }

  .node-details {
    font-size: 14px;
    color: #cccccc;
    line-height: 1.6;
    margin-bottom: 20px;
  }

  .node-details p {
    margin: 0 0 12px 0;
  }

  .node-actions {
    display: flex;
    gap: 12px;
  }

  .action-button {
    padding: 10px 20px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.2s ease;
  }

  .action-button.primary {
    background: #007bff;
    color: white;
  }

  .action-button.primary:hover {
    background: #0056b3;
  }

  .action-button.secondary {
    background: #6c757d;
    color: white;
  }

  .action-button.secondary:hover {
    background: #545b62;
  }

  .color-display {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 16px;
    color: #cccccc;
    margin-bottom: 20px;
  }

  .color-value {
    font-weight: 600;
    font-family: 'Courier New', monospace;
    font-size: 18px;
  }

  .color-input {
    width: 100%;
    max-width: 400px;
    height: 60px;
    border: 2px solid #2d2d30;
    border-radius: 8px;
    cursor: pointer;
    transition: border-color 0.2s ease;
  }

  .color-input:hover {
    border-color: #007acc;
  }

  /* Custom scrollbar */
  .modal-body::-webkit-scrollbar,
  .code-container::-webkit-scrollbar {
    width: 12px;
  }

  .modal-body::-webkit-scrollbar-track,
  .code-container::-webkit-scrollbar-track {
    background: #1e1e1e;
  }

  .modal-body::-webkit-scrollbar-thumb,
  .code-container::-webkit-scrollbar-thumb {
    background: #424242;
    border-radius: 6px;
  }

  .modal-body::-webkit-scrollbar-thumb:hover,
  .code-container::-webkit-scrollbar-thumb:hover {
    background: #4e4e4e;
  }
</style>
