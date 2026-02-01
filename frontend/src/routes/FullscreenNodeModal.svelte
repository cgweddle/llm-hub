<script lang="ts">
  import { fullscreenNode } from '$lib/stores/fullscreenNode';
  import { EditorView, basicSetup } from 'codemirror';
  import { python } from '@codemirror/lang-python';
  import { oneDark } from '@codemirror/theme-one-dark';
  import { EditorState } from '@codemirror/state';
  import { onDestroy } from 'svelte';
  import { editToolCodeStream, updateAgent } from '$lib/api';
  import type { LLMProvider } from '$lib/store';
  import type { Tool } from '$lib/api';

  export let llmProviders: LLMProvider[] = [];
  export let allTools: Tool[] = [];
  export let onToolUpdated: ((nodeId: string, updatedData: any) => void) | undefined = undefined;
  export let onAgentUpdated: ((agentId: number, updatedData: any) => void) | undefined = undefined;

  // Resolve tool IDs to tool objects for agent display
  function getToolsByIds(toolIds: number[]): Tool[] {
    return toolIds
      .map(id => allTools.find(t => t.id === id))
      .filter((t): t is Tool => t !== undefined);
  }

  $: nodeData = $fullscreenNode;

  // Agent editing state
  let isEditingAgent = false;
  let isSavingAgent = false;
  let agentSaveMessage = '';
  let agentSaveError = false;
  let editedAgentName = '';
  let editedAgentDescription = '';
  let editedAgentSystemPrompt = '';
  let editedAgentLLMConfig = '';
  let editedAgentToolIds: number[] = [];

  // Initialize agent edit state when entering edit mode
  function startEditingAgent() {
    if (!nodeData) return;
    editedAgentName = nodeData.data.name || '';
    editedAgentDescription = nodeData.data.description || '';
    editedAgentSystemPrompt = nodeData.data.system_prompt || '';
    editedAgentLLMConfig = nodeData.data.llm_config?.model_name || '';
    editedAgentToolIds = [...(nodeData.data.tools_config?.tool_ids || [])];
    isEditingAgent = true;
  }

  function cancelEditingAgent() {
    isEditingAgent = false;
    agentSaveMessage = '';
  }

  function toggleAgentTool(toolId: number) {
    if (editedAgentToolIds.includes(toolId)) {
      editedAgentToolIds = editedAgentToolIds.filter(id => id !== toolId);
    } else {
      editedAgentToolIds = [...editedAgentToolIds, toolId];
    }
  }

  async function handleSaveAgent() {
    if (!nodeData?.data.agentId) {
      agentSaveError = true;
      agentSaveMessage = 'Cannot save: Agent ID not found';
      setTimeout(() => { agentSaveMessage = ''; }, 3000);
      return;
    }

    isSavingAgent = true;
    agentSaveError = false;

    try {
      const updatedAgent = await updateAgent(nodeData.data.agentId, {
        name: editedAgentName,
        description: editedAgentDescription,
        system_prompt: editedAgentSystemPrompt,
        llm_config: { model_name: editedAgentLLMConfig },
        tools_config: { tool_ids: editedAgentToolIds }
      });

      // Update modal's live data
      nodeData.data.name = updatedAgent.name;
      nodeData.data.description = updatedAgent.description || '';
      nodeData.data.system_prompt = updatedAgent.system_prompt || '';
      nodeData.data.llm_config = updatedAgent.llm_config || {};
      nodeData.data.tools_config = updatedAgent.tools_config || {};

      // Notify parent to update sidebar + canvas
      if (onAgentUpdated) {
        onAgentUpdated(nodeData.data.agentId, updatedAgent);
      }

      agentSaveError = false;
      agentSaveMessage = 'Agent saved successfully!';
      isEditingAgent = false;
      setTimeout(() => { agentSaveMessage = ''; }, 3000);
    } catch (error) {
      agentSaveError = true;
      agentSaveMessage = `Error: ${error}`;
      setTimeout(() => { agentSaveMessage = ''; }, 5000);
    } finally {
      isSavingAgent = false;
    }
  }

  // Tool node state
  let isEditing = false;
  let isSaving = false;
  let saveMessage = '';
  let saveError = false;
  let editorContainer: HTMLDivElement;
  let editorView: EditorView | null = null;
  let editedCode = '';
  let mainFunction = '';

  // Edit with AI state
  let showEditWithAI = false;
  let editingInstructions = '';
  let isEditingWithAI = false;
  let selectedLLMProvider: LLMProvider | null = null;

  // Fetch fresh tool data from database (only called after save)
  async function fetchFreshToolData(toolId: number) {
    try {
      const response = await fetch(`http://localhost:8000/tools/${toolId}`);
      if (!response.ok) {
        throw new Error(`Failed to fetch tool: ${response.statusText}`);
      }

      const freshTool = await response.json();

      // Update local modal state
      if (nodeData && nodeData.data) {
        nodeData.data.script_code = freshTool.script_code;
        nodeData.data.main_function = freshTool.main_function;
        nodeData.data.description = freshTool.description;
        nodeData.data.input_schema = freshTool.input_schema;
        nodeData.data.output_schema = freshTool.output_schema;

        mainFunction = freshTool.main_function || '';
        editedCode = freshTool.script_code || '';
      }

      // Notify parent to update canvas node
      if (onToolUpdated && nodeData?.nodeId) {
        onToolUpdated(nodeData.nodeId, {
          script_code: freshTool.script_code,
          main_function: freshTool.main_function,
          description: freshTool.description,
          input_schema: freshTool.input_schema,
          output_schema: freshTool.output_schema,
        });
      }

      return freshTool;
    } catch (error) {
      console.error('Failed to fetch fresh tool data:', error);
      throw error;
    }
  }

  // Initialize main function from node data
  $: mainFunction = nodeData?.data?.main_function || '';

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
      const updateData: { script_code: string; main_function?: string } = {
        script_code: editedCode
      };

      // Include main function if it's changed
      if (mainFunction && mainFunction !== nodeData.data.main_function) {
        updateData.main_function = mainFunction;
      }

      const response = await fetch(`http://localhost:8000/tools/${nodeData.data.toolId}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(updateData)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Failed to update tool: ${response.statusText}`);
      }

      // Fetch fresh data after successful save
      await fetchFreshToolData(nodeData.data.toolId);

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
    mainFunction = nodeData?.data.main_function || '';
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

  /**
   * Edit existing code using AI with streaming
   */
  async function handleEditWithAI() {
    // Validate editing instructions
    if (!editingInstructions.trim()) {
      saveError = true;
      saveMessage = 'Please enter editing instructions';
      setTimeout(() => { saveMessage = ''; }, 3000);
      return;
    }

    // Validate code exists
    if (!editedCode.trim() && !(nodeData?.data.script_code || '').trim()) {
      saveError = true;
      saveMessage = 'No code to edit.';
      setTimeout(() => { saveMessage = ''; }, 3000);
      return;
    }

    // Validate LLM provider is selected
    if (!selectedLLMProvider) {
      saveError = true;
      saveMessage = 'Please select an LLM provider';
      setTimeout(() => { saveMessage = ''; }, 5000);
      return;
    }

    try {
      isEditingWithAI = true;
      saveError = false;

      // Get the current code
      const currentCode = isEditing ? editedCode : (nodeData?.data.script_code || '');

      // Clear existing code and enter edit mode if not already editing
      if (!isEditing) {
        isEditing = true;
        editedCode = '';
      } else {
        editedCode = '';
      }

      if (editorView) {
        const transaction = editorView.state.update({
          changes: {
            from: 0,
            to: editorView.state.doc.length,
            insert: ''
          }
        });
        editorView.dispatch(transaction);
      }

      // Update editor to editable mode
      updateEditorMode();

      // Start streaming edited code
      await editToolCodeStream(
        {
          existing_code: currentCode,
          editing_instructions: editingInstructions.trim(),
          tool_name: nodeData?.data.name || 'tool',
          tool_description: nodeData?.data.description || '',
          provider: selectedLLMProvider.provider,
          model: selectedLLMProvider.model,
          api_key: selectedLLMProvider.apiKey,
          base_url: selectedLLMProvider.baseUrl
        },
        // onChunk: append text to editor as it arrives
        (chunk: string) => {
          editedCode += chunk;
          if (editorView) {
            const transaction = editorView.state.update({
              changes: {
                from: editorView.state.doc.length,
                to: editorView.state.doc.length,
                insert: chunk
              }
            });
            editorView.dispatch(transaction);
          }
        },
        // onDone: update with final cleaned code and main function
        (scriptCode: string, mainFunctionName: string) => {
          // Replace editor content with cleaned code (markdown stripped)
          editedCode = scriptCode;
          if (editorView) {
            const transaction = editorView.state.update({
              changes: {
                from: 0,
                to: editorView.state.doc.length,
                insert: scriptCode
              }
            });
            editorView.dispatch(transaction);
          }

          // Update main function name if provided
          if (mainFunctionName) {
            mainFunction = mainFunctionName;
          }

          // Show success message
          saveError = false;
          saveMessage = 'Code edited successfully! Review the changes.';
          setTimeout(() => { saveMessage = ''; }, 3000);

          isEditingWithAI = false;
          editingInstructions = '';
        },
        // onError: show error message
        (error: string) => {
          saveError = true;
          saveMessage = `Failed to edit code: ${error}`;
          setTimeout(() => { saveMessage = ''; }, 5000);
          isEditingWithAI = false;
        }
      );

    } catch (error) {
      saveError = true;
      saveMessage = `Failed to edit code: ${error}`;
      setTimeout(() => { saveMessage = ''; }, 5000);
      isEditingWithAI = false;
    }
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
          {:else if nodeData.nodeType === 'agent'}
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

          <div class="section">
            <div class="section-label">Main Function</div>
            <input
              type="text"
              bind:value={mainFunction}
              placeholder="e.g. process_data"
              class="main-function-input"
              readonly={!isEditing}
            />
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

            <!-- Edit with AI Section -->
            <div class="section">
              <button
                class="create-tool-expand-header"
                on:click={() => showEditWithAI = !showEditWithAI}
                type="button"
              >
                <span class="expand-icon">{showEditWithAI ? '∨' : '→'}</span>
                <span>Edit with AI</span>
              </button>

              {#if showEditWithAI}
                <div class="create-tool-ai-expanded">
                  <div class="create-tool-ai-field-left">
                    <label class="create-tool-label" for="editInstructions">Editing Instructions</label>
                    <textarea
                      id="editInstructions"
                      bind:value={editingInstructions}
                      placeholder="e.g., Add error handling, convert to async/await, optimize performance..."
                      rows="8"
                      class="create-tool-input create-tool-textarea-full"
                    ></textarea>
                  </div>

                  <div class="create-tool-ai-field-right">
                    <div class="create-tool-ai-field">
                      <label class="create-tool-label" for="editLlmProvider">LLM Provider</label>
                      <select
                        id="editLlmProvider"
                        class="create-tool-input"
                        bind:value={selectedLLMProvider}
                      >
                        <option value={null}>-- Select LLM --</option>
                        {#each llmProviders as provider}
                          <option value={provider}>{provider.name}</option>
                        {/each}
                      </select>
                      {#if llmProviders.length === 0}
                        <div class="create-tool-helper-text" style="color: #f59e0b;">
                          No LLM providers configured. Configure one in the sidebar's "Attach LLM" panel.
                        </div>
                      {/if}
                    </div>

                    <div class="create-tool-ai-field">
                      <button
                        class="edit-ai-button"
                        on:click={handleEditWithAI}
                        disabled={isEditingWithAI}
                        type="button"
                      >
                        {isEditingWithAI ? 'Editing...' : 'Edit with AI'}
                      </button>
                    </div>
                  </div>
                </div>
              {/if}
            </div>
          {:else}
            <div class="section">
              <div class="no-code">No script code available</div>
            </div>
          {/if}
        {:else if nodeData.nodeType === 'agent'}
          <!-- Agent Node Content -->
          <div class="section">
            <div class="section-header">
              <div class="section-label" style="margin-bottom: 0;">Agent Details</div>
              <div class="action-buttons">
                {#if !isEditingAgent}
                  <button class="edit-button" on:click={startEditingAgent}>Edit</button>
                {:else}
                  <button class="save-button" on:click={handleSaveAgent} disabled={isSavingAgent}>
                    {isSavingAgent ? 'Saving...' : 'Save'}
                  </button>
                  <button class="cancel-button" on:click={cancelEditingAgent} disabled={isSavingAgent}>
                    Cancel
                  </button>
                {/if}
              </div>
            </div>

            {#if agentSaveMessage}
              <div class="save-message" class:error={agentSaveError} class:success={!agentSaveError}>
                {agentSaveMessage}
              </div>
            {/if}
          </div>

          <!-- Name -->
          <div class="section">
            <div class="section-label">Name</div>
            {#if isEditingAgent}
              <input type="text" class="main-function-input" bind:value={editedAgentName} placeholder="Agent name" />
            {:else}
              <p class="description-text">{nodeData.data.name}</p>
            {/if}
          </div>

          <!-- Description -->
          <div class="section">
            <div class="section-label">Description</div>
            {#if isEditingAgent}
              <input type="text" class="main-function-input" bind:value={editedAgentDescription} placeholder="Agent description" />
            {:else}
              <p class="description-text">{nodeData.data.description || 'No description available'}</p>
            {/if}
          </div>

          <!-- System Prompt -->
          <div class="section">
            <div class="section-label">System Prompt</div>
            {#if isEditingAgent}
              <textarea
                class="agent-edit-textarea"
                bind:value={editedAgentSystemPrompt}
                placeholder="Enter the system prompt..."
                rows="12"
              ></textarea>
            {:else}
              <pre class="system-prompt-text">{nodeData.data.system_prompt || 'No system prompt configured'}</pre>
            {/if}
          </div>

          <!-- LLM Configuration -->
          <div class="section">
            <div class="section-label">LLM Configuration</div>
            {#if isEditingAgent}
              <select class="main-function-input" bind:value={editedAgentLLMConfig}>
                <option value="">-- Select LLM --</option>
                {#each llmProviders as provider}
                  <option value={provider.name}>{provider.name}</option>
                {/each}
              </select>
            {:else if nodeData.data.llm_config && nodeData.data.llm_config.model_name}
              <p class="description-text">Model: <strong>{nodeData.data.llm_config.model_name}</strong></p>
            {:else}
              <p class="description-text no-data">No LLM configured</p>
            {/if}
          </div>

          <!-- Assigned Tools -->
          <div class="section">
            <div class="section-label">Assigned Tools</div>
            {#if isEditingAgent}
              {#if allTools.length === 0}
                <p class="description-text no-data">No tools available</p>
              {:else}
                <div class="agent-tools-list">
                  {#each allTools as tool}
                    <label class="agent-tool-item agent-tool-editable">
                      <input
                        type="checkbox"
                        checked={editedAgentToolIds.includes(tool.id)}
                        on:change={() => toggleAgentTool(tool.id)}
                      />
                      <span class="agent-tool-name">{tool.name}</span>
                      {#if tool.description}
                        <span class="agent-tool-desc">{tool.description}</span>
                      {/if}
                    </label>
                  {/each}
                </div>
              {/if}
            {:else if nodeData.data.tools_config?.tool_ids?.length > 0}
              {@const resolvedTools = getToolsByIds(nodeData.data.tools_config.tool_ids)}
              <div class="agent-tools-list">
                {#each resolvedTools as tool}
                  <div class="agent-tool-item">
                    <span class="agent-tool-name">{tool.name}</span>
                    {#if tool.description}
                      <span class="agent-tool-desc">{tool.description}</span>
                    {/if}
                  </div>
                {/each}
              </div>
            {:else}
              <p class="description-text no-data">No tools assigned</p>
            {/if}
          </div>

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

  /* Main function input */
  .main-function-input {
    width: 100%;
    padding: 10px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    color: #cccccc;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .main-function-input:focus {
    outline: none;
    border-color: #007acc;
  }

  .main-function-input::placeholder {
    color: #6c6c6c;
  }

  .main-function-input:read-only {
    background: #252526;
    cursor: default;
  }

  /* Edit with AI expandable section - matching Create Tool modal */
  .create-tool-expand-header {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 10px 14px;
    width: 100%;
    color: #cccccc;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  }

  .create-tool-expand-header:hover {
    background: #3e3e42;
    color: #ffffff;
  }

  .expand-icon {
    font-size: 12px;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 16px;
    height: 16px;
  }

  .create-tool-ai-expanded {
    margin-top: 12px;
    padding: 16px;
    background: #1e1e1e;
    border: 1px solid #2d2d30;
    border-radius: 4px;
    display: flex;
    flex-direction: row;
    gap: 16px;
  }

  .create-tool-ai-field-left {
    flex: 1;
    display: flex;
    flex-direction: column;
  }

  .create-tool-ai-field-right {
    width: 280px;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }

  .create-tool-ai-field {
    display: flex;
    flex-direction: column;
  }

  .create-tool-textarea-full {
    flex: 1;
    min-height: 150px;
    resize: vertical;
  }

  .create-tool-label {
    font-size: 13px;
    font-weight: 500;
    color: #cccccc;
    margin-bottom: 8px;
  }

  .create-tool-input {
    width: 100%;
    padding: 10px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    color: #cccccc;
    font-size: 14px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
    resize: vertical;
  }

  .create-tool-input:focus {
    outline: none;
    border-color: #007acc;
  }

  .create-tool-input::placeholder {
    color: #6c6c6c;
  }

  .create-tool-helper-text {
    font-size: 12px;
    color: #858585;
    margin-top: 6px;
  }

  .edit-ai-button {
    padding: 10px 20px;
    background: #0e7a0d;
    border: none;
    border-radius: 4px;
    color: white;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    align-self: flex-start;
  }

  .edit-ai-button:hover:not(:disabled) {
    background: #0c6b0c;
  }

  .edit-ai-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  /* Agent-specific styles */
  .system-prompt-text {
    font-size: 14px;
    color: #cccccc;
    line-height: 1.6;
    margin: 0;
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Consolas', 'Courier New', monospace;
    background: #1e1e1e;
    padding: 16px;
    border-radius: 4px;
    border: 1px solid #2d2d30;
    max-height: 50vh;
    overflow-y: auto;
  }

  .no-data {
    color: #858585;
    font-style: italic;
  }

  .agent-tools-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .agent-tool-item {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 10px 14px;
    background: #1e1e1e;
    border: 1px solid #2d2d30;
    border-radius: 4px;
  }

  .agent-tool-name {
    font-size: 14px;
    font-weight: 600;
    color: #cccccc;
    white-space: nowrap;
  }

  .agent-tool-desc {
    font-size: 13px;
    color: #858585;
  }

  .agent-tool-editable {
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .agent-tool-editable:hover {
    background: #2d2d30;
  }

  .agent-tool-editable input[type="checkbox"] {
    width: 16px;
    height: 16px;
    cursor: pointer;
    accent-color: #007acc;
    flex-shrink: 0;
  }

  .agent-edit-textarea {
    width: 100%;
    min-height: 200px;
    padding: 16px;
    background: #1e1e1e;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    color: #cccccc;
    font-size: 14px;
    font-family: 'Consolas', 'Courier New', monospace;
    line-height: 1.6;
    resize: vertical;
  }

  .agent-edit-textarea:focus {
    outline: none;
    border-color: #007acc;
  }
</style>
