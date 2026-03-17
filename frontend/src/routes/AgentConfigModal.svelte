<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Button } from "$lib/components/ui/button";
  import { getAgentTemplate, type AgentTypeKey } from '$lib/agentTemplates';
  import type { LLMProvider } from '$lib/store';
  import type { Tool } from '$lib/api';
  import { generateUserPromptStream, type SystemPromptGenerateRequest } from '$lib/api';

  const dispatch = createEventDispatcher<{
    save: {
      nodeId: string;
      name: string;
      systemPrompt: string;
      userPrompt: string;
      assignedTools: number[];
      llmProvider: string;
    };
    close: void;
  }>();

  // Props
  export let open = false;
  export let nodeId = '';
  export let agentType: AgentTypeKey = 'react';
  export let initialName = '';
  export let initialSystemPrompt = '';
  export let initialUserPrompt = '';
  export let initialAssignedTools: number[] = [];
  export let initialLLMProvider = '';
  export let tools: Tool[] = [];
  export let llmProviders: LLMProvider[] = [];
  export let selectedLLMForGeneration: LLMProvider | null = null;

  // Local state
  let name = initialName;
  let systemPrompt = initialSystemPrompt;
  let userPrompt = initialUserPrompt;
  let assignedTools = [...initialAssignedTools];
  let llmProvider = initialLLMProvider;

  // AI generation state
  let showGenerateAI = false;
  let additionalInstructions = '';
  let isGenerating = false;
  let generationError = '';

  // Reset state when modal opens with new data
  $: if (open) {
    name = initialName;
    systemPrompt = initialSystemPrompt;
    userPrompt = initialUserPrompt;
    assignedTools = [...initialAssignedTools];
    llmProvider = initialLLMProvider;
    showGenerateAI = false;
    additionalInstructions = '';
    isGenerating = false;
    generationError = '';
  }

  $: template = getAgentTemplate(agentType);

  function toggleTool(toolId: number) {
    if (assignedTools.includes(toolId)) {
      assignedTools = assignedTools.filter(id => id !== toolId);
    } else {
      assignedTools = [...assignedTools, toolId];
    }
  }

  function handleSave() {
    dispatch('save', {
      nodeId,
      name: name.trim() || template.name,
      systemPrompt: systemPrompt.trim() || template.defaultSystemPrompt,
      userPrompt: userPrompt.trim(),
      assignedTools,
      llmProvider
    });
  }

  function handleClose() {
    dispatch('close');
  }

  function handleOverlayClick(event: MouseEvent) {
    if (event.target === event.currentTarget) {
      handleClose();
    }
  }

  async function handleGeneratePrompt() {
    if (!selectedLLMForGeneration) {
      generationError = 'Please select an LLM provider for generation';
      return;
    }

    generationError = '';
    isGenerating = true;
    userPrompt = '';

    // Get selected tool names
    const selectedToolNames = tools
      .filter(t => assignedTools.includes(t.id))
      .map(t => t.name);

    try {
      await generateUserPromptStream(
        {
          agent_name: name.trim() || template.name,
          agent_description: `${template.name}: ${template.description}`,
          tool_names: selectedToolNames,
          model: selectedLLMForGeneration.name,
          additional_instructions: additionalInstructions.trim() || undefined
        },
        // onChunk
        (chunk: string) => {
          userPrompt += chunk;
        },
        // onDone
        (finalPrompt: string) => {
          userPrompt = finalPrompt;
          isGenerating = false;
        },
        // onError
        (error: string) => {
          generationError = error;
          isGenerating = false;
        }
      );
    } catch (error) {
      generationError = `Generation failed: ${error}`;
      isGenerating = false;
    }
  }

  function useDefaultPrompt() {
    systemPrompt = template.defaultSystemPrompt;
  }
</script>

{#if open}
  <div class="modal-overlay" on:click={handleOverlayClick} on:keydown={(e) => e.key === 'Escape' && handleClose()} role="dialog" aria-modal="true" tabindex="-1">
    <div class="modal-content" style="--agent-color: {template.color};">
      <div class="modal-header">
        <div class="header-title">
          <span class="header-icon">{template.icon}</span>
          <h2>Configure {template.name}</h2>
        </div>
        <button class="close-button" on:click={handleClose}>×</button>
      </div>

      <div class="modal-body">
        <!-- Name Section -->
        <div class="config-section">
          <label class="section-label" for="agent-name">Agent Name</label>
          <input
            id="agent-name"
            class="config-input"
            bind:value={name}
            placeholder={template.name}
          />
        </div>

        <!-- System Prompt Section -->
        <div class="config-section">
          <div class="section-header">
            <label class="section-label" for="system-prompt">System Prompt</label>
            <button class="use-default-btn" on:click={useDefaultPrompt} type="button">
              Use Default
            </button>
          </div>
          <textarea
            id="system-prompt"
            class="config-textarea"
            bind:value={systemPrompt}
            placeholder={template.defaultSystemPrompt}
            rows="8"
          ></textarea>
        </div>

        <!-- User Prompt Section -->
        <div class="config-section">
          <label class="section-label" for="user-prompt">User Prompt</label>
          <div class="helper-text" style="margin-bottom: 4px;">
            A task-specific instruction for this agent. Gets prepended to runtime input.
          </div>
          <textarea
            id="user-prompt"
            class="config-textarea small"
            bind:value={userPrompt}
            placeholder="What specific task should this agent perform?"
            rows="4"
          ></textarea>
        </div>

        <!-- Generate User Prompt with AI Section -->
        <div class="config-section">
          <button
            class="expand-header"
            on:click={() => showGenerateAI = !showGenerateAI}
            type="button"
          >
            <span class="expand-icon">{showGenerateAI ? '∨' : '→'}</span>
            <span>Generate User Prompt with AI</span>
          </button>

          {#if showGenerateAI}
            <div class="ai-expanded">
              <div class="ai-field">
                <label class="field-label" for="additional-instructions">Additional Instructions (optional)</label>
                <textarea
                  id="additional-instructions"
                  class="config-textarea small"
                  bind:value={additionalInstructions}
                  placeholder="Any specific requirements for the user prompt..."
                  rows="3"
                ></textarea>
              </div>

              <div class="ai-field">
                <label class="field-label" for="generation-llm">LLM for Generation</label>
                <select
                  id="generation-llm"
                  class="config-input"
                  bind:value={selectedLLMForGeneration}
                >
                  <option value={null}>-- Select LLM --</option>
                  {#each llmProviders as provider}
                    <option value={provider}>{provider.name}</option>
                  {/each}
                </select>
              </div>

              {#if generationError}
                <div class="error-message">{generationError}</div>
              {/if}

              <Button
                onclick={handleGeneratePrompt}
                disabled={isGenerating || !selectedLLMForGeneration}
                class="bg-purple-600 hover:bg-purple-700"
              >
                {#snippet children()}
                  {isGenerating ? 'Generating...' : 'Generate User Prompt'}
                {/snippet}
              </Button>
            </div>
          {/if}
        </div>

        <!-- LLM Provider Section -->
        <div class="config-section">
          <label class="section-label" for="llm-provider">LLM Provider (for agent execution)</label>
          <select
            id="llm-provider"
            class="config-input"
            bind:value={llmProvider}
          >
            <option value="">-- Select LLM --</option>
            {#each llmProviders as provider}
              <option value={provider.name}>{provider.name} ({provider.model})</option>
            {/each}
          </select>
          <div class="helper-text">This LLM will be used when the agent executes tasks.</div>
        </div>

        <!-- Tools Section -->
        <div class="config-section">
          <label class="section-label">Assign Tools</label>
          <div class="helper-text" style="margin-bottom: 10px;">
            Select tools this agent can use during execution.
          </div>

          {#if tools.length === 0}
            <div class="empty-message">No tools available. Create tools first.</div>
          {:else}
            <div class="tools-list">
              {#each tools as tool}
                <label class="tool-item">
                  <input
                    type="checkbox"
                    checked={assignedTools.includes(tool.id)}
                    on:change={() => toggleTool(tool.id)}
                  />
                  <div class="tool-info">
                    <span class="tool-name">{tool.name}</span>
                    {#if tool.description}
                      <span class="tool-desc">{tool.description}</span>
                    {/if}
                  </div>
                </label>
              {/each}
            </div>
          {/if}
        </div>
      </div>

      <div class="modal-footer">
        <Button variant="outline" onclick={handleClose}>
          {#snippet children()}Cancel{/snippet}
        </Button>
        <Button onclick={handleSave} style="background: {template.color};">
          {#snippet children()}Save Configuration{/snippet}
        </Button>
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
    z-index: 10000;
    animation: fadeIn 0.2s ease-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .modal-content {
    background: #1e1e1e;
    border-radius: 12px;
    width: 600px;
    max-width: 95vw;
    max-height: 90vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6);
    border: 2px solid var(--agent-color);
    animation: slideUp 0.3s ease-out;
  }

  @keyframes slideUp {
    from {
      transform: translateY(30px);
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
    padding: 16px 20px;
    background: var(--agent-color);
    color: white;
  }

  .header-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .header-icon {
    font-size: 24px;
  }

  .header-title h2 {
    margin: 0;
    font-size: 18px;
    font-weight: 600;
  }

  .close-button {
    background: rgba(255, 255, 255, 0.2);
    border: none;
    color: white;
    font-size: 28px;
    cursor: pointer;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    transition: background 0.2s;
  }

  .close-button:hover {
    background: rgba(255, 255, 255, 0.35);
  }

  .modal-body {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 20px;
  }

  .config-section {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .section-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--agent-color);
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .use-default-btn {
    background: #2d2d30;
    border: 1px solid #3e3e42;
    color: #b0b0b0;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 11px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .use-default-btn:hover {
    background: #3e3e42;
    color: #ffffff;
  }

  .config-input {
    width: 100%;
    background: #2d2d30;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .config-input:focus {
    border-color: var(--agent-color);
  }

  .config-textarea {
    width: 100%;
    background: #2d2d30;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 14px;
    outline: none;
    resize: vertical;
    min-height: 120px;
    transition: border-color 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .config-textarea.small {
    min-height: 60px;
  }

  .config-textarea:focus {
    border-color: var(--agent-color);
  }

  .helper-text {
    font-size: 12px;
    color: #888888;
  }

  /* Expand/collapse section */
  .expand-header {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    padding: 10px 14px;
    width: 100%;
    color: #cccccc;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .expand-header:hover {
    background: #3e3e42;
    color: #ffffff;
  }

  .expand-icon {
    font-size: 12px;
    font-weight: bold;
  }

  .ai-expanded {
    margin-top: 12px;
    padding: 16px;
    background: #252526;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .ai-field {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .field-label {
    font-size: 12px;
    font-weight: 500;
    color: #b0b0b0;
  }

  .error-message {
    padding: 8px 12px;
    background: #5c2020;
    border: 1px solid #7f2d2d;
    border-radius: 4px;
    color: #fca5a5;
    font-size: 12px;
  }

  /* Tools list */
  .tools-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 200px;
    overflow-y: auto;
  }

  .tool-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 12px;
    background: #2d2d30;
    border: 1px solid #3e3e42;
    border-radius: 6px;
    cursor: pointer;
    transition: background 0.2s;
  }

  .tool-item:hover {
    background: #3e3e42;
  }

  .tool-item input[type="checkbox"] {
    margin-top: 2px;
    width: 16px;
    height: 16px;
    accent-color: var(--agent-color);
  }

  .tool-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .tool-name {
    font-size: 13px;
    font-weight: 500;
    color: #cccccc;
  }

  .tool-desc {
    font-size: 11px;
    color: #888888;
  }

  .empty-message {
    padding: 16px;
    text-align: center;
    color: #707070;
    font-style: italic;
  }

  .modal-footer {
    display: flex;
    justify-content: flex-end;
    gap: 12px;
    padding: 16px 20px;
    border-top: 1px solid #2d2d30;
    background: #252526;
  }

  /* Scrollbar styling */
  .modal-body::-webkit-scrollbar,
  .tools-list::-webkit-scrollbar {
    width: 8px;
  }

  .modal-body::-webkit-scrollbar-track,
  .tools-list::-webkit-scrollbar-track {
    background: #1e1e1e;
  }

  .modal-body::-webkit-scrollbar-thumb,
  .tools-list::-webkit-scrollbar-thumb {
    background: #424242;
    border-radius: 4px;
  }

  .modal-body::-webkit-scrollbar-thumb:hover,
  .tools-list::-webkit-scrollbar-thumb:hover {
    background: #4e4e4e;
  }
</style>
