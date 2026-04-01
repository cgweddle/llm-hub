<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import {
    fetchEvaluations,
    createEvaluation,
    updateEvaluation,
    deleteEvaluation,
    loadLLMProvidersConfig,
    type Evaluation,
    type EvaluationCategory,
  } from '$lib/api';

  interface Props {
    userId: number;
    onchange?: (evaluations: Evaluation[]) => void;
  }

  let { userId, onchange }: Props = $props();

  let evaluations: Evaluation[] = $state([]);
  let showModal = $state(false);
  let showManager = $state(false);
  let editingEvaluation: Evaluation | null = $state(null);
  let saving = $state(false);
  let llmProviderNames: string[] = $state([]);
  let formLlmProvider = $state('');       // Model used to run the evaluation
  let formGenerationModel = $state('');   // Model used to generate the judge prompt

  // Form state
  let formName = $state('');
  let formDescription = $state('');
  let formJudgeSystemPrompt = $state('');
  let formJudgeUserPrompt = $state('');
  let formScoringRubric = $state('');
  let formScoreType = $state<'NUMERIC' | 'CATEGORICAL' | 'BOOLEAN'>('NUMERIC');
  let formCategories: EvaluationCategory[] = $state([]);
  let formInputVariables: string[] = $state(['output']);
  let formReturnFields: string[] = $state([]);
  let formIsPublic = $state(false);

  // AI generation state
  let isGeneratingPrompt = $state(false);

  async function loadEvaluations() {
    evaluations = await fetchEvaluations(userId);
    onchange?.(evaluations);
  }

  async function loadProviders() {
    try {
      const config = await loadLLMProvidersConfig();
      llmProviderNames = (config.models || []).map((m: any) => m.name);
      if (llmProviderNames.length > 0) {
        if (!formLlmProvider) formLlmProvider = llmProviderNames[0];
        if (!formGenerationModel) formGenerationModel = llmProviderNames[0];
      }
    } catch (e) {
      console.error('Failed to load LLM providers:', e);
    }
  }

  onMount(() => {
    loadEvaluations();
    loadProviders();
  });

  function openNewModal() {
    editingEvaluation = null;
    formName = '';
    formDescription = '';
    formJudgeSystemPrompt = '';
    formJudgeUserPrompt = '';
    formScoringRubric = '';
    formScoreType = 'NUMERIC';
    formCategories = [];
    formInputVariables = ['output'];
    formReturnFields = [];
    formIsPublic = false;
    formLlmProvider = llmProviderNames[0] || '';
    formGenerationModel = llmProviderNames[0] || '';
    rebuildUserPrompt();
    showModal = true;
  }

  function openEditModal(evaluation: Evaluation) {
    editingEvaluation = evaluation;
    formName = evaluation.name;
    formDescription = evaluation.description || '';
    formJudgeSystemPrompt = evaluation.judge_system_prompt || '';
    formJudgeUserPrompt = evaluation.judge_user_prompt || '';
    formScoringRubric = evaluation.scoring_rubric || '';
    formScoreType = evaluation.score_type;
    formCategories = (evaluation.score_categories || []).map(c =>
      typeof c === 'string' ? { name: c } : c
    );
    formLlmProvider = evaluation.llm_provider;
    formInputVariables = evaluation.input_variables || ['output'];
    formReturnFields = evaluation.return_fields || [];
    formIsPublic = evaluation.is_public;
    showModal = true;
  }

  function closeModal() {
    showModal = false;
    editingEvaluation = null;
  }

  // Category management
  function addCategory() {
    formCategories = [...formCategories, { name: '', description: '' }];
  }

  function removeCategory(index: number) {
    formCategories = formCategories.filter((_, i) => i !== index);
  }

  // Return field management
  function addReturnField() {
    formReturnFields = [...formReturnFields, ''];
  }

  function removeReturnField(index: number) {
    formReturnFields = formReturnFields.filter((_, i) => i !== index);
  }

  // Input variable toggle
  function toggleInputVariable(variable: string) {
    if (formInputVariables.includes(variable)) {
      formInputVariables = formInputVariables.filter(v => v !== variable);
    } else {
      formInputVariables = [...formInputVariables, variable];
    }
    rebuildUserPrompt();
  }

  function rebuildUserPrompt() {
    const sections: string[] = [];
    const varLabels: Record<string, string> = {
      input: '## Agent Input\n{input}',
      output: '## Agent Output\n{output}',
      tool_output: '## Tool Output\n{tool_output}',
    };
    for (const v of ['input', 'output', 'tool_output']) {
      if (formInputVariables.includes(v)) {
        sections.push(varLabels[v]);
      }
    }
    sections.push('Evaluate the above and provide your score.');
    formJudgeUserPrompt = sections.join('\n\n');
  }

  // Generate prompt with AI
  async function generatePromptWithAI() {
    if (!formName.trim()) { alert('Please enter an evaluation name first'); return; }
    if (!formGenerationModel) { alert('Please select a generation model first'); return; }
    isGeneratingPrompt = true;
    formJudgeSystemPrompt = '';

    try {
      const response = await fetch('http://localhost:8000/evaluations/generate-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          eval_name: formName.trim(),
          eval_description: formDescription.trim(),
          score_type: formScoreType,
          score_categories: formCategories.filter(c => c.name.trim()),
          return_fields: formReturnFields.filter(f => f.trim()),
          input_variables: formInputVariables,
          model: formGenerationModel,
        }),
      });

      if (!response.ok) throw new Error(`Failed to generate prompt: ${response.statusText}`);
      if (!response.body) throw new Error('No response body');

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);
            if (data.chunk) formJudgeSystemPrompt += data.chunk;
            if (data.error) throw new Error(data.error);
          } catch (e) {
            if (line.trim() && !line.startsWith('{')) formJudgeSystemPrompt += line;
          }
        }
      }
    } catch (error) {
      console.error('Failed to generate prompt:', error);
      alert(`Failed to generate prompt: ${error}`);
    } finally {
      isGeneratingPrompt = false;
    }
  }

  async function saveEvaluation() {
    if (!formName.trim()) { alert('Please enter a name'); return; }
    if (!formJudgeSystemPrompt.trim()) { alert('Please enter a judge system prompt'); return; }
    if (!formLlmProvider) { alert('Please select an LLM provider'); return; }
    if (formScoreType === 'CATEGORICAL' && formCategories.filter(c => c.name.trim()).length === 0) {
      alert('Please add at least one category');
      return;
    }

    saving = true;
    const categories = (formScoreType === 'CATEGORICAL' || formScoreType === 'NUMERIC')
      ? formCategories.filter(c => c.name.trim()).map(c => ({ name: c.name.trim(), description: c.description?.trim() || undefined }))
      : null;

    const data: any = {
      name: formName.trim(),
      description: formDescription.trim() || null,
      judge_system_prompt: formJudgeSystemPrompt,
      judge_user_prompt: formJudgeUserPrompt || null,
      scoring_rubric: formScoringRubric.trim() || null,
      llm_provider: formLlmProvider,
      score_type: formScoreType,
      score_categories: categories && categories.length > 0 ? categories : null,
      input_variables: formInputVariables.length > 0 ? formInputVariables : null,
      return_fields: formReturnFields.filter(f => f.trim()).length > 0 ? formReturnFields.filter(f => f.trim()) : null,
      is_public: formIsPublic,
    };

    if (editingEvaluation) {
      await updateEvaluation(editingEvaluation.id, data);
    } else {
      await createEvaluation(userId, data);
    }

    await loadEvaluations();
    saving = false;
    closeModal();
  }

  async function handleDelete(id: number) {
    if (!confirm('Delete this evaluation?')) return;
    await deleteEvaluation(id);
    await loadEvaluations();
  }
</script>

<div class="eval-section">
  <h4>Evaluations</h4>
  <Button size="sm" onclick={() => { showManager = !showManager; }} class="w-full mb-2 bg-purple-600 hover:bg-purple-700">
    {#snippet children()}
      {showManager ? 'Hide Evaluations' : 'Manage Evaluations'}
    {/snippet}
  </Button>

  {#if showManager}
    <Button size="sm" onclick={openNewModal} class="w-full mb-2" variant="outline">
      {#snippet children()}+ New Evaluation{/snippet}
    </Button>

    {#if evaluations.length === 0}
      <div class="no-items">No evaluations defined</div>
    {:else}
      <div class="eval-list">
        {#each evaluations as evaluation}
          <div class="eval-item">
            <div class="eval-info">
              <span class="eval-name">{evaluation.name}</span>
              <span class="eval-meta">
                {evaluation.score_type.toLowerCase()}
                {#if evaluation.is_public}
                  &middot; public
                {/if}
              </span>
            </div>
            <div class="eval-actions">
              <button class="action-btn edit" onclick={() => openEditModal(evaluation)} title="Edit">&#9998;</button>
              <button class="action-btn delete" onclick={() => handleDelete(evaluation.id)} title="Delete">&times;</button>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  {/if}
</div>

<!-- Create/Edit Modal -->
{#if showModal}
  <div class="fixed inset-0 z-50 flex items-center justify-center">
    <div
      class="fixed inset-0 bg-black/50"
      onclick={closeModal}
      onkeydown={(e) => e.key === 'Escape' && closeModal()}
      role="button"
      tabindex="-1"
    ></div>

    <div class="relative z-50 w-full max-w-lg bg-background border rounded-lg shadow-lg p-6 mx-4 max-h-[85vh] overflow-y-auto">
      <div class="mb-4">
        <h2 class="text-lg font-semibold">{editingEvaluation ? 'Edit Evaluation' : 'New Evaluation'}</h2>
        <p class="text-sm text-muted-foreground">Define a custom LLM-as-a-judge evaluation.</p>
      </div>

      <button
        class="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
        onclick={closeModal}
        aria-label="Close"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      </button>

      <div class="space-y-4">
        <!-- Section 1: Basics -->
        <div class="space-y-2">
          <Label for="evalName">Name</Label>
          <Input id="evalName" bind:value={formName} placeholder="e.g. Helpfulness" />
        </div>

        <div class="space-y-2">
          <Label for="evalDesc">Description</Label>
          <Input id="evalDesc" bind:value={formDescription} placeholder="What this evaluation measures" />
        </div>

        <!-- Section 2: Data Type -->
        <div class="space-y-2">
          <Label for="evalScoreType">Data Type</Label>
          <select
            id="evalScoreType"
            bind:value={formScoreType}
            class="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="NUMERIC">Numeric (0-1)</option>
            <option value="BOOLEAN">Boolean (pass/fail)</option>
            <option value="CATEGORICAL">Categorical</option>
          </select>
        </div>

        <!-- Categories (for Numeric and Categorical) -->
        {#if formScoreType === 'CATEGORICAL' || formScoreType === 'NUMERIC'}
          <div class="space-y-2">
            <Label>{formScoreType === 'CATEGORICAL' ? 'Categories' : 'Score Ranges'}</Label>
            <p class="text-xs text-muted-foreground">
              {formScoreType === 'CATEGORICAL'
                ? 'Define the allowed category values.'
                : 'Optionally define what different score ranges mean.'}
            </p>
            {#each formCategories as category, index}
              <div class="category-row">
                <Input
                  bind:value={formCategories[index].name}
                  placeholder={formScoreType === 'CATEGORICAL' ? 'Category name' : 'Range (e.g. 0.0-0.3)'}
                  class="flex-1"
                />
                <Input
                  bind:value={formCategories[index].description}
                  placeholder="Description (optional)"
                  class="flex-1"
                />
                <button class="remove-btn" onclick={() => removeCategory(index)} title="Remove">&times;</button>
              </div>
            {/each}
            <button class="add-btn" onclick={addCategory}>
              + Add {formScoreType === 'CATEGORICAL' ? 'category' : 'range'}
            </button>
          </div>
        {/if}

        <!-- Section 3: Inputs -->
        <div class="space-y-2">
          <Label>Inputs</Label>
          <p class="text-xs text-muted-foreground">Select which execution data the judge receives.</p>
          <div class="input-vars">
            {#each [
              { key: 'output', label: 'Agent Output' },
              { key: 'input', label: 'Agent Input' },
              { key: 'tool_output', label: 'Tool Output' }
            ] as variable}
              <label class="input-var-item">
                <input
                  type="checkbox"
                  checked={formInputVariables.includes(variable.key)}
                  onchange={() => toggleInputVariable(variable.key)}
                />
                <span>{variable.label}</span>
                <code>&#123;{variable.key}&#125;</code>
              </label>
            {/each}
          </div>
        </div>

        <!-- Section 4: Return Fields -->
        <div class="space-y-2">
          <Label>Return Fields (optional)</Label>
          <p class="text-xs text-muted-foreground">Additional fields the judge should return beyond the score.</p>
          {#each formReturnFields as field, index}
            <div class="return-field-row">
              <Input
                bind:value={formReturnFields[index]}
                placeholder="e.g. reasoning, confidence"
              />
              <button class="remove-btn" onclick={() => removeReturnField(index)} title="Remove">&times;</button>
            </div>
          {/each}
          <button class="add-btn" onclick={addReturnField}>+ Add return field</button>
        </div>

        <!-- Evaluation Model -->
        <div class="space-y-2">
          <Label for="evalModel">Evaluation Model</Label>
          <p class="text-xs text-muted-foreground">The model that runs this evaluation at execution time.</p>
          <select
            id="evalModel"
            bind:value={formLlmProvider}
            class="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            {#each llmProviderNames as name}
              <option value={name}>{name}</option>
            {/each}
          </select>
        </div>

        <!-- Section 5: Judge Prompt -->
        <div class="space-y-2">
          <div class="prompt-header">
            <Label for="evalPrompt">Judge System Prompt</Label>
            <div class="generate-row">
              <select
                bind:value={formGenerationModel}
                class="generate-model-select"
              >
                {#each llmProviderNames as name}
                  <option value={name}>{name}</option>
                {/each}
              </select>
              <button
                class="generate-btn"
                onclick={generatePromptWithAI}
                disabled={isGeneratingPrompt}
              >
                {isGeneratingPrompt ? 'Generating...' : 'Generate with AI'}
              </button>
            </div>
          </div>
          <textarea
            id="evalPrompt"
            bind:value={formJudgeSystemPrompt}
            placeholder="You are an expert evaluator..."
            class="flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-mono"
            rows="8"
          ></textarea>
        </div>

        <!-- User Prompt (built from input variables) -->
        <div class="space-y-2">
          <Label for="evalUserPrompt">User Prompt</Label>
          <p class="text-xs text-muted-foreground">This is sent to the judge with the actual execution data substituted in.</p>
          <textarea
            id="evalUserPrompt"
            bind:value={formJudgeUserPrompt}
            class="flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring font-mono"
            rows="6"
          ></textarea>
        </div>

        <div class="space-y-2">
          <Label for="evalRubric">Scoring Rubric (optional)</Label>
          <textarea
            id="evalRubric"
            bind:value={formScoringRubric}
            placeholder="Additional scoring criteria appended to the user message..."
            class="flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm shadow-sm focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            rows="3"
          ></textarea>
        </div>

        <div class="flex items-center gap-2">
          <input
            type="checkbox"
            id="evalPublic"
            bind:checked={formIsPublic}
            class="rounded border-input"
          />
          <Label for="evalPublic">Public (visible to all users)</Label>
        </div>
      </div>

      <div class="flex justify-end gap-2 mt-6">
        <Button variant="outline" onclick={closeModal}>
          {#snippet children()}Cancel{/snippet}
        </Button>
        <Button onclick={saveEvaluation} disabled={saving} class="bg-purple-600 hover:bg-purple-700">
          {#snippet children()}{saving ? 'Saving...' : (editingEvaluation ? 'Update' : 'Create')}{/snippet}
        </Button>
      </div>
    </div>
  </div>
{/if}

<style>
  .eval-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  h4 {
    font-size: 12px;
    margin: 0 0 8px 0;
    color: #333;
    font-weight: bold;
  }

  .no-items {
    font-size: 11px;
    color: #666;
    text-align: center;
    padding: 8px;
  }

  .eval-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .eval-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 8px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
  }

  .eval-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    overflow: hidden;
  }

  .eval-name {
    font-size: 12px;
    font-weight: 500;
    color: #333;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .eval-meta {
    font-size: 10px;
    color: #666;
  }

  .eval-actions {
    display: flex;
    gap: 4px;
  }

  .action-btn {
    background: none;
    border: none;
    padding: 2px 6px;
    cursor: pointer;
    font-size: 14px;
    border-radius: 3px;
    transition: background 0.2s;
  }

  .action-btn.edit { color: #666; }
  .action-btn.edit:hover { background: #e0e0e0; }
  .action-btn.delete { color: #dc3545; }
  .action-btn.delete:hover { background: #fee; }

  textarea {
    resize: vertical;
    min-height: 60px;
    font-family: inherit;
  }

  .category-row, .return-field-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .remove-btn {
    background: none;
    border: none;
    color: #dc3545;
    font-size: 18px;
    cursor: pointer;
    padding: 0 4px;
    flex-shrink: 0;
  }
  .remove-btn:hover { color: #b02a37; }

  .add-btn {
    background: none;
    border: 1px dashed #ccc;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 12px;
    color: #666;
    cursor: pointer;
    width: 100%;
    text-align: center;
  }
  .add-btn:hover { border-color: #999; color: #333; }

  .input-vars {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .input-var-item {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    cursor: pointer;
  }
  .input-var-item code {
    font-size: 11px;
    color: #888;
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
  }

  .prompt-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .generate-row {
    display: flex;
    gap: 6px;
    align-items: center;
  }

  .generate-model-select {
    height: 28px;
    padding: 0 6px;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 11px;
    background: white;
    max-width: 140px;
  }

  .generate-btn {
    background: #2563eb;
    color: white;
    border: none;
    padding: 4px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }
  .generate-btn:hover:not(:disabled) { background: #1d4ed8; }
  .generate-btn:disabled { opacity: 0.5; cursor: not-allowed; }

</style>
