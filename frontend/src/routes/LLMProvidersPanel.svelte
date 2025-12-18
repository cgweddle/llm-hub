<script lang="ts">
  import { onMount } from 'svelte';
  import { Button } from "$lib/components/ui/button";
  import { Input } from "$lib/components/ui/input";
  import { Label } from "$lib/components/ui/label";
  import { loadLLMProvidersConfig, saveLLMProvidersConfig } from '$lib/api';

  // LLM Provider types
  type ProviderType = 'anthropic' | 'openai' | 'gemini' | 'lmstudio';

  interface LLMProvider {
    name: string;
    provider: ProviderType;
    apiKey?: string;
    baseUrl?: string;
    model: string;
  }

  // Provider configurations
  const providerConfigs: Record<ProviderType, { label: string; models: string[]; requiresApiKey: boolean; requiresBaseUrl: boolean; defaultBaseUrl?: string }> = {
    anthropic: {
      label: 'Anthropic',
      models: ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
      requiresApiKey: true,
      requiresBaseUrl: false
    },
    openai: {
      label: 'OpenAI',
      models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo', 'o1-preview', 'o1-mini'],
      requiresApiKey: true,
      requiresBaseUrl: false
    },
    gemini: {
      label: 'Google Gemini',
      models: ['gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-1.0-pro', 'gemini-2.0-flash-exp'],
      requiresApiKey: true,
      requiresBaseUrl: false
    },
    lmstudio: {
      label: 'LM Studio',
      models: [],
      requiresApiKey: false,
      requiresBaseUrl: true,
      defaultBaseUrl: 'http://localhost:1234/v1'
    }
  };

  // Props
  interface Props {
    selectedProvider?: LLMProvider | null;
    providers?: LLMProvider[];
  }

  let { selectedProvider = $bindable(null), providers = $bindable([]) }: Props = $props();

  // State using Svelte 5 runes
  let showModal = $state(false);
  let editingProvider = $state<LLMProvider | null>(null);

  // Form state
  let formName = $state('');
  let formProvider = $state<ProviderType>('anthropic');
  let formApiKey = $state('');
  let formBaseUrl = $state('');
  let formModel = $state('');
  let formCustomModel = $state('');


  // Derived state
  let currentConfig = $derived(providerConfigs[formProvider]);

  function openNewModal() {
    editingProvider = null;
    formName = '';
    formProvider = 'anthropic';
    formApiKey = '';
    formBaseUrl = '';
    formModel = providerConfigs.anthropic.models[0] || '';
    formCustomModel = '';
    showModal = true;
  }

  function openEditModal(provider: LLMProvider) {
    editingProvider = provider;
    formName = provider.name;
    formProvider = provider.provider;
    formApiKey = provider.apiKey || '';
    formBaseUrl = provider.baseUrl || '';
    formModel = provider.model;
    formCustomModel = providerConfigs[provider.provider].models.includes(provider.model) ? '' : provider.model;
    showModal = true;
  }

  function closeModal() {
    showModal = false;
    editingProvider = null;
  }

  function handleProviderChange(value: string | undefined) {
    if (!value) return;
    formProvider = value as ProviderType;
    const config = providerConfigs[formProvider];
    formModel = config.models[0] || '';
    formCustomModel = '';
    formBaseUrl = config.defaultBaseUrl || '';
  }

  function handleModelChange(value: string | undefined) {
    if (value !== undefined) {
      formModel = value;
      if (value !== '__custom__') {
        formCustomModel = '';
      }
    }
  }

  function saveProvider() {
    const finalModel = formModel === '__custom__' ? formCustomModel : (formCustomModel || formModel);

    if (!formName.trim()) {
      alert('Please enter a name for this LLM configuration');
      return;
    }

    if (!finalModel.trim()) {
      alert('Please select or enter a model');
      return;
    }

    const config = providerConfigs[formProvider];
    if (config.requiresApiKey && !formApiKey.trim()) {
      alert('API key is required for this provider');
      return;
    }

    if (config.requiresBaseUrl && !formBaseUrl.trim()) {
      alert('Base URL is required for this provider');
      return;
    }

    const newProvider: LLMProvider = {
      name: formName.trim(),
      provider: formProvider,
      apiKey: formApiKey.trim() || undefined,
      baseUrl: formBaseUrl.trim() || undefined,
      model: finalModel
    };

    if (editingProvider) {
      // Replace provider with same name
      providers = providers.map(p => p.name === editingProvider!.name ? newProvider : p);
      // Update selected provider if it was the one being edited
      if (selectedProvider?.name === editingProvider.name) {
        selectedProvider = newProvider;
      }
    } else {
      providers = [...providers, newProvider];
    }

    // Auto-select if first provider
    if (providers.length === 1) {
      selectedProvider = providers[0];
    }

    closeModal();
  }

  function deleteProvider(name: string) {
    providers = providers.filter(p => p.name !== name);
    if (selectedProvider?.name === name) {
      selectedProvider = providers[0] || null;
    }
  }

  function selectProvider(provider: LLMProvider) {
    selectedProvider = provider;
  }

  function maskApiKey(key: string | undefined): string {
    if (!key) return '';
    if (key.length <= 8) return '****';
    return key.substring(0, 4) + '****' + key.substring(key.length - 4);
  }

  // Load config on mount
  onMount(async () => {
    try {
      const config = await loadLLMProvidersConfig();

      if (config.models && config.models.length > 0) {
        // Map loaded config to LLMProvider format (adjusting for snake_case)
        providers = config.models.map(p => ({
          name: p.name,
          provider: p.provider as ProviderType,
          apiKey: p.api_key,
          baseUrl: p.base_url,
          model: p.model
        }));

        // Auto-select first provider if any exist
        if (providers.length > 0 && !selectedProvider) {
          selectedProvider = providers[0];
        }
      }
    } catch (error) {
      console.error('Failed to load LLM providers config:', error);
    }
  });

  // Auto-save whenever providers or selectedProvider changes
  $effect(() => {
    // Skip the initial run (when providers is empty on first load)
    if (providers.length > 0 || selectedProvider !== null) {
      saveConfigDebounced();
    }
  });

  // Debounced save to avoid too many writes
  let saveTimeout: ReturnType<typeof setTimeout> | null = null;
  function saveConfigDebounced() {
    if (saveTimeout) {
      clearTimeout(saveTimeout);
    }

    saveTimeout = setTimeout(async () => {
      try {
        // Convert to snake_case format for API
        const modelsToSave = providers.map(p => ({
          name: p.name,
          provider: p.provider,
          api_key: p.apiKey,
          base_url: p.baseUrl,
          model: p.model
        }));

        await saveLLMProvidersConfig({
          models: modelsToSave
        });

        console.log('LLM providers config saved to ~/.llm_hub/config.yaml');
      } catch (error) {
        console.error('Failed to save LLM providers config:', error);
      }
    }, 500); // Debounce for 500ms
  }
</script>

<div class="llm-section">
  <h4>Attach LLM</h4>

  <Button size="sm" onclick={openNewModal} class="w-full mb-2 bg-blue-600 hover:bg-blue-700">
    {#snippet children()}
      New LLM
    {/snippet}
  </Button>

  {#if providers.length === 0}
    <div class="no-providers">
      No LLMs configured
    </div>
  {:else}
    <div class="provider-list">
      {#each providers as provider}
        <div
          class="provider-item {selectedProvider?.name === provider.name ? 'selected' : ''}"
          onclick={() => selectProvider(provider)}
          onkeydown={(e) => e.key === 'Enter' && selectProvider(provider)}
          role="button"
          tabindex="0"
        >
          <div class="provider-info">
            <span class="provider-name">{provider.name}</span>
            <span class="provider-type">{providerConfigs[provider.provider].label}</span>
          </div>
          <div class="provider-actions">
            <button class="action-btn edit" onclick={(e) => { e.stopPropagation(); openEditModal(provider); }} title="Edit">
              &#9998;
            </button>
            <button class="action-btn delete" onclick={(e) => { e.stopPropagation(); deleteProvider(provider.name); }} title="Delete">
              &times;
            </button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<!-- Modal -->
{#if showModal}
  <div class="fixed inset-0 z-50 flex items-center justify-center">
    <!-- Backdrop -->
    <div
      class="fixed inset-0 bg-black/50"
      onclick={closeModal}
      onkeydown={(e) => e.key === 'Escape' && closeModal()}
      role="button"
      tabindex="-1"
    ></div>

    <!-- Modal Content -->
    <div class="relative z-50 w-full max-w-md bg-background border rounded-lg shadow-lg p-6 mx-4">
      <!-- Header -->
      <div class="mb-4">
        <h2 class="text-lg font-semibold">{editingProvider ? 'Edit LLM' : 'New LLM'}</h2>
        <p class="text-sm text-muted-foreground">Configure your LLM provider settings below.</p>
      </div>

      <!-- Close button -->
      <button
        class="absolute top-4 right-4 text-muted-foreground hover:text-foreground"
        onclick={closeModal}
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
      </button>

      <!-- Form -->
      <div class="space-y-4">
        <div class="space-y-2">
          <Label for="llmName">Configuration Name</Label>
          <Input id="llmName" bind:value={formName} placeholder="My Claude Config" />
        </div>

        <div class="space-y-2">
          <Label for="llmProvider">Provider</Label>
          <select
            id="llmProvider"
            class="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            bind:value={formProvider}
            onchange={() => handleProviderChange(formProvider)}
          >
            {#each Object.entries(providerConfigs) as [key, config]}
              <option value={key}>{config.label}</option>
            {/each}
          </select>
        </div>

        {#if currentConfig.requiresApiKey}
          <div class="space-y-2">
            <Label for="llmApiKey">API Key</Label>
            <Input
              id="llmApiKey"
              type="password"
              bind:value={formApiKey}
              placeholder="sk-..."
            />
          </div>
        {/if}

        {#if currentConfig.requiresBaseUrl}
          <div class="space-y-2">
            <Label for="llmBaseUrl">Base URL</Label>
            <Input
              id="llmBaseUrl"
              bind:value={formBaseUrl}
              placeholder={currentConfig.defaultBaseUrl || 'http://localhost:1234/v1'}
            />
          </div>
        {/if}

        <div class="space-y-2">
          <Label for="llmModel">Model</Label>
          {#if currentConfig.models.length > 0}
            <select
              id="llmModel"
              class="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              bind:value={formModel}
            >
              {#each currentConfig.models as model}
                <option value={model}>{model}</option>
              {/each}
              <option value="__custom__">Custom...</option>
            </select>
          {/if}
          {#if currentConfig.models.length === 0 || formModel === '__custom__'}
            <Input
              bind:value={formCustomModel}
              placeholder="Enter model name"
              class="mt-2"
            />
          {/if}
        </div>

      </div>

      <!-- Footer -->
      <div class="flex justify-end gap-2 mt-6">
        <Button variant="outline" onclick={closeModal}>
          {#snippet children()}Cancel{/snippet}
        </Button>
        <Button onclick={saveProvider} class="bg-blue-600 hover:bg-blue-700">
          {#snippet children()}{editingProvider ? 'Update' : 'Add LLM'}{/snippet}
        </Button>
      </div>
    </div>
  </div>
{/if}

<style>
  .llm-section {
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

  .no-providers {
    font-size: 11px;
    color: #666;
    text-align: center;
    padding: 8px;
  }

  .provider-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .provider-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 8px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }

  .provider-item:hover {
    background: #f5f5f5;
  }

  .provider-item.selected {
    border-color: #007acc;
    background: #e8f4fc;
  }

  .provider-info {
    display: flex;
    flex-direction: column;
    gap: 2px;
    overflow: hidden;
  }

  .provider-name {
    font-size: 12px;
    font-weight: 500;
    color: #333;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .provider-type {
    font-size: 10px;
    color: #666;
  }

  .provider-actions {
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

  .action-btn.edit {
    color: #666;
  }

  .action-btn.edit:hover {
    background: #e0e0e0;
  }

  .action-btn.delete {
    color: #dc3545;
  }

  .action-btn.delete:hover {
    background: #fee;
  }
</style>
