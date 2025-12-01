<script lang="ts">
  import { onMount } from 'svelte';
  import { fetchCondaEnvironments, type CondaEnvironment } from '$lib/api';

  let environments: CondaEnvironment[] = [];
  let loading = true;
  let selectedEnv = '';

  onMount(async () => {
    try {
      environments = await fetchCondaEnvironments();
      if (environments.length > 0) {
        selectedEnv = environments[0].path;
      }
    } catch (err) {
      console.error('Error loading conda environments:', err);
    } finally {
      loading = false;
    }
  });

  function handleChange(event: Event) {
    const target = event.target as HTMLSelectElement;
    selectedEnv = target.value;
    console.log('Selected conda environment:', selectedEnv);
  }
</script>

<div class="conda-section">
  <h4>Python Environment</h4>
  {#if loading}
    <select class="conda-select" disabled>
      <option>Loading...</option>
    </select>
  {:else if environments.length === 0}
    <select class="conda-select" disabled>
      <option>No environments found</option>
    </select>
  {:else}
    <select class="conda-select" bind:value={selectedEnv} on:change={handleChange}>
      {#each environments as env}
        <option value={env.path}>{env.name}</option>
      {/each}
    </select>
  {/if}
</div>

<style>
  .conda-section {
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  h4 {
    font-size: 12px;
    margin: 0 0 5px 0;
    color: #333;
    font-weight: bold;
  }

  .conda-select {
    width: 100%;
    padding: 5px;
    background: white;
    border: 1px solid #ccc;
    font-size: 12px;
    cursor: pointer;
    border-radius: 2px;
  }

  .conda-select:disabled {
    cursor: not-allowed;
    opacity: 0.6;
  }

  .conda-select:focus {
    outline: none;
    border-color: #007acc;
  }
</style>
