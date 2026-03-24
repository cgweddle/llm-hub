<script lang="ts">
  import { fetchExecution, fetchExecutionTrace, type ExecutionDetail, type TraceDetail } from '$lib/api';

  let {
    executionId = null,
    flowName = '',
    onclose = () => {}
  }: {
    executionId: number | null;
    flowName: string;
    onclose: () => void;
  } = $props();

  let execution: ExecutionDetail | null = $state(null);
  let loading = $state(false);
  let expandedNodes: Set<number> = $state(new Set());
  let panelHeight = $state(250);
  let isResizing = $state(false);
  let traceCache: Map<number, TraceDetail | null> = $state(new Map());
  let traceLoading: Set<number> = $state(new Set());

  $effect(() => {
    if (executionId) {
      loadExecution(executionId);
    } else {
      execution = null;
    }
  });

  async function loadExecution(id: number) {
    loading = true;
    try {
      execution = await fetchExecution(id);
    } catch (e) {
      console.error('Failed to load execution:', e);
      execution = null;
    } finally {
      loading = false;
    }
  }

  function toggleNode(node: ExecutionDetail) {
    const next = new Set(expandedNodes);
    if (next.has(node.id)) {
      next.delete(node.id);
    } else {
      next.add(node.id);
      // Fetch LangFuse trace for agent nodes when expanding
      if (node.execution_type === 'agent' && node.langfuse_trace_id && !traceCache.has(node.id)) {
        loadTrace(node.id);
      }
    }
    expandedNodes = next;
  }

  async function loadTrace(executionId: number) {
    const nextLoading = new Set(traceLoading);
    nextLoading.add(executionId);
    traceLoading = nextLoading;
    try {
      const trace = await fetchExecutionTrace(executionId);
      const nextCache = new Map(traceCache);
      nextCache.set(executionId, trace);
      traceCache = nextCache;
    } catch (e) {
      console.error('Failed to load trace:', e);
    } finally {
      const done = new Set(traceLoading);
      done.delete(executionId);
      traceLoading = done;
    }
  }

  function formatDuration(start: string | null, end: string | null): string {
    if (!start || !end) return '';
    const ms = new Date(end).getTime() - new Date(start).getTime();
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function formatJson(data: any): string {
    if (data === null || data === undefined) return 'null';
    if (typeof data === 'string') return data;
    try {
      return JSON.stringify(data, null, 2);
    } catch {
      return String(data);
    }
  }

  function truncate(text: string, maxLen: number = 80): string {
    if (text.length <= maxLen) return text;
    return text.slice(0, maxLen) + '...';
  }

  function startResize(e: MouseEvent) {
    e.preventDefault();
    isResizing = true;
    const startY = e.clientY;
    const startHeight = panelHeight;

    function onMouseMove(e: MouseEvent) {
      const delta = startY - e.clientY;
      panelHeight = Math.max(100, Math.min(600, startHeight + delta));
    }

    function onMouseUp() {
      isResizing = false;
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    }

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  }

  function typeColor(type: string): string {
    switch (type) {
      case 'trigger': return 'var(--info-type-trigger)';
      case 'tool': return 'var(--info-type-tool)';
      case 'agent': return 'var(--info-type-agent)';
      case 'flow': return 'var(--info-type-flow)';
      default: return 'var(--info-type-default)';
    }
  }
</script>

<div class="info-panel" style="height: {panelHeight}px" class:resizing={isResizing}>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="resize-handle" onmousedown={startResize}>
    <div class="resize-grip"></div>
  </div>
  <div class="info-header">
    <button class="close-btn" onclick={onclose} title="Close">&times;</button>
    <div class="info-title">
      <span class="info-label">Info</span>
      {#if execution}
        <span class="info-separator">|</span>
        <span class="info-flow-name">{flowName || execution.name || 'Execution'}</span>
        <span class="status-dot {execution.status}"></span>
        <span class="status-text">{execution.status}</span>
        {#if execution.started_at && execution.completed_at}
          <span class="duration">{formatDuration(execution.started_at, execution.completed_at)}</span>
        {/if}
      {/if}
    </div>
  </div>

  <div class="info-body">
    {#if loading}
      <div class="info-empty">Loading execution...</div>
    {:else if !execution}
      <div class="info-empty">No execution data. Run a flow to see results here.</div>
    {:else if execution.children.length === 0}
      <div class="info-empty">Execution completed with no recorded steps.</div>
    {:else}
      <div class="execution-tree">
        {#each execution.children as child (child.id)}
          {@render treeNode(child, 0)}
        {/each}
      </div>
    {/if}
  </div>
</div>

{#snippet treeNode(node: ExecutionDetail, depth: number)}
  <div class="tree-row" style="padding-left: {12 + depth * 20}px">
    <button
      class="expand-toggle"
      onclick={() => toggleNode(node)}
      aria-label={expandedNodes.has(node.id) ? 'Collapse' : 'Expand'}
    >
      {expandedNodes.has(node.id) ? '▾' : '▸'}
    </button>
    <span class="node-seq">{node.sequence ?? '-'}</span>
    <span class="node-type-badge" style="background: {typeColor(node.execution_type)}">{node.execution_type}</span>
    <span class="node-name">{node.name || node.node_id || 'unnamed'}</span>
    <span class="status-dot {node.status}"></span>
    {#if node.started_at && node.completed_at}
      <span class="duration">{formatDuration(node.started_at, node.completed_at)}</span>
    {/if}
    {#if node.langfuse_trace_id}
      <span class="trace-badge">traced</span>
    {/if}
    {#if node.error_message}
      <span class="error-preview" title={node.error_message}>
        {truncate(node.error_message, 50)}
      </span>
    {/if}
  </div>

  {#if expandedNodes.has(node.id)}
    <div class="node-details" style="padding-left: {32 + depth * 20}px">
      {#if node.input_data !== null && node.input_data !== undefined}
        <div class="detail-section">
          <span class="detail-label">Input:</span>
          <pre class="detail-json">{formatJson(node.input_data)}</pre>
        </div>
      {/if}
      {#if node.output_data !== null && node.output_data !== undefined}
        <div class="detail-section">
          <span class="detail-label">Output:</span>
          <pre class="detail-json">{formatJson(node.output_data)}</pre>
        </div>
      {/if}

      <!-- LangFuse trace data for agent nodes -->
      {#if node.langfuse_trace_id}
        {#if traceLoading.has(node.id)}
          <div class="detail-section">
            <span class="detail-label">LLM Trace:</span>
            <div class="trace-loading">Loading trace...</div>
          </div>
        {:else if traceCache.has(node.id) && traceCache.get(node.id)}
          {@const trace = traceCache.get(node.id)!}
          <div class="detail-section">
            <span class="detail-label">LLM Trace ({trace.observations.length} observations):</span>
            <div class="trace-observations">
              {#each trace.observations as obs (obs.id)}
                <div class="trace-obs">
                  <div class="trace-obs-header">
                    <span class="trace-obs-type" style="background: {obs.type === 'GENERATION' ? 'oklch(0.42 0.12 200)' : 'oklch(0.42 0.1 50)'}">{obs.type}</span>
                    <span class="trace-obs-name">{obs.name || 'unnamed'}</span>
                    {#if obs.model}
                      <span class="trace-obs-model">{obs.model}</span>
                    {/if}
                    {#if obs.usage}
                      <span class="trace-obs-tokens">{obs.usage.input ?? '?'}→{obs.usage.output ?? '?'} tokens</span>
                    {/if}
                  </div>
                  {#if obs.input}
                    <div class="trace-obs-content">
                      <span class="detail-label">Input:</span>
                      <pre class="detail-json">{formatJson(obs.input)}</pre>
                    </div>
                  {/if}
                  {#if obs.output}
                    <div class="trace-obs-content">
                      <span class="detail-label">Output:</span>
                      <pre class="detail-json">{formatJson(obs.output)}</pre>
                    </div>
                  {/if}
                  {#if obs.status_message}
                    <div class="trace-obs-content">
                      <span class="detail-label error-text">Error:</span>
                      <pre class="detail-json error-json">{obs.status_message}</pre>
                    </div>
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {:else if traceCache.has(node.id) && !traceCache.get(node.id)}
          <div class="detail-section">
            <span class="detail-label">LLM Trace:</span>
            <div class="trace-loading">Trace not available</div>
          </div>
        {/if}
      {/if}
    </div>

    {#if node.children && node.children.length > 0}
      {#each node.children as child (child.id)}
        {@render treeNode(child, depth + 1)}
      {/each}
    {/if}
  {/if}
{/snippet}

<style>
  .info-panel {
    background: oklch(0.17 0.005 260);
    display: flex;
    flex-direction: column;
    min-height: 100px;
    font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 12px;
    color: oklch(0.85 0 0);
    flex-shrink: 0;
  }

  .info-panel.resizing {
    user-select: none;
  }

  .resize-handle {
    height: 6px;
    cursor: ns-resize;
    display: flex;
    align-items: center;
    justify-content: center;
    background: oklch(0.15 0.005 260);
    border-top: 1px solid oklch(0.7 0 0 / 0.3);
    flex-shrink: 0;
  }
  .resize-handle:hover {
    background: oklch(0.2 0.01 260);
  }

  .resize-grip {
    width: 40px;
    height: 2px;
    background: oklch(0.4 0 0);
    border-radius: 1px;
  }

  .info-header {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    background: oklch(0.15 0.005 260);
    border-bottom: 1px solid oklch(0.7 0 0 / 0.15);
    flex-shrink: 0;
  }

  .info-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: oklch(0.7 0 0);
  }

  .info-label {
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .info-separator {
    color: oklch(0.4 0 0);
  }

  .info-flow-name {
    color: oklch(0.85 0 0);
    text-transform: none;
    letter-spacing: 0;
    font-weight: 500;
  }

  .close-btn {
    background: none;
    border: none;
    color: oklch(0.5 0 0);
    font-size: 16px;
    cursor: pointer;
    padding: 0 6px 0 0;
    line-height: 1;
    flex-shrink: 0;
  }
  .close-btn:hover {
    color: oklch(0.9 0 0);
  }

  .info-body {
    flex: 1;
    overflow-y: auto;
    padding: 4px 0;
  }

  .info-empty {
    padding: 20px;
    text-align: center;
    color: oklch(0.5 0 0);
    font-style: italic;
  }

  .tree-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 12px;
    cursor: default;
    user-select: none;
  }
  .tree-row:hover {
    background: oklch(0.2 0.005 260);
  }

  .expand-toggle {
    background: none;
    border: none;
    color: oklch(0.6 0 0);
    font-size: 11px;
    cursor: pointer;
    padding: 0;
    width: 14px;
    text-align: center;
    flex-shrink: 0;
  }

  .node-seq {
    color: oklch(0.5 0 0);
    min-width: 16px;
    text-align: right;
    flex-shrink: 0;
  }

  .node-type-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    font-weight: 600;
    color: oklch(0.95 0 0);
    flex-shrink: 0;
  }

  .node-name {
    color: oklch(0.85 0 0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .status-dot.completed { background: oklch(0.7 0.15 145); }
  .status-dot.failed { background: oklch(0.65 0.2 25); }
  .status-dot.running { background: oklch(0.75 0.15 80); }

  .status-text {
    color: oklch(0.6 0 0);
    font-size: 11px;
  }

  .duration {
    color: oklch(0.55 0 0);
    font-size: 11px;
    flex-shrink: 0;
  }

  .error-preview {
    color: oklch(0.7 0.15 25);
    font-size: 11px;
    margin-left: auto;
  }

  .node-details {
    padding: 4px 12px 8px;
  }

  .detail-section {
    margin-bottom: 4px;
  }

  .detail-label {
    color: oklch(0.6 0 0);
    font-size: 11px;
    font-weight: 600;
  }

  .detail-json {
    margin: 2px 0 0 0;
    padding: 6px 10px;
    background: oklch(0.12 0.005 260);
    border-radius: 4px;
    border: 1px solid oklch(0.7 0 0 / 0.1);
    color: oklch(0.78 0.08 180);
    font-size: 11px;
    line-height: 1.4;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* CSS variables for type badge colors */
  .trace-badge {
    font-size: 9px;
    padding: 1px 4px;
    border-radius: 3px;
    background: oklch(0.35 0.08 200);
    color: oklch(0.8 0.05 200);
    text-transform: uppercase;
    letter-spacing: 0.3px;
    flex-shrink: 0;
  }

  .trace-loading {
    color: oklch(0.5 0 0);
    font-style: italic;
    padding: 4px 0;
  }

  .trace-observations {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin-top: 4px;
  }

  .trace-obs {
    border: 1px solid oklch(0.7 0 0 / 0.1);
    border-radius: 4px;
    background: oklch(0.14 0.005 260);
    overflow: hidden;
  }

  .trace-obs-header {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 8px;
    background: oklch(0.16 0.005 260);
    border-bottom: 1px solid oklch(0.7 0 0 / 0.08);
  }

  .trace-obs-type {
    font-size: 9px;
    padding: 1px 5px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
    font-weight: 600;
    color: oklch(0.95 0 0);
    flex-shrink: 0;
  }

  .trace-obs-name {
    color: oklch(0.8 0 0);
    font-weight: 500;
  }

  .trace-obs-model {
    color: oklch(0.55 0.05 200);
    font-size: 10px;
    margin-left: auto;
  }

  .trace-obs-tokens {
    color: oklch(0.5 0 0);
    font-size: 10px;
    flex-shrink: 0;
  }

  .trace-obs-content {
    padding: 4px 8px;
  }

  .error-text {
    color: oklch(0.7 0.15 25);
  }

  .error-json {
    border-color: oklch(0.5 0.1 25 / 0.3);
    color: oklch(0.75 0.1 25);
  }

  :global(:root) {
    --info-type-trigger: oklch(0.45 0.12 280);
    --info-type-tool: oklch(0.42 0.12 200);
    --info-type-agent: oklch(0.45 0.12 145);
    --info-type-flow: oklch(0.42 0.12 50);
    --info-type-default: oklch(0.35 0 0);
  }
</style>
