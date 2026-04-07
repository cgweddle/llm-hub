# Plan: Agent-Level Evaluation Assignment + Evals Tab

## Context

The LLM-as-a-judge evaluation system exists but currently uses inline buttons in the InfoPanel. The user wants evaluations to be **assigned to agents** (like tools are) and run automatically when a flow executes with an "Evals" toggle enabled. Results should appear in a dedicated "Evals" tab in the InfoPanel.

---

## Part 1: Load Evaluations in Page Server

**File:** `frontend/src/routes/+page.server.ts`

Add `evaluations` to the data loaded at page level, alongside tools/flows/agents.

- Logged-in: fetch `GET /evaluations/?user_id={userId}` in the `Promise.all`
- Not logged-in: return `evaluations: []`
- Add `evaluations` to both return objects

---

## Part 2: Simple Agent — Eval Checkbox List

**File:** `frontend/src/routes/+page.svelte`

Mirror the existing `newAgentSelectedTools` / `toggleToolSelection` pattern.

- **State** (near line 139): add `let newAgentSelectedEvals: number[] = [];`
- **Toggle function** (near line 1301): add `toggleEvalSelection(evalId)` — same pattern as `toggleToolSelection`
- **Reset** in `closeCreateAgentModal()` and `handleConfigureNewAgent()`: add `newAgentSelectedEvals = [];`
- **Graph config** (line 1160, where `tool_ids: newAgentSelectedTools`): add `eval_ids: newAgentSelectedEvals,`
- **Template** (after the "Select Tools" section at line 2136): add "Assign Evaluations" checkbox section using `data.evaluations`

---

## Part 3: Complex Agent Nodes — Eval Assignment

**File:** `frontend/src/routes/FullscreenNodeModal.svelte` (Svelte 4 — uses `export let`, `$:`)

- **Props**: add `export let allEvaluations: Evaluation[] = [];`
- **State** (near line 37): add `let editedAgentEvalIds: number[] = [];`
- **`startEditingAgent()`** (line 42): add `editedAgentEvalIds = [...(nodeData.data.eval_ids || [])];`
- **Toggle function**: add `toggleAgentEval(evalId)` — same pattern as `toggleAgentTool`
- **Save logic** (lines 109, 147): add `eval_ids: [...editedAgentEvalIds],` alongside `tool_ids`
- **Template** (after "Assigned Tools" section ~line 803): add "Assigned Evaluations" checkbox section

**File:** `frontend/src/routes/+page.svelte` (line 1744, FullscreenNodeModal usage)
- Pass `allEvaluations={data.evaluations}`

**File:** `frontend/src/routes/AgentBuilder.svelte`
- `addConfiguredAgentNode()` param type: add `eval_ids: number[]`
- Node data construction: add `eval_ids: nodeData.eval_ids`
- Default node: add `eval_ids: []`

**File:** `frontend/src/routes/+page.svelte` (agent builder node config, ~line 1433)
- When building node data for complex agent config, include `eval_ids: newAgentSelectedEvals`

---

## Part 4: Complex Agent Top-Level Evals

Complex agents need evals for the **entire agent output** (not just per-sub-node).

- Store at `graph_config.eval_ids: number[]` (root level, not per-node)
- **AgentBuilder sidebar**: add an "Agent-Level Evals" section for the top-level complex agent
- When saving the complex agent, include root `eval_ids` in the graph_config alongside `nodes`, `edges`, `entry_point`, `exit_points`

---

## Part 5: Flow "Evals" Toggle

**File:** `frontend/src/routes/+page.svelte`

- **State** (near line 100): add `let evalsEnabled = false;`
- **Template** (after Run button, line 1655, inside `.flow-controls`):
  ```svelte
  <label class="evals-toggle">
    <input type="checkbox" bind:checked={evalsEnabled} />
    <span>Evals</span>
  </label>
  ```
- **`runFlow()` function** (line 688): after execution completes and `result.execution_id` is set, if `evalsEnabled`:
  - Call new function `runPostExecutionEvals(result.execution_id)`
  - This function fetches the execution tree, iterates agent children, looks up `eval_ids` from the agent's `graph_config.nodes[nodeId]`, and calls `evaluateExecution()` for each
  - The agent's `graph_config` is already available on the flow canvas node data (`node.data.graph_config`)
  - For complex agent top-level evals: also check `graph_config.eval_ids` on the root

---

## Part 6: InfoPanel Tab System + Evals Tab

**File:** `frontend/src/routes/InfoPanel.svelte`

### Remove inline eval UI
- Remove from tree rows: `eval-btn` button, `eval-badge evaluating` span
- Remove from expanded details: inline scores display, eval selector popover
- Remove state: `evalSelectorNode`, `selectedEvalIds`, `availableEvaluations`, `evaluatingSet`
- Remove functions: `openEvalSelector`, `closeEvalSelector`, `toggleEvalSelection`, `runEvaluation`
- Remove unused imports: `evaluateExecution`, `fetchEvaluations`
- **Keep**: `scoresCache`, `scoresLoading`, `loadScores`, `expandedScores`, `toggleScoreExpand`, `formatScoreValue` — reused by Evals tab

### Add tab system
- **New state**: `let activeTab: 'info' | 'evals' = $state('info');`
- **Header**: replace static "Info" label with two tab buttons
  ```
  <div class="info-tabs">
    <button class="tab-btn" class:active={activeTab === 'info'} onclick={() => activeTab = 'info'}>Info</button>
    <button class="tab-btn" class:active={activeTab === 'evals'} onclick={() => activeTab = 'evals'}>Evals</button>
  </div>
  ```
- **Body**: wrap existing execution tree in `{#if activeTab === 'info'}`, add Evals tab content in `{:else if activeTab === 'evals'}`

### Evals tab content
- On tab switch (or auto when evals complete), load scores for all traced agent children
- Display results grouped by agent node:
  ```
  Agent: "Research Agent" — completed (2.1s)
    Helpfulness: 0.85 ▸ (expand for reasoning)
    Factual Accuracy: 0.92 ▸

  Agent: "Summary Agent" — completed (1.3s)
    Conciseness: true ▸
  ```
- Reuse existing score display CSS (`.scores-list`, `.score-item`, `.score-header`)
- Loading state while evals are running

### New props
- `evalsEnabled?: boolean` — show loading indicator on Evals tab when evals are running
- Auto-switch to Evals tab when eval results arrive (if `evalsEnabled`)

---

## Part 7: Wire Everything in +page.svelte

- Pass `evalsEnabled` to InfoPanel
- Pass `allEvaluations={data.evaluations}` to FullscreenNodeModal (line 1744)
- When loading a flow from DB (line 554-589), ensure `eval_ids` is read from agent's graph_config entry node alongside `tool_ids`

---

## Key Files

| File | Changes |
|------|---------|
| `frontend/src/routes/+page.server.ts` | Fetch evaluations |
| `frontend/src/routes/+page.svelte` | Eval state, toggle function, checkbox UI in modal, evalsEnabled toggle, runPostExecutionEvals, pass props |
| `frontend/src/routes/InfoPanel.svelte` | Tab system, remove inline eval UI, add Evals tab content |
| `frontend/src/routes/FullscreenNodeModal.svelte` | Eval assignment for complex agent sub-nodes (Svelte 4 style) |
| `frontend/src/routes/AgentBuilder.svelte` | Pass eval_ids through node creation |
| `frontend/src/routes/EvaluationManager.svelte` | No changes — stays as sidebar CRUD panel |

No backend changes needed — `eval_ids` is stored in `graph_config` (JSON column), and the existing `evaluateExecution` endpoint handles running evals.

---

## Verification

1. **Simple agent**: Create agent with evals checked → verify `eval_ids` in graph_config
2. **Complex agent**: Assign evals to sub-nodes and top-level → verify both in graph_config
3. **Flow execution with Evals toggle**: Run flow with Evals on → verify evals auto-run for agent nodes
4. **Evals tab**: After execution, switch to Evals tab → verify scores displayed per agent node, expandable reasoning from LangFuse
5. **Evals toggle off**: Run flow without Evals → verify no evals run, Evals tab shows "No evaluations ran"
6. **Backward compat**: Existing agents/flows without `eval_ids` work normally (defaults to `[]`)
