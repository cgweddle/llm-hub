# Plan: Visual Agent Builder with SvelteFlow

## Summary

Replace the simple "Create Agent" modal with a full visual SvelteFlow-based agent builder. Users will drag-and-drop agent type nodes (Planning, React, Reflection) onto a canvas, connect them into workflows, assign tools, and save the result as a composed agent.

**Key architectural decisions**:
- Add a **mode toggle** to the existing page rather than a new route. This reuses the SvelteFlow canvas, sidebar, and controls infrastructure already in `+page.svelte`.
- **Both creation paths coexist**: The simple "Create New Agent" modal remains for quick single agents. A new "Compose Agent" button switches to the visual builder for multi-agent workflows.
- Composed agents are saved to the existing Agent table using the `agent_metadata` JSON field to store the graph config.

---

## Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `frontend/src/lib/stores/builderMode.ts` | Svelte store for `'flow' \| 'agent'` mode toggle |
| 2 | `frontend/src/lib/agentTemplates.ts` | Agent type definitions (Planning/React/Reflection) with colors, icons, default system prompts |
| 3 | `frontend/src/routes/AgentBuilderNode.svelte` | New SvelteFlow node component with type-based theming (blue/green/purple), tool assignment display, and config button |
| 4 | `frontend/src/routes/AgentConfigModal.svelte` | Per-node config modal: name, system prompt, tool checkboxes, LLM provider |
| 5 | `frontend/src/lib/agentGraphBuilder.ts` | Converts agent builder nodes/edges → `graph_config` for `agent_metadata` (mirrors `flowBuilder.ts`) |
| 6 | `frontend/src/routes/LoopEdge.svelte` | Animated dashed edge for feedback loops (backward connections) |

## Files to Modify

| # | File | Changes |
|---|------|---------|
| 1 | `frontend/src/routes/+page.svelte` | Mode toggle button, conditional sidebar (agent types vs tools/flows), separate `agentNodes`/`agentEdges` state, agent drop handler, save composed agent dialog, register `agentBuilderNode` + `loopEdge` types |
| 2 | `frontend/src/lib/api.ts` | Add `ComposedAgentMetadata` interface, update `AgentCreateData` to support `agent_type: 'composed'` + `agent_metadata` |
| 3 | `src/executors/agent_executor.py` | Add `execute_composed_agent()` method that orchestrates sub-agents following the graph config |
| 4 | `src/api/backend.py` | Ensure `agent_metadata` is accepted and stored on agent creation |

---

## Implementation Steps

### Step 1 — Mode toggle infrastructure
- Create `builderMode.ts` store (`'flow' | 'agent'`)
- In `+page.svelte`: import store, add toggle button in the flow controls bar
- Add separate state arrays: `agentNodes`/`agentEdges` alongside existing `nodes`/`edges`
- Use reactive bindings to swap which arrays the SvelteFlow canvas uses based on mode

### Step 2 — Agent type templates
- Create `agentTemplates.ts` with three templates:
  - **Planning** (blue `#3b82f6`, 📋) — task decomposition
  - **React** (green `#10b981`, ⚡) — reason + act loop with tools
  - **Reflection** (purple `#a855f7`, 🔍) — self-critique and improvement
- Each template includes: type key, display name, description, default system prompt, color, icon

### Step 3 — AgentBuilderNode component
- Create `AgentBuilderNode.svelte` based on `ToolNode.svelte` patterns but simplified:
  - Type-colored header bar (blue/green/purple)
  - Icon + name display
  - List of assigned tools (as small chips)
  - "Configure" button → opens AgentConfigModal
  - Input handle (left, blue) + Output handle (right, matching type color)
- Register as `agentBuilderNode` in `nodeTypes` map in `+page.svelte`

### Step 4 — Sidebar conditional rendering
- When mode is `'agent'`:
  - Show **"Agent Types"** section with three draggable items (Planning, React, Reflection) styled with colored left borders
  - Show **"Available Tools"** section (read-only list for reference — tools are assigned per-node via config modal)
  - Show **"Save Agent"** button instead of flow save controls
  - Hide flows section and tool creation
  - Keep "Create New Agent" button for simple agents; add "Compose Agent" button to enter agent builder mode
- When mode is `'flow'`: show existing sidebar (unchanged, including simple agent creation modal)

### Step 5 — Canvas drop handler for agent types
- Modify the `ondrop` handler to check `builderMode`:
  - In `'agent'` mode: read `event.dataTransfer.getData('agent-type')`, call `addAgentNode(type, position)`
  - In `'flow'` mode: existing behavior (unchanged)
- `addAgentNode()` creates a node with `type: 'agentBuilderNode'` and data from the template

### Step 6 — AgentConfigModal
- Create `AgentConfigModal.svelte`:
  - Name text input
  - System prompt textarea (with "Generate with AI" option, reusing existing streaming pattern)
  - Tool selection checkboxes (from `data.tools`)
  - LLM provider dropdown (from `llmProviders`)
  - Save/Cancel buttons
- Wire up: double-click or configure button on `AgentBuilderNode` → opens modal → saves updates back to node data

### Step 7 — Loop edge support
- Create `LoopEdge.svelte`: curved bezier path with animated dashed stroke (purple)
- Auto-detect loops: when a new connection's target is positionally left of its source, use `type: 'loop'`
- Register `loop` edge type in `edgeTypes` map
- Update `onconnect` handler to detect loop direction

### Step 8 — Agent graph builder & save
- Create `agentGraphBuilder.ts`:
  - `buildAgentGraph(nodes, edges)` → returns `ComposedAgentMetadata`
  - Structure: `{ graph_config: { nodes: {}, edges: [], entry_point, exit_points }, is_composed: true }`
- Add save dialog in `+page.svelte`: name + description fields
- On save: call `createAgent()` with `agent_type: 'composed'` and graph config in `agent_metadata`
- After save: add new agent to sidebar agent list, clear agent canvas

### Step 9 — API & backend updates
- `api.ts`: Add `ComposedAgentMetadata` type, ensure `createAgent` passes `agent_metadata`
- `backend.py`: Verify `agent_metadata` flows through to the database (likely already supported via existing JSON field)
- `agent_executor.py`: Add composed agent execution:
  - Parse `graph_config` from `agent_metadata`
  - Walk nodes following edges from `entry_point` to `exit_points`
  - For each node: create a temporary sub-agent, execute it, pass output to next node
  - Handle loops with a max-iteration guard

---

## Verification

1. **Frontend mode toggle**: Click mode toggle → sidebar switches between flow/agent views, canvas clears to the other mode's state
2. **Drag & drop**: Drag each agent type → colored node appears on canvas with correct theming
3. **Node configuration**: Click configure → modal opens → edit name/prompt/tools/LLM → save reflects on node
4. **Edge connections**: Connect Planning→React→Reflection → edges render with arrows. Connect Reflection→React backward → loop edge renders with animation
5. **Save composed agent**: Fill canvas with connected agents → click Save → enter name → agent appears in sidebar agent list
6. **Use in flow**: Switch to flow mode → drag the composed agent from sidebar onto flow canvas → it appears as a regular agent node
7. **Backend execution**: Execute a flow containing a composed agent → `agent_executor` routes to `execute_composed_agent` → sub-agents run in sequence → result returns
