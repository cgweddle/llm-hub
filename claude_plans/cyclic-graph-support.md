# Plan: Cyclic Graph Support for Multi-Agent Reflection Loops

## Context

The agent executor (`_execute_graph()` in `agent_executor.py`) already has the *skeleton* for cyclic graphs — `is_loop` edges, `loop_counts`, `max_loop_iterations` — but three bugs/gaps make reflection patterns (Writer → Reviewer → Writer) non-functional:

1. **Exit points block loop edges** — line 344 `if node_id in exit_points: continue` kills ALL successor traversal, including loop-back edges
2. **No message history** — looped nodes lose their previous output and the original input on re-entry
3. **No conditional termination** — loops always run to `max_loop_iterations` with no early exit

The frontend also has a fragility: loop detection is position-based (`targetNode.x < sourceNode.x`), which breaks if users rearrange nodes.

## Scope: Complex Agents Only (not Flows)

All changes apply **only** to `AgentExecutor._execute_graph()` — the BFS-based multi-node agent executor.

`FlowExecutor._execute_from_node()` is a separate sequential DAG walker with no cycle support. Reflection loops are an agent pattern. **No changes to `flow_executor.py`.**

## Changes

### 1. Backend: Fix exit-point-blocks-loop-edges (`agent_executor.py` ~line 344)

Move the exit-point check *inside* the successor loop so exit points skip forward edges but still follow loop edges:

```python
# BEFORE (broken):
if node_id in exit_points:
    continue
for successor in adjacency.get(node_id, []):
    ...

# AFTER (fixed):
is_exit = node_id in exit_points
for successor in adjacency.get(node_id, []):
    edge_key = (node_id, successor)
    is_loop_edge = edge_key in loop_edges
    if is_exit and not is_loop_edge:
        continue
    # ... rest of loop/successor logic
```

### 2. Backend: Message history via PydanticAI (`agent_executor.py`)

Use PydanticAI's built-in `message_history` so each node retains proper conversation turns across loop iterations.

**Data structures** — add alongside `node_outputs`:
```python
node_messages: Dict[str, list] = {}  # node_id → result.all_messages() from last run
```

**Changes to `_run_pydanticai_node()`** (line 412-446):
- Accept optional `message_history` parameter
- Pass to `agent.run(node_input, message_history=message_history)`
- Return tuple `(output_str, result.all_messages())` instead of just the string

**Changes to `_run_sub_agent()`** (line 387-402):
- Accept optional `message_history` parameter
- Pass through to `_run_pydanticai_node()`
- Return tuple `(output_str, messages)`

**Changes to BFS loop in `_execute_graph()`**:
- After each node runs: `node_messages[node_id] = messages`
- When a node re-enters via loop edge: `message_history=node_messages.get(node_id)`
- First iteration: `message_history=None` → identical to current behavior (backward-compatible)

**What the Writer sees on iteration 2:**
```
[system] You are a technical writer...
[user] Write a blog post about X          ← original input
[assistant] Here's my draft: ...           ← Writer's iteration 1 output (from message_history)
[user] Please revise: <reviewer feedback>  ← new input via loop edge
```

### 3. Backend: Conditional routing via PydanticAI structured output unions (`agent_executor.py`)

Use **PydanticAI's idiomatic union output types** for conditional routing. Nodes with multiple output paths define union members — one per path. The executor uses `isinstance()` to route to the matching edge.

**Node config gains `output_paths`:**
```json
"reviewer": {
  "agent_type": "pydanticai",
  "system_prompt": "Review the draft for clarity and accuracy.",
  "output_paths": {
    "revise": "Choose when the draft needs improvement. Provide feedback on what to fix.",
    "approve": "Choose when the draft meets all criteria. Return the final approved text."
  }
}
```

**Edges gain `output_path`:**
```json
{"from_node": "reviewer", "to_node": "writer", "output_path": "revise", "is_loop": true}
{"from_node": "reviewer", "to_node": "formatter", "output_path": "approve", "is_loop": false}
```

**Executor behavior for nodes with `output_paths`:**
1. Dynamically build one Pydantic model per path, then a union:
   ```python
   class Revise(BaseModel):
       """Choose when the draft needs improvement. Provide feedback on what to fix."""
       content: str

   class Approve(BaseModel):
       """Choose when the draft meets all criteria. Return the final approved text."""
       content: str

   # PydanticAI treats each union member as a separate tool the LLM can call
   output_type = Revise | Approve
   ```
2. Create agent with `Agent(output_type=output_type)` — PydanticAI presents each union member as a separate output tool, so the LLM actively chooses which one to call
3. Auto-append routing instructions to system prompt listing the available paths and their descriptions
4. After execution, use `isinstance(result.output, Revise)` to determine which path was chosen
5. Map the class name back to the path name (lowercase) → only follow edges whose `output_path` matches
6. Store `result.output.content` as the node's output string

**Why union types over a single model with a `path` field:** PydanticAI treats union members as separate tools the LLM can call. This is more reliable than asking the LLM to fill a `Literal["revise", "approve"]` field — each union member has its own docstring and schema, giving the LLM clear affordances. This is PydanticAI's recommended pattern for conditional hand-off.

**Nodes WITHOUT `output_paths`:** Behave exactly as today — all successor edges are followed. Backward-compatible.

**Loop termination:** Happens naturally. When Reviewer picks `Approve`, the forward edge fires and the loop edge doesn't. `max_loop_iterations` remains as safety backstop for misconfigured graphs.

### 4. Frontend: Replace position-based loop detection (`agentGraphBuilder.ts` lines 67-79)

Replace `targetNode.position.x < sourceNode.position.x` with explicit `edge.data?.isLoop ?? false`. The UI sets this flag.

### 5. Frontend: Output path handles on agent nodes (`ToolNode.svelte`)

Follow the existing pattern from lines 282-332 where `output_schema.properties` renders per-property handles. For agent nodes with `output_paths`:

- Render one source `Handle` per path name with `id={pathName}`
- Label each handle with the path name (green output style)
- Position vertically: `top: 60 + index * 35px`
- No output paths → single default "Output" handle (backward-compatible)

`agentGraphBuilder.ts` uses `edge.sourceHandle` to populate `output_path` on the edge config.

### 6. Frontend: Output path config in fullscreen modal (`FullscreenNodeModal.svelte`)

Add an "Output Paths" section after the existing Tools section (line ~673) in the agent edit mode:
- Each path has a name field and a description/instruction field
- Add/remove buttons for paths
- Default: no output paths (single output, current behavior)
- Stored on `node.data.output_paths` as `Record<string, string>` (name → description)
- On save, written into `graph_config.nodes[entry_point].output_paths` (extends the save handler at line 78-89)

### 7. Frontend: Auto-detect cycles and style loop edges (`AgentBuilder.svelte`)

In `onConnect`:
- DFS from `target` following existing forward edges. If it reaches `source` → auto-set `edge.data.isLoop = true`
- Style loop edges: dashed, amber color
- Non-loop edges: solid, green (current style)

### 8. Frontend: Add `max_loop_iterations` config (`AgentBuilder.svelte`)

Numeric input (default 5) in the save dialog. Pass through to `graph_config.max_loop_iterations`.

## Files to Modify

| File | Change |
|------|--------|
| `src/executors/agent_executor.py` | Fix exit-point logic, PydanticAI `message_history`, output-path routing with dynamic structured output unions |
| `frontend/src/lib/agentGraphBuilder.ts` | Replace position-based loop detection with `edge.data.isLoop`, map `edge.sourceHandle` to `output_path` |
| `frontend/src/routes/ToolNode.svelte` | Dynamic output handles for nodes with `output_paths` |
| `frontend/src/routes/FullscreenNodeModal.svelte` | Output paths config UI (name + description fields, add/remove) in agent edit mode |
| `frontend/src/routes/AgentBuilder.svelte` | Cycle auto-detection, loop edge styling, `max_loop_iterations` input |
| `frontend/src/lib/api.ts` | Add `output_paths` to `SubAgentNodeConfig`, add `output_path` to `AgentEdgeConfig` |

## Files That Need No Changes

- `src/database/database_setup.py` — `graph_config` is a JSON column, no schema migration needed
- `src/executors/flow_executor.py` — flows don't support cycles

## Implementation Order

1. Backend: Update `_run_pydanticai_node` / `_run_sub_agent` return types + `message_history` param
2. Backend: Add `node_messages` accumulation in `_execute_graph()`
3. Backend: Fix exit-point-blocks-loop-edges (restructure lines 344-364)
4. Backend: Output-path routing — dynamic Pydantic union models, prompt injection, edge filtering by `output_path`
5. Unit tests for steps 1-4
6. Frontend: `api.ts` — add `output_paths` and `output_path` to interfaces
7. Frontend: `agentGraphBuilder.ts` — explicit `isLoop` flag, `sourceHandle` → `output_path`
8. Frontend: `ToolNode.svelte` — dynamic output handles per path
9. Frontend: `FullscreenNodeModal.svelte` — output paths config UI
10. Frontend: `AgentBuilder.svelte` — cycle detection, styling, max iterations input
11. Integration test: manual Writer/Reviewer reflection loop

## Verification

1. **Unit tests** (`tests/test_agent_executor_loops.py`):
   - Message history: mock sub-agents, verify 2nd-iteration receives `message_history` from 1st
   - Exit point + loop: verify loop fires from exit-point nodes
   - Output-path routing: mock agent returning union type, verify only matching edges followed via `isinstance()`
   - Safety limit: verify `max_loop_iterations` stops runaway loops
   - Backward compat: single-node and simple DAG agents produce identical results

2. **Manual integration**: Create Writer/Reviewer composed agent with "revise"/"approve" output paths, verify loop iterates with context and terminates on "approve"

3. **Frontend check**: `cd frontend && npm run check` for TypeScript validity
