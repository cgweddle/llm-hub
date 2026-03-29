# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

LLM Hub is a visual workflow builder for LLM-powered applications. Users create **tools** (Python scripts), **agents** (LLM-powered executors), and **flows** (DAGs connecting tools/agents), then execute them from a drag-and-drop canvas UI.

## Development Commands

### Backend (Python)
```bash
# Start the FastAPI backend (auto-reloads on changes)
python start_backend.py
# Backend runs on http://127.0.0.1:8000

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_pydanticai_components.py

# Run a specific test
pytest tests/test_retry.py::TestRetryDecorator::test_successful_first_attempt -v

# Database setup (SQLite for dev, PostgreSQL for prod)
python src/database/database_setup.py setup
python src/database/database_setup.py info
```

### Frontend (SvelteKit + Svelte 5)
```bash
cd frontend
nvm use          # Node v25.6.0
npm install
npm run dev      # http://localhost:5173
npm run check    # TypeScript checking
npm run build    # Production build
```

### Running Both Together
Start the backend first (`python start_backend.py`), then the frontend (`cd frontend && npm run dev`). The frontend talks to the backend at `http://localhost:8000`.

## Architecture

### Backend Stack
- **FastAPI** API server (`src/api/backend.py`) — single file, all endpoints
- **SQLAlchemy** ORM with SQLite (dev) / PostgreSQL (prod) (`src/database/database_setup.py` for models, `database_setup.py` also has CLI for DB management)
- **Three executor types** following a factory pattern:
  - `src/executors/tool_executor.py` — executes Python script tools (optionally in conda envs)
  - `src/executors/flow_executor.py` — traverses flow graph_config DAG, runs tool and agent nodes in sequence
  - `src/executors/agent_executor.py` — unified BFS graph executor for all agents (single-node and multi-agent), all nodes execute via PydanticAI, supports streaming and cyclic graphs
- **Agent factories**: `src/factories/pydanticai_agent_factory.py` (PydanticAI — sole execution path for all agent types)
- **Tool factory**: `src/factories/python_script_tool_factory.py` — AST-parses Python scripts, extracts type schemas, creates Tool DB records
- **AI integrations**: `src/ai_integrations/` — LLM-powered code generation and prompt generation with streaming. `generate_agent_system_prompt.py` has two generators: `generate_system_prompt_stream` (uses `agent_prompt_gen` DB template) and `generate_user_prompt_stream` (uses `agent_user_prompt_gen` DB template, receives the generated system prompt as context). Prompt templates live in `src/prompts/*.system.md` / `*.user.md` and are uploaded via `python src/prompts/upload_prompts.py`.

### Frontend Stack
- **SvelteKit** with **Svelte 5**, **TypeScript**, **Tailwind CSS v4**, **shadcn-svelte** components
- **@xyflow/svelte** for the visual flow canvas (drag-and-drop nodes and edges)
- **CodeMirror** for Python code editing in the browser
- Main page is a single large file: `frontend/src/routes/+page.svelte`
- Info panel: `frontend/src/routes/InfoPanel.svelte` — execution tree viewer with LangFuse trace integration
- API client: `frontend/src/lib/api.ts` — all backend communication, typed interfaces
- Graph conversion: `frontend/src/lib/flowBuilder.ts` (XYFlow nodes/edges → graph_config for tool flows), `frontend/src/lib/agentGraphBuilder.ts` (for composed agents)
- Auto-layout: `frontend/src/lib/elkLayout.ts` (uses elkjs for node positioning)

### Data Model (SQLAlchemy)

All models are defined in `src/database/database_setup.py`. CRUD helpers live in `src/database/database.py`.

#### Association Tables
- **agent_tool_association** (`agent_id`, `tool_id`) — many-to-many between Agents and Tools
- **agent_flow_association** (`agent_id`, `flow_id`) — many-to-many between Agents and Flows

#### Tables

**users**
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| username | String(50) | unique, not null |
| email | String(120) | unique, not null |
| password_hash | String(255) | not null |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |
| is_active | Boolean | default True |
| **Relationships** | | agents, flows, executions |

**agents**
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| user_id | Integer FK → users.id | not null |
| name | String(100) | not null |
| description | Text | |
| graph_config | JSON | not null — unified agent workflow graph (see below) |
| output_schema | JSON | optional structured output schema |
| is_public | Boolean | default False |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |
| **Relationships** | | user, tools (M2M), flows (M2M), executions |

**Agent graph_config schema** — every agent (simple or complex) uses this structure:
```json
{
  "nodes": {
    "<node_id>": {
      "agent_type": "pydanticai|planning|react|reflection|custom",
      "name": "Agent Name",
      "system_prompt": "...",
      "user_prompt": "{input}",
      "llm_provider": "config name from ~/.llm_hub/config.yaml",
      "tool_ids": [1, 2],
      "output_paths": { "approve": {"description": "Output meets quality bar", "return_behavior": "previous_output"}, "revise": {"description": "Needs improvement", "return_behavior": "node_output"} }
    }
  },
  "edges": [
    { "from_node": "a", "to_node": "b", "is_loop": false }
  ],
  "entry_point": "<node_id>",
  "exit_points": ["<node_id>"],
  "max_loop_iterations": 5
}
```
Simple agents have a single node (`"main"`). Multi-agent workflows have multiple nodes with edges (including `is_loop: true` for cyclic connections). The executor uses BFS traversal with `max_loop_iterations` to prevent infinite loops.

**User prompt templates** — the `user_prompt` field supports `{input}` (replaced with node input text, required) and `{message_history}` (replaced with serialized conversation history from previous nodes). The resolved user prompt is the user message sent to PydanticAI. Template resolution: `src/utils/prompt_template.py`, message serialization: `src/utils/message_serializer.py`.

**Output path return behavior** — each output path has a `return_behavior`: `"node_output"` (default, returns the LLM's response) or `"previous_output"` (returns the node's input unchanged — useful for reflection agents approving the previous node's work). Backward compatible with old string-format output paths.

**tools**
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| user_id | Integer FK → users.id | not null |
| name | String(100) | not null |
| description | Text | |
| tool_type | String(50) | not null — `function` or `custom` |
| main_function | String(100) | entry-point function name |
| function_code | Text | extracted main function source |
| helper_functions | JSON | `{"name": "code"}` map |
| script_code | Text | full original Python script |
| input_schema | JSON | AST-parsed input parameter types |
| output_schema | JSON | AST-parsed output structure |
| api_config | JSON | for API-based tools |
| parameters | JSON | legacy — prefer input_schema |
| is_public | Boolean | default False |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |
| **Relationships** | | agents (M2M) |

**flows**
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| user_id | Integer FK → users.id | not null |
| name | String(100) | not null |
| description | Text | |
| graph_config | JSON | not null — DAG with `nodes`, `edges`, `entry_point`, `exit_points` |
| entry_point | String(100) | not null |
| exit_points | JSON | list of exit node names |
| conda_env | String(500) | optional conda environment path |
| is_public | Boolean | default False |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |
| **Relationships** | | user, agents (M2M), executions |

**executions** — self-referencing tree (replaces the old `executions` + `messages` tables)
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| parent_id | Integer FK → executions.id | nullable — NULL for top-level executions |
| user_id | Integer FK → users.id | not null |
| agent_id | Integer FK → agents.id | nullable |
| flow_id | Integer FK → flows.id | nullable |
| tool_id | Integer FK → tools.id | nullable |
| execution_type | String(50) | not null — `flow`, `agent`, `tool`, `tool_call`, `tool_result`, `trigger` |
| node_id | String(100) | node identifier from graph_config |
| name | String(200) | human-readable name |
| sequence | Integer | execution order within parent |
| input_data | JSON | |
| output_data | JSON | |
| status | String(20) | default `running` — `running` / `completed` / `failed` |
| error_message | Text | |
| started_at | DateTime | default now |
| completed_at | DateTime | |
| execution_metadata | JSON | cost, model name, token counts, etc. |
| langfuse_trace_id | String(200) | LangFuse trace ID for cross-referencing agent internals |
| **Relationships** | | parent (self), children (self, ordered by sequence), user, agent, flow, tool |

Every execution — flow, agent, tool, trigger — is a row. Parent-child relationships form a tree:
```
Flow Execution (parent=NULL, type='flow', flow_id=5)
  ├─ Trigger (parent=1, type='trigger', seq=0)
  ├─ Tool (parent=1, type='tool', seq=1, tool_id=10)
  └─ Agent (parent=1, type='agent', seq=2, agent_id=3, langfuse_trace_id='abc123')
```
Standalone agent executions have `parent_id=NULL, type='agent'`. The `messages` table no longer exists.

**prompts**
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| prompt_name | Text | |
| system_prompt | Text | |
| user_prompt | Text | |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |

### Execution Recording & LangFuse Tracing

**Execution tree** — Both `FlowExecutor` and `AgentExecutor` record executions to the self-referencing `executions` table:
- `FlowExecutor.execute_flow()` creates a top-level `Execution(type='flow')` and child records for each node (trigger, tool, agent) as it traverses the DAG
- `AgentExecutor.execute_agent()` creates a top-level `Execution(type='agent')` for standalone agent runs
- `AgentExecutor.execute_agent_node()` is called by FlowExecutor with a `parent_execution` — records agent details under the flow's tree
- For single-node agents, no redundant child is created; the parent execution record suffices
- CRUD helpers: `create_execution()`, `update_execution()`, `get_execution_by_id()`, `get_user_executions()` in `src/database/database.py`

**LangFuse integration** — Internal agent telemetry (LLM calls, tool calls, system prompts) is captured by LangFuse:
- `Agent.instrument_all()` in `agent_executor.py` auto-traces all PydanticAI agent runs (all agent types use PydanticAI)
- After each agent node runs, the LangFuse trace ID is captured via `@observe` + `get_current_trace_id()` and stored on the execution record's `langfuse_trace_id` column
- Trace ID is captured even on agent failure (important for debugging)
- `GET /executions/{id}/trace` backend endpoint proxies trace data from LangFuse
- Requires `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` in `.env`

**Info panel** — `frontend/src/routes/InfoPanel.svelte`:
- Toggled via "Info" button at bottom-left of the SvelteFlow canvas
- Resizable height via drag handle (100px–600px)
- Shows the execution tree for the latest flow run: each node with type badge, status dot, duration
- Expandable nodes reveal input/output JSON
- Agent nodes with `langfuse_trace_id` show a "traced" badge; expanding fetches LLM conversation details (system prompt, tool calls, responses) from LangFuse via `GET /executions/{id}/trace`

**API endpoints for executions:**
- `GET /executions?user_id={id}&limit=50&offset=0` — list top-level executions
- `GET /executions/{id}` — full execution tree (recursive children)
- `GET /executions/{id}/trace` — LangFuse trace data for an agent execution

### Key Patterns
- **graph_config** is the central data structure: a JSON object with `nodes` (dict), `edges` (list of from/to mappings), `entry_point`, and `exit_points`. The frontend builds it from the canvas; the backend traverses it for execution.
- **Streaming** uses SSE (Server-Sent Events) via FastAPI `StreamingResponse`. The frontend reads streams with `ReadableStream` reader pattern. Used for agent execution, code generation, and system prompt generation.
- **LLM provider config** is stored in `~/.llm_hub/config.yaml` (not in the repo). Supports Anthropic, OpenAI, Gemini, and LM Studio. The backend masks API keys before sending to the frontend.
- **Agent types**: All agents use `graph_config` — simple agents are single-node graphs, multi-agent workflows are multi-node graphs. Per-node `agent_type` is a behavioral template (`pydanticai`, `planning`, `react`, `reflection`, `custom`) — all execute via PydanticAI; the type determines the default system prompt, not the execution engine. Cyclic edges (`is_loop: true`) enable reflection loops with `max_loop_iterations` safety limit. Nodes can define `output_paths` for conditional routing (e.g., `{"approve": "...", "revise": "..."}`).
- **Agent Builder UI**: The AgentBuilder (`frontend/src/routes/AgentBuilder.svelte`) presents template types as clickable buttons. Clicking opens the same Create Agent modal from `+page.svelte` pre-filled with the template's default system prompt and user prompt. The modal supports both simple agent creation and complex agent node configuration. Agent templates with `defaultSystemPrompt` and `defaultUserPrompt` are defined in `frontend/src/lib/agentTemplates.ts`. The "Generate with AI" button generates both prompts sequentially — first the system prompt, then the user prompt (with awareness of the system prompt to avoid overlap). Opening an existing complex agent enables "Update Agent" mode (vs "Save Complex Agent" for new ones).
- **Tool types**: Python scripts with AST-parsed input/output schemas for flow compatibility validation (`src/validate/tool_compatibility.py`)

### Environment
- Python 3.10 (conda environment `llm-hub`)
- Node v25.6.0 (see `frontend/.nvmrc`)
- Backend `.env` requires `DATABASE_URL` (defaults to `sqlite:///database/llm_hub.db`)
- Backend `.env` LangFuse config: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (e.g. `http://localhost:3000` for self-hosted)
- Frontend `.env` has its own `DATABASE_URL` for Drizzle/better-sqlite3 (Lucia auth)
- LangFuse runs locally via Podman (`podman compose up` in the langfuse repo)
