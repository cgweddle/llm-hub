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
  - `src/executors/evaluation_executor.py` — LLM-as-a-judge evaluator, runs a judge LLM via PydanticAI structured output and posts scores to LangFuse
- **Agent factories**: `src/factories/pydanticai_agent_factory.py` (PydanticAI — sole execution path for all agent types)
- **Tool factory**: `src/factories/python_script_tool_factory.py` — AST-parses Python scripts, extracts type schemas, creates Tool DB records
- **AI integrations**: `src/ai_integrations/` — LLM-powered code generation and prompt generation with streaming. `generate_agent_system_prompt.py` has two generators: `generate_system_prompt_stream` (uses `agent_prompt_gen` DB template) and `generate_user_prompt_stream` (uses `agent_user_prompt_gen` DB template, receives the generated system prompt as context). `generate_eval_prompt.py` generates judge system prompts for evaluations (uses `eval_prompt_gen` DB template). All generators accept optional `additional_instructions` for user-provided guidance. Prompt templates live in `src/prompts/*.system.md` / `*.user.md` and are uploaded via `python src/prompts/upload_prompts.py`.

### Frontend Stack
- **SvelteKit** with **Svelte 5**, **TypeScript**, **Tailwind CSS v4**, **shadcn-svelte** components
- **Svelte 5 runes only** — all components must use `$state`, `$derived`, `$effect`, `$props`, `$bindable`. Do NOT use Svelte 4 patterns (`$:` reactive statements, `export let`, `on:click`, `svelte/store` writable/readable). Use `onclick` not `on:click`, `onchange` not `on:change`, etc.
- **@xyflow/svelte** for the visual flow canvas (drag-and-drop nodes and edges)
- **CodeMirror** for Python code editing in the browser
- Main page is a single large file: `frontend/src/routes/+page.svelte` (note: this file still uses some Svelte 4 patterns — new code in this file should use Svelte 5 runes where possible, but be aware of the mixed context)
- Info panel: `frontend/src/routes/InfoPanel.svelte` — execution tree viewer with LangFuse trace and evaluation score integration
- Evaluation manager: `frontend/src/routes/EvaluationManager.svelte` — CRUD for LLM-as-a-judge evaluation types, with collapsible "Generate with AI" section for judge prompt generation (same pattern as agent modal)
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
| required_packages | JSON | list of PyPI package names detected by pigar at create/update; nullable; informational (no auto-install) |
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

**evaluations** — reusable LLM-as-a-judge evaluation type definitions
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| user_id | Integer FK → users.id | not null |
| name | String(100) | not null, e.g. "Helpfulness" |
| description | Text | human-readable description |
| judge_system_prompt | Text | not null — system prompt for the judge LLM |
| scoring_rubric | Text | rubric text included in the judge's user message |
| score_type | String(20) | `NUMERIC`, `CATEGORICAL`, or `BOOLEAN` |
| score_categories | JSON | for CATEGORICAL, e.g. `["good","bad","neutral"]` |
| llm_provider | String(100) | not null — name from `~/.llm_hub/config.yaml` |
| is_public | Boolean | default False |
| created_at / updated_at | DateTime | standard timestamps |

**evaluation_results** — lightweight pointer to LangFuse (scores stored in LangFuse, not here)
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| evaluation_id | FK → evaluations.id | not null |
| execution_id | FK → executions.id | not null |
| user_id | FK → users.id | not null |
| langfuse_trace_id | String(200) | the execution's trace ID |
| langfuse_score_id | String(200) | ID from `langfuse_client.score()` — primary reference to result in LangFuse |
| status | String(20) | `running` / `completed` / `failed` |
| error_message | Text | only on failure |
| created_at | DateTime | default now |
| completed_at | DateTime | |

### Execution Recording & LangFuse Tracing

**Execution tree** — Both `FlowExecutor` and `AgentExecutor` record executions to the self-referencing `executions` table:
- `FlowExecutor.execute_flow(initial_input, conda_env, execution_id=None)` creates a top-level `Execution(type='flow')` and child records for each node (trigger, tool, agent) as it traverses the DAG. The optional `execution_id` parameter reuses a pre-existing row (used by the Celery task path — see "Production Execution" below) instead of creating a new one.
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
- `GET /executions/{id}/scores` — LangFuse score data for an execution's trace (proxy)
- `GET /executions/{id}/evaluations` — evaluation result records for an execution
- `POST /executions/{id}/evaluate` — run evaluation(s) against an execution

### LLM-as-a-Judge Evaluations

Users define custom evaluation types (judge prompt + rubric + score type) and run them against agent executions. **LangFuse is the single source of truth for evaluation results** — the local `evaluation_results` table only stores a `langfuse_score_id` reference, not the actual score or reasoning.

**Evaluation flow:**
1. User creates an evaluation type via `EvaluationManager.svelte` (stored in `evaluations` table)
2. User clicks "eval" on a traced execution node in `InfoPanel.svelte`, selects criteria, clicks "Run"
3. `POST /executions/{id}/evaluate` calls `EvaluationExecutor.evaluate()` for each selected evaluation
4. The executor builds a judge prompt from the evaluation's `judge_system_prompt` + execution input/output + `scoring_rubric`
5. PydanticAI runs the judge LLM with structured output (`NumericJudgeResponse`, `BooleanJudgeResponse`, or `CategoricalJudgeResponse`)
6. Score is posted to LangFuse via `langfuse_client.score(trace_id=..., name=..., value=..., comment=reasoning)`
7. `evaluation_results` row updated with `langfuse_score_id` and `status='completed'`
8. InfoPanel fetches scores from LangFuse via `GET /executions/{id}/scores` proxy endpoint

**API endpoints for evaluations:**
- `POST /evaluations/?user_id={id}` — create evaluation type
- `GET /evaluations/?user_id={id}` — list user's evaluations (+ public ones)
- `GET /evaluations/{id}` — get single evaluation
- `PATCH /evaluations/{id}` — update evaluation
- `DELETE /evaluations/{id}` — delete evaluation

**LLM provider resolution:** `src/utils/llm_config.py` has `resolve_model_name(llm_provider)` — a shared utility that resolves a provider name to a `"provider:model"` string for PydanticAI and sets API key env vars. The source it reads from depends on deployment mode (see below).

### LLM Provider Configuration (Local vs Hosted)

The system supports two mutually-exclusive config backends, selected by the `ENVIRONMENT` env var:

- **LOCAL** (default, unset or `ENVIRONMENT=LOCAL`) — providers are read from and written to `~/.llm_hub/config.yaml` (YAML file, plaintext API keys). Used for local dev. The `llm_provider_configs` DB table is **not created** in this mode — `DatabaseManager.create_tables()` filters it out via `is_local()` so dev SQLite DBs stay lean.
- **HOSTED** (`ENVIRONMENT=HOSTED`, set in `deploy/podman-compose.yml`) — providers are stored per-user in the `llm_provider_configs` Postgres table with API keys encrypted at rest via Fernet (symmetric). Enables multi-tenant production where each user manages their own credentials through the UI.

**Environment detection:** `src/utils/environment.py` exposes `is_hosted()` / `is_local()` — both `llm_config.py` and `database_setup.py` branch on these.

**Encryption (HOSTED only):** `src/database/database.py` has `_get_fernet()`, `_encrypt_api_key()`, `_decrypt_api_key()` using `cryptography.fernet.Fernet` with key from the `LLM_CONFIG_ENCRYPTION_KEY` env var. The key is a base64-encoded 32-byte Fernet key (generate with `Fernet.generate_key()`). Passed through via GitHub Actions secret → `podman-compose.yml` → backend container. API keys are encrypted on write and decrypted on read; the plaintext never touches disk.

**Data model:** `LLMProviderConfig` (in `database_setup.py`) — per-user row with `name`, `provider`, `model`, `api_key_encrypted` (Text), `base_url`. Related to `User` via `llm_configs` back-reference.

**Caveats:**
- Do NOT mount the `llm-config` volume or forward provider API keys as env vars into the HOSTED backend container — credentials must come from the DB.
- Rotating `LLM_CONFIG_ENCRYPTION_KEY` invalidates all existing encrypted rows; plan a re-encryption migration if the key ever changes.
- `resolve_model_name()` transparently picks the right backend, so caller code (executors, factories) does not need to branch on environment.

### Production Execution (Celery + Podman)

Execution behavior splits on `ENVIRONMENT` (default `local`):

- **`ENVIRONMENT=local`** (dev default): `POST /flows/{id}/execute` runs `FlowExecutor.execute_flow()` synchronously in the API process. No Redis, no Celery, no Podman required. This is the existing behavior and all dev workflows assume it.
- **`ENVIRONMENT=production`**: The endpoint pre-creates an `Execution(status='pending')` row, dispatches `execute_flow_task.delay(...)` to Redis, and returns HTTP 202 with `{execution_id, status}`. Frontend polls `GET /executions/{id}` for status transitions (`pending` → `running` → `completed`).

Single gate in `_is_production()` at `src/api/backend.py:934`. The Celery task import is **lazy** (inside the production branch) so local dev doesn't require celery/redis installed.

**Celery app** (`src/celery_app.py`) — shared between API and worker. Reads `CELERY_BROKER_URL` (default `redis://localhost:6379/0`). Configured with `task_acks_late=True`, `worker_prefetch_multiplier=1`, JSON serialization. Autodiscovers tasks from `src.tasks`.

**Celery task** (`src/tasks/flow_tasks.py::execute_flow_task`) — receives `(flow_id, user_id, initial_input, conda_env, execution_id)`. Default behavior: spawns an `llmhub-flow-runner` Podman container via `podman-py` bound to `CONTAINER_HOST` (the host's rootless socket). Falls back to in-worker execution when `FLOW_RUNNER_USE_PODMAN=false` (useful for testing the Celery layer without Podman).

**Flow-runner container** (`deploy/flow-runner/Containerfile`, image `llmhub-flow-runner`):
- Based on `ghcr.io/astral-sh/uv:python3.10-bookworm-slim` (provides python + `uv` on PATH), with `src/` and a **runner-specific** `deploy/flow-runner/requirements.txt` (not the repo-root `requirements.txt`). The runner deps are the minimal set needed by the flow-runner parent process: `pydantic`, `pydantic-ai`, `anthropic`, `openai` (provider SDKs — `pydantic-ai` does not pull these transitively; `openai` also covers LM Studio's OpenAI-compatible API), `sqlalchemy`, `psycopg2-binary`, `httpx`, `python-dotenv`, `langfuse`. No `numpy`/`pandas`/`pyyaml` are pre-installed — tool-specific packages are installed at container startup via `install_required_packages_for_flow` instead. Gemini is not included because the PydanticAI factory only dispatches to Anthropic/OpenAI/LM Studio (Gemini is only supported through the legacy `google.adk` factory, which is not in the flow execution path).
- Entrypoint: `python -m src.tasks.run_flow` (`src/tasks/run_flow.py`)
- Reads `FLOW_RUNNER_FLOW_ID`, `FLOW_RUNNER_USER_ID`, `FLOW_RUNNER_EXECUTION_ID`, `FLOW_RUNNER_INITIAL_INPUT` (JSON), `FLOW_RUNNER_CONDA_ENV` from env vars set by the worker's `podman run -e`
- Opens its own DB session via `DatabaseManager().get_session()`, runs `install_required_packages_for_flow(session, flow_id)` to `uv pip install --user` every package listed in `required_packages` across the flow's tools and agent-attached tools (best-effort — warns but continues on failure), then calls `FlowExecutor.execute_flow(..., execution_id=...)` — **exactly the same code path as local mode**. Tools still run as host-style subprocesses; they don't know they're in a container.
- Spawned with `--rm`, `--read-only`, `--tmpfs /tmp:size=100M`, `--memory=1g`, `--cpus=2`, on the `llmhub-net` network. Unrestricted network (tools/agents need outbound access for external APIs).
- Image is built outside `podman-compose.yml` (via `podman build` in `.github/workflows/deploy.yml`) because it's ephemeral, not a long-running service.

**Services in `deploy/podman-compose.yml`** (production-only additions):
- `redis` — broker/result backend
- `worker` — same image as `backend`, entrypoint `/start_worker.sh` (`deploy/backend/start_worker.sh`) running `celery -A src.celery_app:celery_app worker`. Mounts host's rootless Podman socket (`${XDG_RUNTIME_DIR}/podman/podman.sock:/run/podman/podman.sock`) so it can spawn flow-runner containers.
- `llmhub-net` and `llm-config` are pinned with explicit `name:` in compose (no project prefix) so the dynamically-spawned flow-runner can reference them by stable name.

**Env var forwarding** (`_FORWARDED_ENV_VARS` in `flow_tasks.py`): worker → flow-runner only forwards `DATABASE_URL`, `SQL_DEBUG`, `ENVIRONMENT`, `LANGFUSE_*`. LLM provider credentials are NOT forwarded — in production they live in Postgres (per-user), looked up at runtime via `DATABASE_URL`. Local dev still uses `~/.llm_hub/config.yaml`.

**Safety net**: if the flow-runner container crashes or exits non-zero without updating the DB, `_mark_failed_if_still_running(execution_id, reason)` updates the Execution row to `status='failed'` so the frontend doesn't see it stuck.

**When modifying flow execution**: any change to `FlowExecutor`, `ToolExecutor`, or `AgentExecutor` automatically applies to both local and production — the flow-runner runs the same code. No branching on environment inside the executors.

### Key Patterns
- **graph_config** is the central data structure: a JSON object with `nodes` (dict), `edges` (list of from/to mappings), `entry_point`, and `exit_points`. The frontend builds it from the canvas; the backend traverses it for execution.
- **Streaming** uses SSE (Server-Sent Events) via FastAPI `StreamingResponse`. The frontend reads streams with `ReadableStream` reader pattern. Used for agent execution, code generation, and system prompt generation.
- **LLM provider config** in **local dev** is stored in `~/.llm_hub/config.yaml` (not in the repo). Supports Anthropic, OpenAI, Gemini, and LM Studio. The backend masks API keys before sending to the frontend. **Production** (`ENVIRONMENT=production`) stores per-user credentials in Postgres instead — don't assume the YAML file exists in production code paths and don't mount `llm-config` volumes into production-only containers.
- **LLM provider config** has two backends selected by `ENVIRONMENT`: LOCAL reads/writes `~/.llm_hub/config.yaml`; HOSTED reads/writes the `llm_provider_configs` Postgres table with Fernet-encrypted API keys. See "LLM Provider Configuration (Local vs Hosted)" above. Supports Anthropic, OpenAI, Gemini, and LM Studio. The backend masks API keys before sending to the frontend.
- **Agent types**: All agents use `graph_config` — simple agents are single-node graphs, multi-agent workflows are multi-node graphs. Per-node `agent_type` is a behavioral template (`pydanticai`, `planning`, `react`, `reflection`, `custom`) — all execute via PydanticAI; the type determines the default system prompt, not the execution engine. Cyclic edges (`is_loop: true`) enable reflection loops with `max_loop_iterations` safety limit. Nodes can define `output_paths` for conditional routing (e.g., `{"approve": "...", "revise": "..."}`).
- **Agent Builder UI**: The AgentBuilder (`frontend/src/routes/AgentBuilder.svelte`) presents template types as clickable buttons. Clicking opens the same Create Agent modal from `+page.svelte` pre-filled with the template's default system prompt and user prompt. The modal supports both simple agent creation and complex agent node configuration. Agent templates with `defaultSystemPrompt` and `defaultUserPrompt` are defined in `frontend/src/lib/agentTemplates.ts`. The "Generate with AI" button generates both prompts sequentially — first the system prompt, then the user prompt (with awareness of the system prompt to avoid overlap). Opening an existing complex agent enables "Update Agent" mode (vs "Save Complex Agent" for new ones).
- **Tool types**: Python scripts with AST-parsed input/output schemas for flow compatibility validation (`src/validate/tool_compatibility.py`)

### Environment
- Python 3.10 (conda environment `llm-hub`)
- Node v25.6.0 (see `frontend/.nvmrc`)
- Backend `.env` requires `DATABASE_URL` (defaults to `sqlite:///database/llm_hub.db`)
- Backend `.env` LangFuse config: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` (e.g. `http://localhost:3000` for self-hosted)
- `ENVIRONMENT`: `LOCAL` (default) or `HOSTED` — toggles LLM config backend between YAML file and encrypted DB rows
- `LLM_CONFIG_ENCRYPTION_KEY`: required in HOSTED mode only — Fernet key used to encrypt/decrypt per-user API keys in `llm_provider_configs`
- Frontend `.env` has its own `DATABASE_URL` for Drizzle/better-sqlite3 (Lucia auth)
- LangFuse runs locally via Podman (`podman compose up` in the langfuse repo)
- `ENVIRONMENT` env var (default `local`) gates production behavior — see "Production Execution (Celery + Podman)" above. Only `production` enables Celery dispatch and Podman sandboxing. Unset or `local` for dev.
- Production-only env vars (set in `deploy/podman-compose.yml`, not `.env`): `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`, `CONTAINER_HOST`, `FLOW_RUNNER_IMAGE`, `FLOW_RUNNER_NETWORK`, `FLOW_RUNNER_USE_PODMAN`, optional `FLOW_RUNNER_MEMORY`, `FLOW_RUNNER_CPUS`, `FLOW_RUNNER_TIMEOUT_SECONDS`.
