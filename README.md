# LLM Hub

A visual workflow builder for LLM-powered applications. Create tools (Python scripts), agents (LLM-powered executors), and flows (DAGs connecting tools and agents), then execute them from a drag-and-drop canvas UI.

## Getting Started

### Backend

```bash
# Start the FastAPI backend (auto-reloads on changes)
python start_backend.py
# Runs on http://127.0.0.1:8000
```

### Frontend

```bash
cd frontend
nvm use          # Node v25.6.0
npm install
npm run dev      # http://localhost:5173
```

## How Agent Workflows Work

Every agent in LLM Hub is stored as a **graph** (`graph_config`). This is the same structure whether you're building a simple single-agent or a complex multi-agent pipeline.

### Single Agent

A basic agent is a graph with one node. The node defines what LLM to use, what system prompt to follow, and which tools are available:

```
[ Planner Agent ]
    agent_type: pydanticai
    llm_provider: "My Claude Config"
    system_prompt: "You are a helpful planner..."
    tool_ids: [1, 3]
```

When executed, the agent receives user input, reasons with the LLM, optionally calls tools, and returns a response.

### Multi-Agent Workflows

For complex tasks, multiple agents are connected in a graph. Each agent handles a specific responsibility, and edges define the execution order:

```
[ Planner ] ──> [ Executor ] ──> [ Reviewer ]
                     ^                 │
                     └─── (loop) ──────┘
```

The executor traverses this graph using BFS:
1. Start at the **entry point** (the node with no incoming edges)
2. Run the current agent, passing it the accumulated context
3. Follow edges to the next agent(s)
4. For **loop edges** (`is_loop: true`), repeat until the agent signals completion or `max_loop_iterations` is reached
5. Return the output from the **exit point** (the node with no outgoing forward edges)

This pattern supports reflection loops (agent reviews its own work and retries), pipeline chains (research -> draft -> edit), and supervisor architectures.

### Agents Inside Flows

Flows are acyclic DAGs of tool and agent nodes. When a flow contains an agent node, the agent executes as a single unit -- any internal cycles live inside the agent's graph, invisible to the flow. This means you can mix Python script tools and LLM agents in the same pipeline:

```
[ Fetch Data Tool ] ──> [ Analysis Agent ] ──> [ Format Output Tool ]
```

Data bridging between node types is handled automatically:
- **Tool to Agent**: tool output is serialized to text
- **Agent to Tool**: agent text output is mapped to the tool's input parameters
- **Agent to Agent**: text passthrough

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Tool** | A Python script with typed inputs/outputs, parsed via AST |
| **Agent** | An LLM executor defined by a `graph_config` with one or more nodes |
| **Flow** | An acyclic DAG connecting tools and agents |
| **graph_config** | JSON structure with `nodes`, `edges`, `entry_point`, and `exit_points` |
| **LLM Provider** | Named configuration from `~/.llm_hub/config.yaml` (Anthropic, OpenAI, Gemini, LM Studio) |

## Architecture

- **Backend**: FastAPI + SQLAlchemy (SQLite dev / PostgreSQL prod)
- **Frontend**: SvelteKit + Svelte 5 + @xyflow/svelte (drag-and-drop canvas)
- **Agent Frameworks**: PydanticAI and Google ADK (ReAct), selected per-node via `agent_type`

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Production Deployment

Behavior switches on the `ENVIRONMENT` variable (default `local`). Local development is unchanged — flows execute synchronously in the API process. `ENVIRONMENT=production` enables async dispatch and sandboxed execution.

### Local vs Production

|  | `ENVIRONMENT=local` (default) | `ENVIRONMENT=production` |
|---|---|---|
| Flow dispatch | Synchronous, blocks the HTTP request | Celery task, API returns HTTP 202 immediately |
| Flow execution | Runs in the API process | Runs in an ephemeral per-flow Podman container |
| Tool subprocess | On the host | Inside the flow-runner container |
| External services needed | None | Redis (broker), Podman socket |
| LLM credentials | `~/.llm_hub/config.yaml` | Postgres (per-user) |

### Production Components

Added to `deploy/podman-compose.yml` alongside the existing postgres / backend / frontend / caddy services:

- **`redis`** — Celery broker and result backend
- **`worker`** — Long-running Celery worker built from the same image as the backend. Pulls tasks off Redis, spawns a flow-runner container per flow, streams its logs, and updates the Execution row on completion.

Plus a new image built outside compose (per-flow, ephemeral — never runs as a service):

- **`llmhub-flow-runner`** — Minimal Python 3.10 image that hosts `FlowExecutor` for one flow execution. Spawned via the host's Podman socket with memory/CPU limits and `--rm`. Deleted after each flow.

### Building the Flow-Runner Image

The flow-runner image is built separately from the compose services since it's ephemeral (spawned per-flow, not long-running). Build it with:

```bash
podman build -t llmhub-flow-runner -f deploy/flow-runner/Containerfile .
```

The image is a two-stage build (`deploy/flow-runner/Containerfile`):
1. **Build stage**: Installs Python deps from `requirements.txt` plus common data-science packages (`numpy`, `pandas`, `pyyaml`) that user-authored tools may need
2. **Runtime stage**: Copies installed packages and `src/` into a minimal `python:3.10-slim` image, runs as the unprivileged `llmhub` user

The image contains the full `src/` tree because the flow-runner runs the same `FlowExecutor` code as local mode — it needs access to executors, database modules, and utilities. Since the container is ephemeral and auto-deleted, the copied code disappears with it.

Rebuild this image whenever `src/` or `requirements.txt` changes. The CI/CD pipeline (`deploy.yml`) builds it automatically.

### Execution Flow (production)

```
Frontend → POST /flows/{id}/execute
              ↓
Backend    pre-creates Execution row (status=pending)
           dispatches execute_flow_task to Redis
           returns HTTP 202 with execution_id
              ↓
Worker     picks up task from Redis
           spawns llmhub-flow-runner container on llmhub-net
           streams container logs, waits for exit
              ↓
Flow-runner  runs FlowExecutor.execute_flow()
             writes execution tree to postgres
             exits; container is removed
              ↓
Frontend polls GET /executions/{id} → sees pending → running → completed
```

### Key Design Decisions

- **One Celery task per flow** (not per node): DAG traversal stays in one process, avoiding distributed-transaction complexity.
- **Podman wraps the whole flow**, not individual tools: `FlowExecutor` and `subprocess.run()` for tools work identically inside or outside the container — no code branching on environment.
- **Unrestricted network on the flow-runner**: tools and agents legitimately need outbound access for external APIs. Sandbox is resource + filesystem isolation only.
- **Credentials pre-resolved by the worker**: the worker decrypts LLM credentials before spawning the flow-runner and passes only the needed providers via env var. The flow-runner never has the encryption key or access to other users' credentials.

## Security

### Flow-Runner Container Isolation

The flow-runner executes user-authored tool scripts, so it operates with minimal privileges:

| Resource | Flow-Runner Access |
|---|---|
| Database user | `flowrunner` — restricted Postgres role |
| DB read access | `flows`, `tools`, `agents`, `executions`, association tables |
| DB write access | `executions` only (INSERT/UPDATE) |
| Sensitive tables | No access to `llm_provider_configs`, `users`, `sessions`, `evaluations`, `prompts` |
| LLM API keys | Only the providers needed for the current flow, pre-resolved by the worker |
| `LLM_CONFIG_ENCRYPTION_KEY` | Not available in the container |
| Filesystem | Writable but ephemeral — container is auto-deleted after execution |
| Network | Unrestricted (tools/agents need outbound API access) |
| Resources | Memory and CPU capped via Podman limits |

### How Credential Isolation Works

1. **Worker** receives the Celery task with `user_id` and `agent_llms` (a map of node names to provider names)
2. **Worker** calls `load_llm_provider_config(user_id)` to decrypt the user's LLM credentials from Postgres
3. **Worker** filters to only the providers referenced in `agent_llms` and serializes them as `FLOW_RUNNER_LLM_CONFIG`
4. **Flow-runner** reads the pre-resolved config from the env var — no database query, no decryption needed

### Restricted Database Role

The `flowrunner` Postgres role is created automatically during `database_setup.py setup` (runs on every deploy). It requires `FLOWRUNNER_DB_PASSWORD` in the environment. The role has:
- `SELECT` on reference tables (flows, tools, agents, executions, association tables)
- `INSERT`, `UPDATE` on `executions` (for recording execution results)
- `USAGE` on `executions_id_seq` (for auto-increment IDs)
- No access to `llm_provider_configs`, `users`, `sessions`, `evaluations`, `evaluation_results`, or `prompts`

See [deploy/DEPLOY.md](deploy/DEPLOY.md) for the hosting / CI-CD setup.
