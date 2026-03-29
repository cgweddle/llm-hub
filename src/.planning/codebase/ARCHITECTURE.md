# Architecture

**Analysis Date:** 2026-03-15

## Pattern Overview

**Overall:** Three-tier client-server architecture with visual workflow builder frontend and unified graph-based execution backend.

**Key Characteristics:**
- Frontend builds visual DAGs that convert to `graph_config` JSON (nodes, edges, entry/exit points)
- Backend executes unified graph structures for both flows (tool DAGs) and agents (multi-agent workflows)
- Streaming support for long-running LLM operations via Server-Sent Events (SSE)
- Factory pattern for agent creation (ReAct vs PydanticAI) and tool execution
- SQLAlchemy ORM with SQLite (dev) / PostgreSQL (prod) for persistence

## Layers

**Presentation Layer (Frontend):**
- Purpose: Visual workflow canvas, code editor, configuration panels, execution monitoring
- Location: `frontend/src/`
- Contains: Svelte components, SvelteKit routes, TypeScript API client
- Depends on: Backend API at `http://localhost:8000`
- Used by: End users building and executing flows/agents

**API Layer:**
- Purpose: Single entry point handling all HTTP requests/responses and streaming
- Location: `src/api/backend.py`
- Contains: FastAPI app with ~40 endpoints, request/response models, middleware
- Depends on: Executors, factories, database, AI integrations
- Used by: Frontend client, direct API calls

**Execution Layer:**
- Purpose: Run flows (tool DAGs) and agents (multi-agent graphs)
- Location: `src/executors/`
- Contains: `FlowExecutor` (DAG traversal), `AgentExecutor` (BFS graph traversal), `ToolExecutor` (Python script execution)
- Depends on: Factories, database, utilities
- Used by: API endpoints

**Factory/Construction Layer:**
- Purpose: Build executable agents and tools from database records
- Location: `src/factories/`
- Contains: `agent_factory.py` (ReAct agents), `pydanticai_agent_factory.py` (PydanticAI agents), `python_script_tool_factory.py` (tools)
- Depends on: Database, AI integrations, utilities
- Used by: Executors, API endpoints

**Data Layer:**
- Purpose: Persistent storage and query interface
- Location: `src/database/`
- Contains: SQLAlchemy ORM models (`database_setup.py`), CRUD helpers (`database.py`)
- Depends on: SQLAlchemy, SQLite/PostgreSQL
- Used by: All backend layers

**Utility/Integration Layer:**
- Purpose: LLM provider config, code generation, system prompt generation, validation
- Location: `src/ai_integrations/`, `src/utils/`, `src/validate/`
- Contains: LLM setup, streaming code generators, tool compatibility validator, retry logic
- Depends on: External LLM APIs, database
- Used by: API endpoints, executors, factories

## Data Flow

**Flow Execution (Tool DAG):**

1. Frontend user arranges tool nodes on canvas, connects edges with optional parameter mappings
2. `buildEnhancedGraphConfig()` in `frontend/src/lib/flowBuilder.ts` converts XYFlow nodes/edges to JSON `graph_config` (nodes dict, edges list, entry_point, exit_points)
3. POST `/flows` sends `FlowCreate` request with `graph_config` to `src/api/backend.py`
4. Backend stores Flow record with `graph_config` JSON in database
5. User requests flow execution via POST `/flows/{id}/execute` with initial input
6. `FlowExecutor` (in `src/executors/flow_executor.py`) loads flow, traverses graph via BFS:
   - For each tool node: Calls `ToolExecutor.create_executable_function()` → creates callable
   - For each agent node: Delegates to `AgentExecutor`
   - Applies parameter mappings from edges to route outputs → inputs
   - Stores execution record, messages in database
7. Frontend polls/streams execution status and displays results

**Agent Execution (Multi-Node Graph):**

1. Frontend builds composed agent in `AgentBuilder.svelte`, defines sub-agents with prompts/tools
2. `buildAgentGraph()` in `frontend/src/lib/agentGraphBuilder.ts` converts to agent `graph_config` (nodes dict with sub-agent configs, edges list with loop markers, entry_point, exit_points)
3. POST `/agents` sends `AgentCreate` with `graph_config`
4. Backend stores Agent record with `graph_config` JSON
5. User requests agent execution via POST `/agents/{id}/execute` with input
6. `AgentExecutor` (in `src/executors/agent_executor.py`) loads agent, traverses graph via BFS:
   - For each node: Checks `agent_type` (pydanticai or react)
   - Calls `PydanticAIAgentFactory.create_pydanticai_agent()` or `AgentFactory.create_react_agent()`
   - Constructs agent with tools (from database), system prompt, LLM config
   - Executes node, captures messages
   - Follows edges (including loops with `max_loop_iterations` limit)
7. If `stream=true`: Wraps generator in `StreamingResponse`, yields SSE chunks to frontend
8. Stores execution record, messages, results in database

**State Management:**

- Frontend: Svelte writable stores for canvas state (`nodes`, `edges`, `viewport`), UI state (`showValidationToast`, `isSaving`), builder mode (flow vs agent)
- Backend: SQLAlchemy ORM manages database state; Execution and Message records track execution history
- User data isolation: All queries filtered by `user_id` (except public items)

## Key Abstractions

**graph_config (Central Data Structure):**
- Purpose: Unified representation of DAGs (tools, agents) as JSON
- Structure: `{nodes: {<id>: config}, edges: [{from_node, to_node, is_loop/mapping}], entry_point, exit_points}`
- Examples: `src/database/database_setup.py` Agent.graph_config, Flow.graph_config
- Pattern: Frontend builds via graph builders (`flowBuilder.ts`, `agentGraphBuilder.ts`), backend traverses with executors

**ExecutorBase Pattern:**
- Purpose: Generic traversal of graph_config DAGs
- Classes: `FlowExecutor`, `AgentExecutor`
- Pattern: Load graph_config, BFS/DFS traversal, execute node, store results/messages

**Factory Pattern for Agents:**
- Purpose: Decouple agent construction from execution
- Classes: `AgentFactory` (ReAct), `PydanticAIAgentFactory` (PydanticAI)
- Usage: Executors call factory methods with node config, get initialized agent
- File paths: `src/factories/agent_factory.py`, `src/factories/pydanticai_agent_factory.py`

**Tool Conversion Pipeline:**
- Purpose: Transform Python script text → callable with type safety
- Path: Script → AST parsing (extract types) → schema storage → execution (eval imports, create closure)
- Files: `src/factories/python_script_tool_factory.py`, `src/executors/tool_executor.py`

## Entry Points

**Frontend Entry:**
- Location: `frontend/src/routes/+page.svelte`
- Triggers: Page load, user interactions (drag/drop, button clicks)
- Responsibilities: Render flow canvas, agent builder, panels, modals; coordinate UI state; send API requests

**Backend Entry:**
- Location: `src/api/backend.py` (FastAPI app)
- Triggers: HTTP requests from frontend
- Responsibilities: Route requests to executors/factories, manage database sessions, serialize/stream responses

**Execution Entry:**
- Location: `src/executors/flow_executor.py::FlowExecutor.execute()` or `src/executors/agent_executor.py::AgentExecutor.execute_agent()`
- Triggers: POST requests to `/flows/{id}/execute` or `/agents/{id}/execute`
- Responsibilities: Traverse graph_config, execute nodes, manage state/messages

## Error Handling

**Strategy:** Try-catch at API layer with JSON error responses; propagate context (file paths, line numbers) for debugging.

**Patterns:**
- API endpoints catch exceptions, return HTTP 400/500 with error message
- Executors log errors, store in Execution.error_message and database
- Frontend displays error toasts, shows validation messages for connection errors
- Tools wrapped in try-catch to isolate failures in node execution
- Retry logic available via `src/utils/retry.py` (exponential backoff, configurable max_retries)

## Cross-Cutting Concerns

**Logging:**
- Framework: Python `logging` module
- Files logged to: `logs/database.log`, `logs/backend.log` (configurable)
- Pattern: Logger per module (`logger = logging.getLogger(__name__)`), DEBUG level for dev

**Validation:**
- Location: `src/validate/tool_compatibility.py`
- Validates: Tool output → input schema matching for flow edges
- Used by: API endpoints (`/validate-tools`, `/validate-connection`) before flow save

**Authentication:**
- Currently: Simple user_id in request headers (placeholder for JWT)
- Pattern: All queries filtered by `user_id` for data isolation
- Future: SvelteKit hooks in `frontend/src/hooks.server.ts` (Lucia auth setup started)

**LLM Provider Configuration:**
- Storage: `~/.llm_hub/config.yaml` (not in repo, user's home directory)
- Providers: Anthropic, OpenAI, Gemini, LM Studio
- Usage: API masks keys before sending to frontend; executors load by `config_name` from flow/agent nodes
- Files: `src/utils/llm_config.py` loads config

**Streaming:**
- Framework: FastAPI `StreamingResponse` with SSE (Server-Sent Events)
- Used for: Agent execution, code generation, system prompt generation
- Pattern: Generator yields `data: {json}\n\n` chunks; frontend reads with `ReadableStream` reader pattern
- Files: Multiple streaming endpoints in `src/api/backend.py` (lines ~325, 399, 433, etc.)

---

*Architecture analysis: 2026-03-15*
