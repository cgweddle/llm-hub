# Codebase Structure

**Analysis Date:** 2026-03-15

## Directory Layout

```
llm-hub/
├── src/                           # Backend Python codebase
│   ├── api/
│   │   └── backend.py             # FastAPI app, all ~40 endpoints
│   ├── database/
│   │   ├── database_setup.py      # SQLAlchemy models (User, Agent, Tool, Flow, Execution, Message, Prompts)
│   │   ├── database.py            # CRUD helpers
│   │   └── setup_sqlite.py        # SQLite setup utilities
│   ├── executors/
│   │   ├── flow_executor.py       # DAG traversal for tool flows
│   │   ├── agent_executor.py      # BFS graph traversal for agents
│   │   └── tool_executor.py       # Python script execution
│   ├── factories/
│   │   ├── agent_factory.py       # ReAct agent construction (Google ADK)
│   │   ├── pydanticai_agent_factory.py  # PydanticAI agent construction
│   │   └── python_script_tool_factory.py  # Tool AST parsing + schema extraction
│   ├── ai_integrations/
│   │   ├── generate_system_prompt.py  # LLM-powered system prompt generation (streaming)
│   │   └── generate_python_tools.py   # LLM code generation
│   ├── validate/
│   │   └── tool_compatibility.py  # Schema matching, edge validation
│   ├── utils/
│   │   ├── __init__.py            # LLM config loading
│   │   ├── llm_config.py          # Parse ~/.llm_hub/config.yaml
│   │   └── retry.py               # Exponential backoff, retry decorator
│   ├── converters/
│   │   └── pydanticai_tool_converter.py  # Tool → PydanticAI format
│   ├── tools/                     # Example/sample tools (RAG, web search, etc.)
│   ├── prompts/                   # Template prompts
│   ├── agents.py                  # Legacy agent utilities
│   ├── graph.py                   # Graph utilities
│   └── .planning/codebase/        # This documentation (ARCHITECTURE.md, STRUCTURE.md, etc.)
│
├── frontend/                      # SvelteKit + Svelte 5 frontend
│   ├── src/
│   │   ├── routes/
│   │   │   ├── +page.svelte       # Main flow builder canvas (large component)
│   │   │   ├── AgentBuilder.svelte  # Agent composition UI
│   │   │   ├── AgentBuilderNode.svelte  # Sub-agent node in composer
│   │   │   ├── AgentConfigModal.svelte  # Agent configuration dialog
│   │   │   ├── ToolNode.svelte    # Tool/agent node on canvas
│   │   │   ├── FullscreenNodeModal.svelte  # Code editor modal
│   │   │   ├── AvailableItemsPanel.svelte  # Tools/agents sidebar
│   │   │   ├── CondaEnvironmentsPanel.svelte  # Conda env selector
│   │   │   ├── LLMProvidersPanel.svelte  # LLM provider config UI
│   │   │   ├── ColorSelectorNode.svelte  # Color picker node
│   │   │   ├── ExpandableNode.svelte  # Expandable node variant
│   │   │   ├── FloatingEdge.svelte  # Edge rendering
│   │   │   ├── login/              # Login page
│   │   │   ├── register/           # Registration page
│   │   │   ├── logout/             # Logout endpoint
│   │   │   └── +page.server.ts    # SvelteKit server-side page logic
│   │   ├── lib/
│   │   │   ├── api.ts             # Backend API client, interfaces (Agent, Tool, Flow, etc.)
│   │   │   ├── flowBuilder.ts     # XYFlow nodes/edges → graph_config JSON
│   │   │   ├── agentGraphBuilder.ts  # XYFlow nodes/edges → agent graph_config JSON
│   │   │   ├── agentTemplates.ts  # Agent type templates (pydanticai, react)
│   │   │   ├── elkLayout.ts       # Auto-layout via elkjs
│   │   │   ├── store.ts           # Legacy store
│   │   │   ├── utils.ts           # Helper functions
│   │   │   ├── stores/
│   │   │   │   └── builderMode.ts # Writable store: 'flow' | 'agent'
│   │   │   │   └── fullscreenNode.ts  # Current node in fullscreen editor
│   │   │   ├── components/ui/     # shadcn-svelte UI components (button, input, label, dialog, select)
│   │   │   ├── server/db/         # Lucia auth database (better-sqlite3)
│   │   │   └── hooks/             # Custom hooks
│   │   ├── app.html               # HTML template
│   │   ├── app.css                # Global CSS (Tailwind v4)
│   │   └── hooks.server.ts        # SvelteKit auth hooks (Lucia)
│   ├── svelte.config.js           # SvelteKit config
│   ├── vite.config.ts             # Vite bundler config
│   ├── tsconfig.json              # TypeScript config
│   ├── .nvmrc                     # Node v25.6.0
│   ├── package.json               # npm dependencies (@xyflow/svelte, codemirror, tailwind, etc.)
│   └── static/                    # Static assets
│
├── database/                      # Legacy database module (also defined in src/database/)
│   ├── database.py
│   ├── database_setup.py
│   └── README.md
│
├── start_backend.py               # Entry point: python start_backend.py
├── CLAUDE.md                      # Development guide (this file)
├── AGENT_BUILDER_PLAN.md          # Agent builder feature plan
├── README.md                       # Project overview
└── .env                           # Environment vars (DATABASE_URL, PYTHONPATH)
```

## Directory Purposes

**Backend Source (`src/`):**
- Purpose: Python FastAPI backend serving all API endpoints and execution logic
- Contains: Models, executors, factories, utilities, integrations
- Key files: `backend.py` (all endpoints), `database_setup.py` (ORM models)

**API (`src/api/`):**
- Purpose: Single FastAPI application file with all HTTP endpoints
- Contains: ~40 route handlers, request/response Pydantic models, dependency injection
- Key files: `backend.py` (1000+ lines, all endpoints)

**Database (`src/database/`):**
- Purpose: SQLAlchemy ORM layer and CRUD helpers
- Contains: Table models (User, Agent, Tool, Flow, Execution, Message, Prompts), association tables, session management
- Key files: `database_setup.py` (models), `database.py` (CRUD), `setup_sqlite.py` (migrations)

**Executors (`src/executors/`):**
- Purpose: Run flows (tool DAGs) and agents (multi-node graphs)
- Contains: Graph traversal logic, node execution, message recording
- Key files: `flow_executor.py` (tool DAG), `agent_executor.py` (multi-agent), `tool_executor.py` (Python execution)

**Factories (`src/factories/`):**
- Purpose: Construct agents and tools from database records
- Contains: Agent builders (ReAct, PydanticAI), tool parser (AST → schema)
- Key files: `agent_factory.py` (ReAct), `pydanticai_agent_factory.py` (PydanticAI), `python_script_tool_factory.py` (tools)

**AI Integrations (`src/ai_integrations/`):**
- Purpose: LLM-powered code/prompt generation
- Contains: Streaming code generators, system prompt generation
- Key files: `generate_system_prompt.py`, `generate_python_tools.py`

**Utilities (`src/utils/`):**
- Purpose: Shared helpers
- Contains: LLM config loading, retry logic
- Key files: `__init__.py` (config), `llm_config.py` (YAML parser), `retry.py` (backoff)

**Validation (`src/validate/`):**
- Purpose: Flow edge compatibility checking
- Contains: Schema matching logic
- Key files: `tool_compatibility.py`

**Frontend (`frontend/src/`):**
- Purpose: SvelteKit web application
- Contains: Pages (flows, agents), components, API client, stores, builders
- Key files: `+page.svelte` (main canvas), `AgentBuilder.svelte` (agent composer), `api.ts` (client)

**Routes (`frontend/src/routes/`):**
- Purpose: SvelteKit page structure
- Contains: Components and page logic
- Key files: `+page.svelte` (primary, handles flow/agent building), `AgentBuilder.svelte` (agent composition), modals

**Lib (`frontend/src/lib/`):**
- Purpose: Shared utilities and builders
- Contains: API client, graph builders, stores, utilities
- Key files: `api.ts` (typed client), `flowBuilder.ts` (graph config generation), `agentGraphBuilder.ts` (agent graphs)

## Key File Locations

**Entry Points:**
- Backend: `start_backend.py` (calls `python -m src.api.backend`)
- Frontend: `frontend/src/routes/+page.svelte` (main page)
- Frontend login: `frontend/src/routes/login/+page.svelte`

**API Endpoints:**
- All 40 endpoints defined in: `src/api/backend.py`

**Data Models:**
- All 7 tables: `src/database/database_setup.py`

**Core Execution:**
- Tool flow execution: `src/executors/flow_executor.py`
- Agent execution: `src/executors/agent_executor.py`
- Tool execution: `src/executors/tool_executor.py`

**Agent Construction:**
- ReAct agents: `src/factories/agent_factory.py`
- PydanticAI agents: `src/factories/pydanticai_agent_factory.py`

**Frontend API Client:**
- All backend communication: `frontend/src/lib/api.ts`

**Graph Builders:**
- Tool flows: `frontend/src/lib/flowBuilder.ts`
- Composed agents: `frontend/src/lib/agentGraphBuilder.ts`

## Naming Conventions

**Files:**
- Python modules: `lowercase_with_underscores.py` (e.g., `agent_executor.py`, `database_setup.py`)
- Svelte components: `PascalCase.svelte` (e.g., `ToolNode.svelte`, `AgentBuilder.svelte`)
- TypeScript/JavaScript: `camelCase.ts` or `PascalCase.ts` for classes
- Routes: SvelteKit convention `+page.svelte`, `+page.server.ts`, `+layout.svelte`

**Directories:**
- Python packages: `lowercase` (e.g., `executors/`, `factories/`, `utils/`)
- Frontend features: `lowercase` (e.g., `routes/`, `lib/`, `stores/`)

**Classes:**
- Python: `PascalCase` (e.g., `FlowExecutor`, `AgentFactory`, `PydanticAIAgentFactory`)
- TypeScript: `PascalCase` (e.g., `Agent`, `Tool`, `GraphConfig`)

**Functions:**
- Python: `snake_case` (e.g., `create_executable_function`, `get_agent_by_id`)
- TypeScript: `camelCase` (e.g., `buildEnhancedGraphConfig`, `buildAgentGraph`)

**Constants:**
- Python: `UPPER_CASE` (e.g., `DEFAULT_LLM_RETRY_CONFIG`)
- TypeScript: `UPPER_CASE` or `camelCase` (context-dependent)

## Where to Add New Code

**New Backend Endpoint:**
- Add route handler to: `src/api/backend.py`
- Create request model in same file (Pydantic BaseModel)
- Implement logic using existing executor/factory, store in database via CRUD helpers

**New Executor or Transformation:**
- File location: `src/executors/<name>_executor.py`
- Pattern: Class with `__init__(self, session, ...)` and `execute(...)` method
- Inherits patterns from `FlowExecutor`, `AgentExecutor`

**New Factory (Agent or Tool):**
- File location: `src/factories/<name>_factory.py`
- Pattern: Class with static/class methods `create_*(...) -> Agent/Tool`
- See `agent_factory.py`, `pydanticai_agent_factory.py` for patterns

**New Frontend Component:**
- File location: `frontend/src/routes/<ComponentName>.svelte` (if top-level page) or `frontend/src/components/<ComponentName>.svelte` (if reusable)
- Pattern: Svelte component with `<script lang="ts">` block, reactive stores, imports from `lib/api.ts`
- Use shadcn-svelte UI components from `lib/components/ui/`

**New Frontend Utility/Builder:**
- File location: `frontend/src/lib/<name>.ts` (e.g., `flowBuilder.ts`)
- Pattern: Export typed functions, use types from `lib/api.ts`

**New Validation Logic:**
- File location: `src/validate/<name>.py`
- Pattern: Functions taking database records or schemas, return validation result
- Used by: API validation endpoints

**New Utility Helper:**
- File location: `src/utils/<name>.py`
- Pattern: Pure functions or helpers with no side effects
- Examples: `llm_config.py` (YAML parsing), `retry.py` (backoff decorator)

## Special Directories

**.planning/codebase/**
- Purpose: Generated documentation (ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, CONCERNS.md)
- Generated: Yes (via `/gsd:map-codebase` command)
- Committed: Yes (to track codebase analysis over time)

**frontend/.svelte-kit/**
- Purpose: SvelteKit generated files (type hints, server/client code splitting)
- Generated: Yes (auto-generated during dev/build)
- Committed: No (in .gitignore)

**frontend/dist/**
- Purpose: Production build output
- Generated: Yes (`npm run build`)
- Committed: No

**node_modules/, __pycache__/**
- Purpose: Package dependencies and compiled Python
- Generated: Yes
- Committed: No

**database/ (root level)**
- Purpose: Legacy database module (duplicate of `src/database/`, kept for compatibility)
- Committed: Yes
- Note: Use `src/database/` for new code

**logs/**
- Purpose: Runtime logs (database.log, backend.log)
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-03-15*
