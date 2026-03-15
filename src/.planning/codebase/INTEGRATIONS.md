# External Integrations

**Analysis Date:** 2025-03-15

## APIs & External Services

**LLM Providers:**
- Anthropic Claude API
  - SDK: `pydantic_ai.models.anthropic` (AnthropicModel)
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Used by: PydanticAI agents (`src/factories/pydanticai_agent_factory.py`)
  - Config: Loaded from `~/.llm_hub/config.yaml` by name

- OpenAI GPT API
  - SDK: `pydantic_ai.models.openai` (OpenAIModel)
  - Auth: `OPENAI_API_KEY` environment variable
  - Used by: PydanticAI agents and tool executors
  - Config: Loaded from `~/.llm_hub/config.yaml` by name
  - Compatibility: Also used for LM Studio (local model serving)

- Google Generative AI
  - SDK: `google.adk.agents` (Google Agent Development Kit)
  - Auth: `GOOGLE_API_KEY` environment variable
  - Used by: ReAct agents (`src/factories/agent_factory.py`)
  - Config: Loaded from `~/.llm_hub/config.yaml` by name

- LM Studio (Local)
  - Uses OpenAI-compatible client (`pydantic_ai.models.openai`)
  - Auth: `OPENAI_API_KEY` = dummy value "lm-studio"
  - Base URL: Configured in `~/.llm_hub/config.yaml`

**LLM Orchestration Frameworks:**
- LangChain & LangGraph
  - Packages: `langchain`, `langgraph`, `langchain-aws`
  - Used by: Legacy tool/agent creation and composition utilities
  - Tool definition: `langchain.tools.Tool`, `langchain.tools.BaseTool`
  - Graph state: `langgraph.graph.MessagesState`, `langgraph.prebuilt.create_react_agent`

**HTTP Client:**
- httpx
  - Purpose: Async HTTP requests for inter-service calls
  - Used by: Agents and tools that need to make external requests

## Data Storage

**Databases:**

**Development:**
- SQLite 3
  - Connection: `sqlite:///database/llm_hub.db` (default, overridable via `DATABASE_URL`)
  - ORM: SQLAlchemy
  - Models: `src/database/database_setup.py` (User, Agent, Tool, Flow, Execution, Message, Prompt)

**Production:**
- PostgreSQL
  - Connection: `postgresql://user:password@host/dbname` (via `DATABASE_URL`)
  - ORM: SQLAlchemy (same models used for both SQLite and PostgreSQL)
  - Migration: No migrations framework detected; schema creation handled by SQLAlchemy `create_all()`

**Frontend Auth Database:**
- SQLite (better-sqlite3)
  - Location: `frontend/src/lib/server/db/` (Lucia auth database)
  - ORM: Drizzle ORM
  - Schema: `frontend/src/lib/server/db/schema.ts` (user table with id, age)

**File Storage:**
- Local filesystem only
  - Tool scripts: Stored in `Tool.script_code` (TEXT column)
  - Python code: Stored in `Tool.function_code` and `Tool.helper_functions` (JSON)
  - No cloud storage integration detected

**Caching:**
- None detected - Each request loads fresh data from database

## Authentication & Identity

**Auth Provider:**
- Custom password-based authentication
  - User table: `User` model in `src/database/database_setup.py`
  - Password hashing: bcrypt via `passlib.hash.bcrypt`
  - Hash field: `User.password_hash`

**Frontend Session Auth:**
- Lucia 3.2.2 - Session-based auth framework
  - Adapter: `@lucia-auth/adapter-sqlite` (frontend auth database)
  - Password hashing: Argon2 via `@node-rs/argon2`
  - Session storage: SQLite table (frontend auth DB)

**API Authentication:**
- Custom user_id parameter in requests (stateless, no JWT/tokens detected)
- Backend API expects: `user_id` in request bodies or query parameters

## Monitoring & Observability

**Error Tracking:**
- None detected - No Sentry, Rollbar, or similar integration

**Logging:**
- Python logging module
  - Configured in executors and factories via `logging.getLogger(__name__)`
  - Log levels: info, warning, error, debug
  - No centralized logging service (local stdout/stderr)

**Streaming Telemetry:**
- Server-Sent Events (SSE) for agent execution progress
  - Backend: `StreamingResponse` from `fastapi.responses`
  - Frontend: `ReadableStream` pattern with fetch API
  - Endpoints: `/agents/{agent_id}/execute/stream`, `/flows/{flow_id}/execute/stream`

## CI/CD & Deployment

**Hosting:**
- Not specified in codebase - deployment target unknown
- Supports: FastAPI runs on Uvicorn (any ASGI-compatible hosting)

**CI Pipeline:**
- pytest configured for local testing (`tests/conftest.py` exists)
- No GitHub Actions, GitLab CI, or similar detected

**Conda Environments:**
- Tool isolation support: `conda_env` field in Flow model
  - Used by: `src/executors/tool_executor.py::create_conda_executable_function()`
  - API endpoint: `/conda/environments` - lists available conda environments
  - Purpose: Run Python tools in isolated conda environments

## Environment Configuration

**Required env vars - Backend:**
- `DATABASE_URL` - Optional, defaults to `sqlite:///database/llm_hub.db`

**Required env vars - Runtime:**
- Set dynamically by executors/factories from `~/.llm_hub/config.yaml`:
  - `ANTHROPIC_API_KEY`
  - `OPENAI_API_KEY`
  - `GOOGLE_API_KEY`
  - `LLMHUB_CONFIG_NAME` - Name of current LLM config being used
  - `LLMHUB_MODEL_NAME` - Model name from config

**Secrets location:**
- User home directory: `~/.llm_hub/config.yaml` (YAML file, not in repo)
  - Structure:
    ```yaml
    models:
      - name: "config_name"
        provider: "anthropic|openai|google|lmstudio"
        model: "gpt-4|claude-3|gemini-pro|etc"
        api_key: "sk-...|XXXXXXXXX|etc"
        base_url: "http://localhost:1234/v1"  # Optional, for LM Studio
        config_name: "config_name"
    ```
- API keys are masked before sending to frontend: `"***MASKED***"` pattern
- Restoration utilities: `mask_credentials()`, `restore_masked_credentials()` in `src/utils/llm_config.py`

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected
- Agents and tools make internal calls only (no external webhooks)

## Tool Execution Environment

**Python Script Execution:**
- Direct subprocess execution for inline Python scripts
- Optional conda environment isolation via `Flow.conda_env` path
- Tool factory: `src/factories/python_script_tool_factory.py`
  - AST-parses Python scripts to extract type schemas
  - Validates tool compatibility in flows via `src/validate/tool_compatibility.py`

**Agent Execution:**
- Two agent types supported in unified `graph_config`:
  1. PydanticAI: Agent framework with streaming support
  2. Google ADK ReAct: Multi-step reasoning agents
- Both routed through `AgentExecutor` in `src/executors/agent_executor.py`

---

*Integration audit: 2025-03-15*
