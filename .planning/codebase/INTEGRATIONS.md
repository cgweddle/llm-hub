# External Integrations

**Analysis Date:** 2025-03-15

## APIs & External Services

**LLM Providers:**
- **Anthropic Claude** - PydanticAI integration via `AnthropicModel` from `pydantic_ai.models.anthropic`
  - SDK/Client: `pydantic_ai`
  - Config: Loaded from `~/.llm_hub/config.yaml` by name reference
  - Used in: `src/factories/pydanticai_agent_factory.py` (line 12), agent execution

- **OpenAI (ChatGPT)** - PydanticAI integration via `OpenAIModel` from `pydantic_ai.models.openai`
  - SDK/Client: `pydantic_ai`
  - Config: Loaded from `~/.llm_hub/config.yaml` by name reference
  - Used in: `src/factories/pydanticai_agent_factory.py` (line 13), agent execution

- **Google Gemini** - Google ADK integration via `google.adk.agents.LlmAgent`
  - SDK/Client: `google-adk`
  - Config: Loaded from `~/.llm_hub/config.yaml` as ReAct model
  - Used in: `src/factories/agent_factory.py`, ReAct agent execution

- **LM Studio (Local)** - OpenAI-compatible API
  - SDK/Client: `pydantic_ai` with `OpenAIModel` and custom base_url
  - Config: Stored in `~/.llm_hub/config.yaml` with custom base_url
  - Used in: Agent execution, code generation, prompt generation

**Search & Web Integration:**
- **Brave Search** - Web search integration
  - SDK/Client: `langchain_community` (BraveSearchLoader)
  - Auth: `BRAVE_SEARCH_API_KEY` environment variable
  - Used in: `src/tools/web_search_tool.py`
  - Purpose: Web search tool for agents

**Code Generation:**
- **LLM-based Code Generation** - Uses configured LLM provider
  - Endpoint: `POST /tools/generate-code` in `src/api/backend.py`
  - Streaming: Yes, via `StreamingResponse`
  - Used for: Auto-generating Python tool code with selected provider

**Prompt Generation:**
- **LLM-based Prompt Generation** - Uses configured LLM provider
  - System Prompt: `POST /agents/generate-system-prompt` in `src/api/backend.py`
  - User Prompt: `POST /agents/generate-user-prompt` in `src/api/backend.py`
  - Streaming: Yes, via `StreamingResponse`
  - Module: `src/ai_integrations/generate_system_prompt.py`
  - Purpose: Auto-generate agent system prompts based on agent name, description, and available tools

## Data Storage

**Databases:**
- **SQLite** (Development)
  - Default location: `database/llm_hub.db`
  - Client: SQLAlchemy ORM
  - Connection: `sqlite:///database/llm_hub.db` (can be overridden via `DATABASE_URL` env var)
  - Used for: User accounts, agents, tools, flows, executions, messages

- **PostgreSQL** (Production)
  - Connection: Via `DATABASE_URL=postgresql://user:password@host/dbname`
  - Client: SQLAlchemy ORM
  - Supported: Full feature set via SQLAlchemy
  - Models: All defined in `src/database/database_setup.py`

**Frontend Auth Database:**
- **SQLite** (Local, frontend only)
  - Client: `better-sqlite3` 12.5.0
  - Location: Hardcoded to `database/llm_hub.db` in `frontend/src/lib/server/auth.ts` (line 7)
  - Schema: Lucia manages `users` and `sessions` tables
  - Used for: Session management, user authentication state

**File Storage:**
- Local filesystem only
  - Python scripts stored as text in `Tool.script_code` column
  - Execution artifacts stored in database as JSON (`executions.execution_metadata`)
  - No cloud storage integration

**Caching:**
- None detected
- LLM responses are not cached; streaming directly to client

## Authentication & Identity

**Auth Provider:**
- **Custom + Lucia** - Hybrid approach
  - Frontend auth: Lucia 3.2.2 with `@lucia-auth/adapter-sqlite`
    - Password hashing: Argon2 via `@node-rs/argon2`
    - Sessions: SQLite table managed by Lucia
    - Location: `frontend/src/lib/server/auth.ts`

  - Backend auth: bcrypt password hashing
    - Algorithm: `passlib.hash.bcrypt`
    - User storage: SQLAlchemy `User` model with `password_hash` column
    - Location: `src/database/database_setup.py`

**Session Management:**
- Frontend: Lucia session cookies (HTTP-only, configurable secure flag for HTTPS)
- Backend: Stateless token-based (user_id passed in request body for agent/flow execution)

**User Model:**
- `User` table in `src/database/database_setup.py`:
  - `username` (unique, required)
  - `email` (unique, required)
  - `password_hash` (required)
  - `is_active` (boolean, default True)
  - Timestamps: `created_at`, `updated_at`

## Monitoring & Observability

**Error Tracking:**
- None detected
- Errors logged via Python's `logging` module

**Logs:**
- **Backend**: Python `logging` module
  - Log level configurable: Default `info` in `start_backend.py`
  - Loggers created per module with `logging.getLogger(__name__)`

- **Frontend**: Browser console only
  - No external logging service

**Execution Tracking:**
- Database-driven: All executions stored in `Execution` table
  - `status` field: `running` / `completed` / `failed`
  - `error_message` field for failure details
  - `execution_metadata` JSON field for additional context
  - Message history stored in `Message` table with timestamps and sender info

## CI/CD & Deployment

**Hosting:**
- Not specified; assumes manual deployment or container-based

**CI Pipeline:**
- None detected
- No GitHub Actions, GitLab CI, or other automated pipelines

**Development Server:**
- Backend: `uvicorn` with `reload=True` for hot reloading
  - Command: `python start_backend.py`
  - Host: `127.0.0.1:8000`

- Frontend: Vite dev server
  - Command: `cd frontend && npm run dev`
  - Host: `localhost:5173` (configured in `frontend/vite.config.ts`)

**Build:**
- Frontend: `npm run build` → Vite build output to `frontend/build/`
- Backend: No build step; runs directly with Python

## Environment Configuration

**Required Environment Variables:**

Backend:
- `DATABASE_URL` (optional) - Database connection string
  - Default: `sqlite:///database/llm_hub.db`
  - PostgreSQL: `postgresql://user:password@localhost/dbname`

Frontend:
- `DATABASE_URL` (optional) - Local SQLite path for auth
  - Default: `local.db`

Optional:
- `BRAVE_SEARCH_API_KEY` - Only if using web search tools

**LLM Configuration:**
- External YAML file: `~/.llm_hub/config.yaml`
- Not in `.env` to keep repo secret-free
- Format:
  ```yaml
  models:
    - name: "Production Claude"
      provider: "anthropic"
      model: "claude-3-5-sonnet-20241022"
      api_key: "sk-ant-..."
    - name: "OpenAI GPT-4"
      provider: "openai"
      model: "gpt-4"
      api_key: "sk-..."
  ```

**Secrets Management:**
- Backend:
  - `~/.llm_hub/config.yaml` contains LLM API keys (user-controlled)
  - `.env` file may contain `DATABASE_URL` (user-controlled, not in repo)
  - API keys are masked before sending to frontend via `mask_credentials()` function in `src/utils/llm_config.py`

- Frontend:
  - `.env` file for database URL (user-controlled)
  - No API keys stored in frontend `.env`

**CORS Configuration:**
- Backend: `src/api/backend.py` allows requests from:
  - `http://localhost:3000`
  - `http://localhost:5173`
  - `http://localhost:5174`
  - `http://127.0.0.1:3000`
  - `http://127.0.0.1:5173`
  - `http://127.0.0.1:5174`
  - Credentials enabled

## Webhooks & Callbacks

**Incoming:**
- None detected

**Outgoing:**
- None detected

## Streaming & Real-time Communication

**Streaming Protocol:**
- **Server-Sent Events (SSE)** via FastAPI `StreamingResponse`
- Used for:
  - Agent execution responses: `POST /agents/{agent_id}/execute` (line 280, `src/api/backend.py`)
  - Flow execution: `POST /flows/{flow_id}/execute` (line 791)
  - Code generation: `POST /tools/generate-code` (line 531)
  - System prompt generation: `POST /agents/generate-system-prompt` (line 383)
  - User prompt generation: `POST /agents/generate-user-prompt` (line 417)

**Streaming Implementation:**
- Backend: Generator functions yield JSON strings (one object per line)
- Frontend: Reads streams via `ReadableStream` API
  - See `frontend/src/lib/api.ts` for stream reading patterns
  - Example: AgentExecutor.execute_agent() yields streaming agent responses as async generator

**No WebSocket Support:**
- Currently SSE only; no bidirectional WebSocket integration

## Tool Execution Environment

**Python Tools:**
- Default: Run in main Python process
- Optional: Isolated conda environments per tool/flow
  - Conda env path stored in `Flow.conda_env` field
  - Used in: `src/executors/tool_executor.py`
  - Activation: `subprocess` with conda environment activation

**Tool Factory:**
- **Python Script Tool Factory**: `src/factories/python_script_tool_factory.py`
  - Parses Python scripts via AST to extract:
    - Main function signature
    - Input parameter types
    - Return type
    - Helper functions
  - Generates Tool database records with schemas

**Code Parsing & Analysis:**
- AST-based parsing for Python tools
- Type extraction from function signatures
- Validation: `src/validate/tool_compatibility.py` validates tool connections in flows

---

*Integration audit: 2025-03-15*
