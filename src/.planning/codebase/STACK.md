# Technology Stack

**Analysis Date:** 2025-03-15

## Languages

**Primary:**
- Python 3.10 - Backend server, executors, factories, database models
- TypeScript 5.0 - Frontend application, SvelteKit components, API client type definitions
- Svelte 5.0 - Frontend UI components and reactive pages

**Secondary:**
- SQL - SQLAlchemy models compile to SQLite (dev) and PostgreSQL (prod)
- YAML - LLM provider configuration at `~/.llm_hub/config.yaml`

## Runtime

**Environment:**
- Python 3.10 (conda environment `llm-hub`)
- Node.js v25.6.0 (specified in `frontend/.nvmrc`)

**Package Manager:**
- pip (Python) - specified in `requirements.txt` and `environment.yml`
- npm (Node.js) - `frontend/package.json` with lockfile `frontend/package-lock.json`

## Frameworks

**Core Backend:**
- FastAPI 0.x - REST API server at `src/api/backend.py`, SSE streaming endpoints
- Uvicorn - ASGI server runner (auto-reload enabled in `start_backend.py`)
- SQLAlchemy - ORM for database models in `src/database/database_setup.py`

**Core Frontend:**
- SvelteKit 2.16.0 - Full-stack framework with server routes and client components
- Svelte 5.0 - Reactive UI components
- Tailwind CSS 4.1.17 - Utility-first CSS framework
- TypeScript 5.0 - Static type checking

**Visualization & UI:**
- @xyflow/svelte 1.2.4 - Node graph canvas library (drag-and-drop flow builder)
- elkjs 0.11.0 - Automatic graph layout engine
- shadcn-svelte 1.0.12 - Accessible UI component library (buttons, dialogs, selects)
- CodeMirror 6.x - In-browser Python code editor
  - @codemirror/lang-python 6.2.1 - Python syntax highlighting
  - @codemirror/theme-one-dark 6.1.3 - Dark theme
- Lucide/Svelte 0.544.0 - Icon library

**Testing:**
- pytest - Python test runner in `tests/` directory

**Build/Dev:**
- Vite 6.0.0 - Frontend build tool and dev server
- @sveltejs/adapter-auto 4.0.0 - SvelteKit adapter for deployment
- Drizzle ORM - Frontend schema management (database/schema.ts) for Lucia auth
- better-sqlite3 12.5.0 - Embedded SQLite for frontend auth database

## Key Dependencies

**Critical Backend:**
- pydantic 2.x - Data validation and BaseModel for API schemas
- pydantic-ai 0.0.13+ - Agent framework with streaming support
- passlib - Password hashing (bcrypt for user auth)
- httpx - Async HTTP client for tool/agent communications

**LLM Agent Frameworks:**
- google-adk - Google Agent Development Kit for ReAct agents (used in `src/factories/agent_factory.py`)
- langchain - Tool definitions and agent utilities
- langgraph - Agent graph orchestration
- langchain-aws - AWS integration for LangChain tools

**PydanticAI Integrations:**
- pydantic_ai.models.anthropic - AnthropicModel for Claude API
- pydantic_ai.models.openai - OpenAIModel for OpenAI and LM Studio (compatible)

**Frontend Auth & Database:**
- lucia 3.2.2 - Session-based authentication framework
- @lucia-auth/adapter-sqlite 3.0.2 - Lucia SQLite adapter for sessions
- @node-rs/argon2 2.0.2 - Argon2 password hashing
- Drizzle Kit - Schema generation and migrations for auth database

**Frontend HTTP & Async:**
- Built-in Fetch API - All backend communication in `frontend/src/lib/api.ts`
- ReadableStream - SSE streaming from backend endpoints

**Utilities:**
- bits-ui 2.14.4 - Unstyled component primitives
- tailwind-merge 3.4.0 - Intelligent Tailwind class merging
- clsx 2.1.1 - Conditional class name utilities
- mode-watcher 1.1.0 - Light/dark mode detection
- shiki 3.17.0 - Syntax highlighting

## Configuration

**Environment:**
- `.env` file (not tracked, exists in repo root)
- Environment variables:
  - `DATABASE_URL` - Database connection string (defaults to `sqlite:///database/llm_hub.db`)
  - `ANTHROPIC_API_KEY` - Set by factories/executors at runtime from `~/.llm_hub/config.yaml`
  - `OPENAI_API_KEY` - Set by factories/executors at runtime from `~/.llm_hub/config.yaml`
  - `GOOGLE_API_KEY` - Set by Google ADK agent factory at runtime

**LLM Provider Config:**
- `~/.llm_hub/config.yaml` - User's home directory (not in repo)
  - Contains named LLM provider configurations (Anthropic, OpenAI, Gemini, LM Studio)
  - Loaded by `src/utils/llm_config.py` functions: `load_llm_provider_config()`, `get_llm_config_by_name()`
  - API keys stored here, masked when sent to frontend

**Backend Start:**
- `start_backend.py` - Entry point that runs `uvicorn src.api.backend:app` with auto-reload on port 8000

**Frontend Dev:**
- `frontend/tsconfig.json` - TypeScript strict mode enabled
- `frontend/components.json` - shadcn-svelte component configuration
- Drizzle migrations in `frontend/src/lib/server/db/` for auth

## Platform Requirements

**Development:**
- Conda with Python 3.10 environment
- Node.js v25.6.0 (via nvm)
- SQLite 3 (built-in to Python)
- (Optional) Conda environments for tool execution isolation

**Production:**
- Python 3.10 runtime
- PostgreSQL database (configurable via `DATABASE_URL`)
- Node.js v25.6.0 for frontend builds
- LLM provider API keys (Anthropic, OpenAI, Google, or LM Studio)

---

*Stack analysis: 2025-03-15*
