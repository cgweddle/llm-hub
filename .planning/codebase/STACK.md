# Technology Stack

**Analysis Date:** 2025-03-15

## Languages

**Primary:**
- Python 3.10 - Backend API, executors, agents, tools, database management
- TypeScript 5.x - Frontend SvelteKit application, client-side logic, API communication
- JavaScript/Node.js v25.6.0 - Build tooling, package management

**Secondary:**
- YAML - LLM provider configuration (`~/.llm_hub/config.yaml`)
- SQL - Database queries via SQLAlchemy ORM

## Runtime

**Environment:**
- Python 3.10 (conda environment `llm-hub`)
- Node.js v25.6.0 (see `frontend/.nvmrc`)

**Package Managers:**
- npm (Node) - lockfile present in `frontend/package-lock.json`
- pip (Python) - lockfile not found; uses `requirements.txt` for reproducibility

## Frameworks

**Backend:**
- **FastAPI** - REST API server with streaming support via `StreamingResponse`
- **Uvicorn** - ASGI web server for FastAPI, auto-reload on development
- **SQLAlchemy** - ORM for database models and relationships, supports both SQLite and PostgreSQL

**Frontend:**
- **SvelteKit** 2.16.0 - Full-stack web framework with server-side rendering and static generation
- **Svelte** 5.0.0 - Reactive UI component framework
- **Tailwind CSS** 4.1.17 - Utility-first CSS framework with typography plugin
- **shadcn-svelte** 1.0.12 - Component library built on Bits UI and Tailwind

**AI/Agent Frameworks:**
- **PydanticAI** 0.0.13+ - LLM agent framework with structured output and tool calling
- **Google ADK** (gen-ai library) - ReAct-style agent implementation with tool use
- **LangChain** - Legacy agent framework; used for tool definitions and embeddings

**Testing:**
- **pytest** - Python test runner

**Code Editing:**
- **CodeMirror** 6.x - In-browser code editor for Python with syntax highlighting, one-dark theme

**Visualization & Layout:**
- **@xyflow/svelte** 1.2.4 - Interactive node-and-edge visual flow editor
- **elkjs** 0.11.0 - Auto-layout engine for flow nodes (ELK algorithm)

**Build & Dev:**
- **Vite** 6.0.0 - Frontend build tool and dev server
- **TypeScript** 5.0.0 - Type checking
- **Svelte-check** 4.0.0 - Svelte-specific type checking

## Key Dependencies

**Backend (from `requirements.txt`):**
- `fastapi` - REST API framework
- `uvicorn` - ASGI server
- `pydantic[email]` - Data validation with email support
- `pydantic-ai>=0.0.13` - Agent framework with LLM integration
- `langchain` - Legacy agent/tool framework
- `langgraph` - LangChain graph execution
- `langchain-aws` - AWS integrations for LangChain
- `google-adk` - Google ADK for ReAct agents
- `httpx>=0.27.0` - Async HTTP client for external integrations
- `passlib` - Password hashing (bcrypt) for user authentication
- `pytest` - Test framework

**Frontend (from `package.json`):**
- `@xyflow/svelte` 1.2.4 - Flow diagram builder
- `elkjs` 0.11.0 - Auto-layout for graphs
- `codemirror` 6.0.2 + plugins - Code editor with Python language support
- `lucia` 3.2.2 - Authentication library
- `@lucia-auth/adapter-sqlite` 3.0.2 - Lucia SQLite session adapter
- `better-sqlite3` 12.5.0 - Embedded SQLite for Lucia auth and local sessions
- `@node-rs/argon2` 2.0.2 - Argon2 password hashing
- `drizzle-orm` (schema only) - Type-safe SQL builder (schema defined in `frontend/src/lib/server/db/schema.ts`)
- `shiki` 3.17.0 - Code syntax highlighting library
- `mode-watcher` 1.1.0 - Dark mode detection and switching
- `bits-ui` 2.14.4 - Headless component library

## Configuration

**Environment Variables:**

Backend:
- `DATABASE_URL` - Database connection string (defaults to `sqlite:///database/llm_hub.db`)
  - Supports SQLite: `sqlite:///path/to/db.db`
  - Supports PostgreSQL: `postgresql://user:password@host/dbname`

Frontend:
- `DATABASE_URL` - Local SQLite database path for Lucia auth (defaults to `local.db`)

LLM Configuration:
- Stored in `~/.llm_hub/config.yaml` (user home directory, not in repo)
- YAML format with `models:` list containing LLM provider configs
- Each model config has: `name`, `provider`, `model`, `api_key`, `base_url`
- Supported providers: `anthropic`, `openai`, `gemini`, `lmstudio`
- API keys are masked in API responses to prevent exposure to frontend

Tool Environment:
- `BRAVE_SEARCH_API_KEY` - For web search tools (optional)
- Python tools may run in isolated conda environments specified per flow

**Build Configuration:**
- `frontend/vite.config.ts` - Vite bundler configuration
- `frontend/tsconfig.json` - TypeScript compiler options with strict mode
- `frontend/tailwind.config.js` - Tailwind CSS configuration
- `frontend/postcss.config.js` - PostCSS with Tailwind support
- `frontend/svelte.config.js` - SvelteKit adapter and options

**Database Configuration:**
- `src/database/database_setup.py` - SQLAlchemy model definitions and database manager
- Database auto-creates tables on first run via `DatabaseManager`
- Supports both SQLite (development) and PostgreSQL (production)

## Platform Requirements

**Development:**
- Python 3.10 with conda environment
- Node.js v25.6.0
- Git
- Shell access for conda environment activation
- Local file system access for `~/.llm_hub/config.yaml`

**Production:**
- Python 3.10 runtime
- Node.js v25.6.0 (for frontend builds; runtime only if using SvelteKit adapter)
- PostgreSQL database (recommended) or SQLite
- LLM provider credentials (Anthropic, OpenAI, Google, or LM Studio)
- For tools: optional conda installation for isolated Python execution

**Browser Support:**
- Modern browsers supporting ES2022, WebSockets (for future streaming), ReadableStream API
- CSS Grid, CSS custom properties (Tailwind v4)

---

*Stack analysis: 2025-03-15*
