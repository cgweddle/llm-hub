# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files:**
- Python modules: lowercase_with_underscores (e.g., `database.py`, `agent_executor.py`)
- TypeScript modules: camelCase for files (e.g., `api.ts`, `flowBuilder.ts`)
- Svelte components: PascalCase (e.g., `AgentBuilder.svelte`, `ToolNode.svelte`)
- Test files: `test_*.py` or `*.test.ts` pattern
- Factory files: `*_factory.py` naming convention (e.g., `pydanticai_agent_factory.py`)

**Functions and Methods:**
- Python: snake_case (e.g., `create_agent()`, `get_session()`, `validate_tool_compatibility()`)
- TypeScript/JavaScript: camelCase (e.g., `buildEnhancedGraphConfig()`, `determineMappingFromEdge()`)
- Async functions: prefix with `async` keyword or named to clearly indicate async behavior
- Private/internal functions: no special prefix, but contextually clear (e.g., `_get_downstream_nodes()`)

**Variables:**
- Python: snake_case (e.g., `graph_config`, `tool_ids`, `entry_point`)
- TypeScript/JavaScript: camelCase (e.g., `nodesConfig`, `edgesConfig`, `toolId`)
- Constants: UPPERCASE_WITH_UNDERSCORES (e.g., `DEFAULT_LLM_RETRY_CONFIG`, `RETRYABLE_STATUS_CODES`)
- Database objects: descriptive names reflecting domain (e.g., `agent_record`, `tool`, `execution`)

**Types and Interfaces:**
- TypeScript: PascalCase (e.g., `AgentCreate`, `GraphConfig`, `NodeConfig`)
- Python Type hints: use full qualified names (e.g., `Dict[str, Any]`, `Optional[str]`)
- Interface naming: descriptive of the contract (e.g., `AgentGraphConfig`, `EdgeMapping`)

## Code Style

**Formatting:**
- Python: Follow PEP 8 conventions
  - 4-space indentation
  - Max line length: implicit (no explicit limit enforced)
  - Import statements at top of file organized by standard/third-party/local
- TypeScript/JavaScript: SvelteKit conventions
  - 2-space indentation (per `frontend/package.json` build tools)
  - strict TypeScript enabled (`strict: true` in `tsconfig.json`)
  - ESModule imports preferred

**Linting:**
- Python: No explicit eslint/flake8 config files found
  - Manual adherence to PEP 8 and project patterns
  - Type hints used extensively (e.g., `from typing import Dict, Any, Optional`)
- TypeScript: TypeScript strict mode enabled
  - `checkJs: true` for JavaScript validation
  - `forceConsistentCasingInFileNames: true` enforced
  - Source maps enabled for debugging

## Import Organization

**Python:**
1. Standard library imports (e.g., `import os`, `import sys`, `from typing import`)
2. Third-party imports (e.g., `from fastapi import`, `from sqlalchemy import`)
3. Local imports (e.g., `from src.database import`, `from src.utils import`)
4. Path manipulation for imports (e.g., `sys.path.append()`) placed near imports when needed

**TypeScript/JavaScript:**
1. Third-party library imports (e.g., `import type from '@xyflow/svelte'`)
2. Local module imports (e.g., `import type { GraphConfig } from './api'`)
3. Type imports: use `import type` for type-only imports to avoid runtime overhead
4. Relative paths preferred for local imports

**Path Aliases:**
- TypeScript: Uses relative imports, no configured aliases in `tsconfig.json`
- Python: Explicit relative imports with `sys.path.append()` when needed; absolute imports within package using `from src.*`

## Error Handling

**Patterns:**

**Python:**
- Exceptions: Raise descriptive errors with context
  ```python
  # src/executors/agent_executor.py
  if not agent_record:
      raise ValueError(f"Agent with ID {agent_id} not found")

  if not graph_config:
      raise ValueError(f"Agent {agent_id} has no graph_config")
  ```
- Try/finally blocks for resource cleanup: `try: ... finally: session.close()`
- Logging on error with context (see `src/database/database.py` logging patterns)
- Exception handling in async contexts: catch and re-wrap as `RuntimeError` with message context

**TypeScript:**
- Error handling in API calls: typically returns error in response object
- Type safety: interfaces define success and error states (e.g., `FlowExecutionResult`)
- Error messages: descriptive validation errors thrown from utility functions
  ```typescript
  // src/lib/flowBuilder.ts
  if (entryNodes.length === 0) {
    throw new Error('No entry point found in flow - all nodes have incoming edges');
  }
  ```

## Logging

**Framework:** Python uses `logging` module (standard library)

**Patterns:**

**Python logging:**
- Logger setup per module: `logger = logging.getLogger(__name__)`
- Log levels used:
  - `DEBUG`: Detailed information (e.g., "getting environment variable")
  - `INFO`: Confirmations (e.g., "Database session created successfully")
  - `WARNING`: Non-fatal issues (e.g., "Streaming not supported for multi-node agents")
  - `ERROR`: Errors and failures (e.g., "Agent execution failed")
- Contextual information in messages: include IDs, names, and operation details
  ```python
  # src/executors/agent_executor.py
  logger.info(f"Executing agent: {agent_record.name} (nodes: {len(graph_config.get('nodes', {}))}, user: {user_id}, stream: {stream})")
  ```
- File logging: setup in `src/database/database.py` writes to `logs/database.log`

**TypeScript logging:**
- `console.log()`, `console.warn()`, `console.error()` used directly
- No centralized logging framework in frontend code

## Comments

**When to Comment:**
- Module level: brief docstring at top of file describing purpose
- Class level: docstrings for public classes explaining purpose and key attributes
- Function level: docstrings for public functions, especially with complex logic
- Inline comments: rare; used only for non-obvious logic or important design decisions
- Code comments focus on "why" not "what"

**JSDoc/TSDoc:**
- Python: Uses docstring format with sections (Args, Returns, Example)
  ```python
  def calculate_delay(self, attempt: int) -> float:
      """
      Calculate delay for a given retry attempt using exponential backoff.

      Args:
          attempt: The current attempt number (0-indexed)

      Returns:
          Delay in seconds

      Example:
          >>> config = RetryConfig(base_delay=1.0, exponential_base=2)
          >>> config.calculate_delay(0)  # ~1 second
      """
  ```
- TypeScript: Uses JSDoc comments for functions
  ```typescript
  /**
   * Convert @xyflow visual nodes/edges to graph_config
   */
  export function buildEnhancedGraphConfig(
    nodes: XYFlowNode[],
    edges: XYFlowEdge[],
    tools: Tool[]
  ): GraphConfig {
  ```

## Function Design

**Size Guidelines:**
- Python: Functions typically 20-50 lines; longer functions broken into private helpers
- TypeScript: Similar range; utility functions kept focused on single responsibility

**Parameters:**
- Python: Explicit parameter names, type hints used (e.g., `session: Session`, `user_id: int`)
- TypeScript: Explicit typing (e.g., `nodes: XYFlowNode[]`, `config: RetryConfig`)
- Required vs optional: optional parameters come at end, marked with `Optional` or `=` default
- Configuration objects: prefer dict/object parameters over multiple flags
  ```python
  # src/utils/retry.py - uses RetryConfig object instead of multiple parameters
  def retry_sync(
      func: Callable[..., T],
      config: Optional[RetryConfig] = None,
      on_retry: Optional[Callable[[int, Exception, float], None]] = None,
  ) -> T:
  ```

**Return Values:**
- Python: Single return type clearly specified in type hint and docstring
- TypeScript: Type explicitly defined (e.g., `GraphConfig`, `Dict<string, Any>`)
- Early returns: used for error cases and guard clauses
  ```python
  # src/executors/agent_executor.py
  if not agent_record:
      raise ValueError(f"Agent with ID {agent_id} not found")
  ```

## Module Design

**Exports:**
- Python: All public classes/functions defined at module top level; no `__all__` explicitly used
- TypeScript: Use `export` keyword for public APIs; type exports use `export type` or `export interface`
  ```typescript
  export interface Agent {
    id: number;
    name: string;
    // ...
  }
  ```

**Barrel Files:**
- Python: No barrel exports pattern; direct imports from modules (e.g., `from src.database.database import get_session`)
- TypeScript: `index.ts` files used in some directories (frontend components), but mostly direct module imports

**Module Organization:**
- Related functionality grouped in single file or package:
  - `src/executors/`: Each executor type (agent, flow, tool) in separate file
  - `src/factories/`: Each factory pattern implementation in separate file
  - `src/database/`: Database models in `database_setup.py`, CRUD in `database.py`
  - Frontend: Utilities grouped by concern (api.ts for API communication, flowBuilder.ts for graph conversion)

---

*Convention analysis: 2026-03-15*
