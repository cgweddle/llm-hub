# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files (Python):**
- Module naming: `snake_case` (e.g., `database.py`, `agent_executor.py`)
- Test files: `test_<module>.py` or `<module>_test.py` (e.g., `test_pydanticai_components.py`, `test_retry.py`)
- Factory files: `<thing>_factory.py` (e.g., `pydanticai_agent_factory.py`, `python_script_tool_factory.py`)
- Converter files: `<target>_tool_converter.py` (e.g., `pydanticai_tool_converter.py`)

**Files (TypeScript/Svelte):**
- Component files: `PascalCase.svelte` (e.g., `ToolNode.svelte`, `AgentBuilder.svelte`, `FullscreenNodeModal.svelte`)
- Library files: `camelCase.ts` (e.g., `api.ts`, `flowBuilder.ts`, `elkLayout.ts`)
- Store files: `camelCase.ts` in `lib/stores/` directory (e.g., `builderMode.ts`, `fullscreenNode.ts`)

**Functions (Python):**
- Function naming: `snake_case` (e.g., `create_agent`, `get_agent_by_id`, `validate_tool_compatibility`)
- Private functions: Prefix with underscore `_` (e.g., `_create_execution`, `_complete_execution`, `_fail_execution`)
- Method naming: `snake_case` (e.g., `convert_tool`, `create_from_node_config`, `validate_agent_config`)

**Functions (TypeScript/Svelte):**
- Function naming: `camelCase` (e.g., `handleParameterClick`, `saveParameterValue`, `updateNodeInternals`)
- Event handlers: `handle<Action>` prefix (e.g., `handleParameterClick`, `handleParameterChange`)
- Reactive assignments: Use `$: { }` blocks (e.g., `$: ({ name, description } = data)`)

**Variables (Python):**
- Variable naming: `snake_case` (e.g., `agent_id`, `input_schema`, `graph_config`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `RETRYABLE_STATUS_CODES`, `DEFAULT_LLM_RETRY_CONFIG`, `AGGRESSIVE_RETRY_CONFIG`)
- Private attributes: Prefix with underscore `_` (e.g., `_tool_cache`, `_filters`, `_name`)

**Variables (TypeScript/Svelte):**
- Variable naming: `camelCase` (e.g., `flowName`, `showValidationToast`, `selectedCondaEnv`)
- Boolean/flag naming: `is<Property>` or `<has><Property>` (e.g., `isConnectable`, `isActive`, `showSaveDialog`)
- Store subscription: `$<storeName>` (e.g., `$builderMode`, `$fullscreenNode`, `llmProvidersStore`)

**Classes (Python):**
- Class naming: `PascalCase` (e.g., `RetryConfig`, `RetryContext`, `PydanticAIAgentFactory`, `AgentExecutor`)
- Mock/Test classes: `Mock<ClassName>` (e.g., `MockSession`, `MockQuery`, `MockTool`, `MockAgent`)

**Types (TypeScript):**
- Interface naming: `PascalCase` (e.g., `Agent`, `Tool`, `Flow`, `User`, `NodeConfig`)
- Type naming: `PascalCase` with descriptive suffix (e.g., `AgentGraphConfig`, `ValidationResult`, `CodeGenerateRequest`)

## Code Style

**Formatting:**
- Python: No explicit formatter configured (rely on PEP 8 conventions)
- TypeScript/Svelte: No explicit formatter configured (follow Prettier conventions informally)
- Indentation: 4 spaces (Python), 2 spaces (TypeScript/Svelte)

**Linting:**
- Python: No explicit linter configured. Type hints used throughout (see `typing` imports in files like `src/database/database.py`, `src/executors/agent_executor.py`)
- TypeScript: `strict: true` enforced in `frontend/tsconfig.json` - enables strict null checks, strict function types, strict property initialization, strict bind call apply, strict class properties

**Comments & Documentation:**
- Module docstrings: Triple-quoted at top of file describing purpose and features (e.g., `src/api/backend.py`, `src/executors/agent_executor.py`)
- Class docstrings: Describe purpose, attributes, and usage (e.g., `RetryConfig`, `AgentExecutor`)
- Function docstrings: Describe purpose, args, returns, and notable behavior
- Inline comments: Used sparingly to explain non-obvious logic
- JSDoc/TSDoc: Not prominently used; comments are plain English

## Import Organization

**Python imports (Order):**
1. Standard library imports (e.g., `import logging`, `from typing import List, Optional`)
2. Third-party imports (e.g., `from pydantic import BaseModel`, `from sqlalchemy.orm import Session`)
3. Local imports (e.g., `from .database import get_agent_by_id`, `from executors.agent_executor import AgentExecutor`)

Examples from `src/executors/agent_executor.py`:
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any, AsyncGenerator, Optional
from sqlalchemy.orm import Session

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from database.database import get_agent_by_id, get_tool_by_id
from database.database_setup import Execution, Message
from utils.retry import (
    RetryConfig,
    retry_async,
    DEFAULT_LLM_RETRY_CONFIG,
)
```

**TypeScript/Svelte imports (Order):**
1. Standard library and framework imports (e.g., `import { writable } from 'svelte/store'`)
2. Component imports (e.g., `import ToolNode from './ToolNode.svelte'`)
3. Library/utility imports (e.g., `import { fullscreenNode } from '$lib/stores/fullscreenNode'`)
4. Type imports (e.g., `import type { NodeProps } from '@xyflow/svelte'`)

**Path Aliases:**
- TypeScript: `$lib` = `frontend/src/lib/` (configured implicitly via SvelteKit)
- No other path aliases detected

## Error Handling

**Patterns (Python):**
- Exceptions: Raised with descriptive messages (e.g., `raise ValueError(f"Agent with ID {agent_id} not found")`)
- Try-except blocks: Used to handle specific exceptions with logging
- Graceful degradation: Import try-except for optional modules (e.g., in `src/executors/agent_executor.py`):
  ```python
  try:
      from utils.retry import RetryConfig, retry_async, DEFAULT_LLM_RETRY_CONFIG
      RETRY_AVAILABLE = True
  except ImportError:
      RETRY_AVAILABLE = False
  ```
- Status fields: Execution records track `status` (running, completed, failed) and `error_message`

**Patterns (TypeScript/Svelte):**
- Promise handling: No explicit try-catch shown; relies on API client (`frontend/src/lib/api.ts`)
- Validation: Inline validation with toast notifications (e.g., `showValidationToast`, `validationMessage`)
- State checking: Defensive checks before accessing properties (e.g., `if (input_schema && typeof input_schema === 'object')`)

## Logging

**Framework:** Python `logging` module (standard library)

**Patterns (Python):**
- Module-level logger: `logger = logging.getLogger(__name__)` at top of each module
- Log levels used: DEBUG (detailed info), INFO (confirmations), WARNING (potential issues), ERROR (failures)
- Examples from `src/executors/agent_executor.py`:
  ```python
  logger.info(f"Executing agent: {agent_record.name} (nodes: {len(graph_config.get('nodes', {}))}, user: {user_id}, stream: {stream})")
  logger.debug(f"Retry enabled: max_retries={self.retry_config.max_retries}, base_delay={self.retry_config.base_delay}s")
  logger.warning("Retry requested but retry utilities not available")
  ```
- File logging: Logs written to `logs/database.log` (created dynamically in `src/database/database.py`)

**Patterns (TypeScript/Svelte):**
- Browser console: `console.log` used sparingly (e.g., in `frontend/src/routes/ToolNode.svelte`: `console.log('Edge was deleted, dismissing validation toast')`)
- No structured logging framework detected

## Function Design

**Size:** No hard limits observed, but larger modules are structured with clear sections:
- `src/api/backend.py` (917 lines) — organized by endpoint groups with clear API model definitions
- `src/executors/agent_executor.py` (568 lines) — organized with clear method separation for creation, execution, message handling

**Parameters:**
- Prefer keyword arguments for optional params (e.g., `stream: bool = False`)
- Use type hints throughout (e.g., `agent_id: int`, `user_id: int`, `input_data: str`)
- Config objects used when many related params (e.g., `RetryConfig` class with multiple optional fields)

**Return Values:**
- Functions return tuples for multiple values (e.g., `Tuple[Callable, Type[BaseModel], Optional[Type[BaseModel]]]` in converter)
- Dict with status/metadata pattern for execution results (e.g., `Dict[str, Any]` with keys like `status`, `result`, `error_message`)
- Optional returns explicitly typed (e.g., `Optional[Dict[str, Any]]`)

## Module Design

**Exports:**
- Factories export a single main class (e.g., `PydanticAIAgentFactory`) plus module-level helper functions (e.g., `create_pydanticai_agent_from_database`)
- Database modules export CRUD functions (e.g., `create_agent`, `get_agent_by_id`, `update_agent`)
- Executors export executor class (e.g., `AgentExecutor`)

**Barrel Files:**
- Not used; individual module imports are typical
- Example: `from src.executors.agent_executor import AgentExecutor` (not `from src.executors import AgentExecutor`)

**Organization by layer:**
- `src/database/` — Data models and CRUD operations
- `src/factories/` — Agent/tool creation from configs
- `src/executors/` — Execution logic (flow, agent, tool)
- `src/converters/` — Format conversion (JSON schema → Pydantic models)
- `src/ai_integrations/` — LLM interactions and code generation
- `src/utils/` — Shared utilities (retry logic)
- `src/validate/` — Validation logic (tool compatibility)
- `frontend/src/lib/` — Shared utilities and API client
- `frontend/src/routes/` — Page-level components and modals

## Type Annotations

**Python:**
- Type hints present throughout (e.g., `def create_agent(session: Any, user_id: int, name: str, description: str, graph_config: Dict, output_schema: Dict = None) -> Agent`)
- Generic types used (e.g., `Dict[str, Any]`, `List[int]`, `Optional[str]`, `Tuple[Callable, Type[BaseModel], Optional[Type[BaseModel]]]`)
- `typing` module imports include: `List`, `Optional`, `Dict`, `Any`, `Callable`, `Type`, `Union`, `Set`, `Tuple`, `AsyncGenerator`

**TypeScript:**
- Strict mode enabled (`strict: true` in tsconfig.json)
- Interface definitions for all major data types (e.g., `Agent`, `Tool`, `Flow`, `User`)
- Type imports with `type` keyword (e.g., `import type { NodeProps } from '@xyflow/svelte'`)
- Generic types used (e.g., `Writable<LLMProvider[]>`)

---

*Convention analysis: 2026-03-15*
