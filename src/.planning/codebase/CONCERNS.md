# Codebase Concerns

**Analysis Date:** 2026-03-15

## Security Issues

**Hardcoded CORS origins:**
- Issue: Localhost addresses hardcoded into FastAPI middleware. Will fail in production.
- Files: `src/api/backend.py` lines 35-42
- Impact: Dev origins exposed in production, or CORS will break in production if not updated
- Fix approach: Move CORS_ORIGINS to environment variables with sensible defaults

**Missing authentication/authorization:**
- Issue: All endpoints accept user_id as query parameter with no validation that user_id matches authenticated user
- Files: `src/api/backend.py` endpoints like `/agents/`, `/tools/`, `/flows/` (lines 256-790)
- Impact: Users can access/modify other users' data by changing user_id parameter
- Fix approach: Implement JWT or session-based auth; extract user_id from token, not request params

**Code execution via eval():**
- Issue: Type string evaluation uses eval() with restricted builtins but still vulnerable
- Files: `src/executors/tool_executor.py` line 107
- Impact: Potential code injection if tool code contains malicious type strings
- Fix approach: Use ast.literal_eval() for safe type parsing, or whitelist allowed types

**Subprocess execution without proper path validation:**
- Issue: Conda environment path passed directly to subprocess.run() without validation
- Files: `src/executors/tool_executor.py` line 217
- Impact: Path traversal attacks possible; arbitrary code execution in wrong environment
- Fix approach: Validate conda_env path is within expected directory; use absolute paths only

**User input to LLM without sanitization:**
- Issue: Tool descriptions, agent names, flow inputs passed directly to LLM prompts
- Files: `src/ai_integrations/generate_python_tools.py` lines 70-71
- Impact: Prompt injection attacks possible; LLM could generate malicious code
- Fix approach: Implement prompt injection detection; sanitize/escape user inputs in prompts

**Pickle deserialization from untrusted subprocess output:**
- Issue: Tool output unpickled directly from subprocess stdout without validation
- Files: `src/executors/tool_executor.py` line 236
- Impact: Malicious pickled objects could execute arbitrary code during unpickling
- Fix approach: Use safer serialization (JSON); implement pickle allowlist if pickle required

## Authentication & Access Control

**No password verification endpoint:**
- Issue: Users created but no login endpoint; password hashing happens but is never verified
- Files: `src/api/backend.py` line 454, no login endpoint
- Impact: No way to authenticate users; anyone knowing user_id can access their data
- Fix approach: Implement POST /login endpoint with password verification; return JWT token

**Public flows/agents have no access control:**
- Issue: is_public flag controls visibility but no endpoint checks file permissions
- Files: `src/database/database.py` public query functions
- Impact: Private flows might be accessible via direct ID if endpoints don't validate ownership
- Fix approach: Add ownership verification on all GET/PATCH/DELETE endpoints before returning data

## Data Validation & Input Handling

**No validation of graph_config structure:**
- Issue: graph_config JSON accepted without schema validation for nodes/edges/entry_point
- Files: `src/api/backend.py` line 62 (AgentCreate), line 154 (FlowCreate)
- Impact: Malformed graphs could crash executor; invalid node references cause executor failures
- Fix approach: Create Pydantic models for GraphConfig with validation; validate in endpoint

**No input_schema validation for tool execution:**
- Issue: Flow/agent execution accepts arbitrary input_data without validating against tool input schemas
- Files: `src/api/backend.py` line 792 (FlowExecuteRequest), line 86 (AgentExecuteRequest)
- Impact: Type mismatches could crash tools; string concatenation with wrong types fails
- Fix approach: Validate input against tool.input_schema before execution

**Missing JSON bounds checking:**
- Issue: Large JSON payloads (graph_config, script_code) accepted without size limits
- Files: `src/api/backend.py` all POST endpoints accepting JSON
- Impact: Memory exhaustion via large graph_config or tool script uploads
- Fix approach: Add request size limit to FastAPI; validate JSON field sizes

## Performance & Scaling Issues

**Blocking file I/O in subprocess execution:**
- Issue: Subprocess execution writes temporary files and waits synchronously
- Files: `src/executors/tool_executor.py` lines 209-225
- Impact: Large tool execution can block FastAPI event loop; multiple tools cause cascading delays
- Fix approach: Use asyncio.to_thread() for subprocess calls; implement connection pooling for subprocess workers

**No connection pooling for SQLite:**
- Issue: SQLite configured with check_same_thread=False but no pool management
- Files: `src/database/database_setup.py` lines 176-182
- Impact: Multiple threads accessing same SQLite database; potential write contention
- Fix approach: Use PostgreSQL for production; if SQLite, serialize writes or use WAL mode

**Database session never explicitly rolled back on error:**
- Issue: Exceptions in endpoints don't rollback uncommitted changes
- Files: `src/database/database.py` create/update functions, no try/except/rollback
- Impact: Partial writes on error; database inconsistency if creation fails mid-transaction
- Fix approach: Wrap all session operations in try/except/rollback; use context managers

**Memory leak in streaming handlers:**
- Issue: async generators in stream_generator() don't guarantee cleanup if client disconnects
- Files: `src/api/backend.py` lines 313-340, 387-405
- Impact: Orphaned async tasks pile up if users cancel streams; resources leak
- Fix approach: Use try/finally in stream generators; implement connection timeouts

## Error Handling Issues

**Generic Exception catching with poor messages:**
- Issue: Broad except Exception clauses that swallow specific error info
- Files: `src/executors/flow_executor.py` lines 518-526, `src/api/backend.py` line 803
- Impact: Difficult to debug; same error response for all failures (network, syntax, timeout)
- Fix approach: Catch specific exceptions (TimeoutError, SyntaxError, etc.); provide detailed error types to client

**Subprocess errors lost in stderr capture:**
- Issue: Tool execution failures only return stderr string; original exception context lost
- Files: `src/executors/tool_executor.py` line 232
- Impact: Stack traces and cause information unavailable; hard to debug tool failures
- Fix approach: Parse stderr for traceback; include execution environment info in error

**No timeout on graph traversal:**
- Issue: BFS loop in _execute_graph runs until nodes exhausted; infinite loops if graph has cycles without visit tracking
- Files: `src/executors/agent_executor.py` lines 216-280
- Impact: Agent execution can hang indefinitely if loop_edges misconfigured
- Fix approach: Add max_iterations overall limit separate from loop_edges limit; track visited node combinations

## Fragile Areas

**Tight coupling between tool schema and flow execution:**
- Issue: Tool input_schema must exactly match node output format; no type coercion or adaptation
- Files: `src/executors/flow_executor.py` line 96, `src/executors/agent_executor.py` line 225
- Impact: Minor schema changes break flows; no backward compatibility
- Fix approach: Implement adapter layer for type conversion; handle common transformations (dict→JSON string, etc.)

**AST parsing for function extraction is brittle:**
- Issue: Reliance on ast.get_docstring() and lineno for code extraction; formatting sensitive
- Files: `src/factories/python_script_tool_factory.py` lines 87-100
- Impact: Tools with complex formatting or multi-line signatures extract incorrectly
- Fix approach: Use getsource() from inspect module; add validation that extracted code is valid Python

**Graph cycle detection insufficient for multi-loop workflows:**
- Issue: loop_counts tracks individual edges but doesn't prevent complex cycle patterns
- Files: `src/executors/agent_executor.py` line 268
- Impact: Workflows with multiple interdependent loops can exceed max_loop_iterations unexpectedly
- Fix approach: Track visited (node, iteration) combinations; implement topological sort validation on graph load

**Conda environment paths not validated:**
- Issue: Flow conda_env path used directly without checking existence or permissions
- Files: `src/executors/flow_executor.py` line 72
- Impact: Tools fail at runtime with cryptic conda errors if environment missing
- Fix approach: Validate conda environment exists at flow creation time; cache environment list

## Missing Critical Features

**No API key masking for public agents/flows:**
- Issue: LLM provider config (api_key, base_url) not masked before returning in public agent details
- Files: `src/api/backend.py` agent endpoints
- Impact: Public agents/flows could expose internal API keys if config exposed
- Fix approach: Strip sensitive fields from responses; separate internal config from public view

**No resource quotas per user:**
- Issue: Users can create unlimited flows/tools/agents; no storage/compute limits
- Files: `src/database/database.py` create functions
- Impact: Denial of service via excessive resource creation; storage exhaustion
- Fix approach: Implement quota system; track per-user resource counts; enforce limits on creation

**No execution history cleanup:**
- Issue: Execution records and messages accumulate indefinitely in database
- Files: `src/database/database_setup.py` Execution and Message models
- Impact: Database grows unbounded; old execution records slow down queries
- Fix approach: Implement retention policy; add cleanup task to delete old executions

**No rollback/undo for agent/tool edits:**
- Issue: Updates overwrite previous versions with no version history
- Files: `src/api/backend.py` PATCH endpoints
- Impact: User mistakes (editing wrong tool) are permanent; can't restore previous version
- Fix approach: Implement soft deletes or version history table

## Test Coverage Gaps

**No integration tests for streaming execution:**
- Issue: Streaming endpoints tested only conceptually; actual SSE stream behavior untested
- Files: `src/api/backend.py` /agents/{agent_id}/execute (streaming branch)
- Risk: Client disconnection handling, partial message handling not validated
- Priority: High

**No tests for cycle detection edge cases:**
- Issue: Cyclic graphs, self-loops, and multi-loop workflows not tested
- Files: `src/executors/agent_executor.py` _execute_graph
- Risk: Infinite loops possible in production
- Priority: High

**No security tests for path traversal/injection:**
- Issue: Subprocess paths, tool script injection, and prompt injection not tested
- Files: `src/executors/tool_executor.py`, `src/ai_integrations/generate_python_tools.py`
- Risk: Security vulnerabilities silently pass
- Priority: Critical

**No database concurrent access tests:**
- Issue: Multiple simultaneous user/flow creations not tested with SQLite
- Files: `src/database/database_setup.py`
- Risk: Race conditions in production
- Priority: Medium

## Dependencies at Risk

**No pinned versions in requirements:**
- Issue: LangGraph, PydanticAI, and LLM client versions likely unpinned or version ranges loose
- Impact: Breaking changes in minor updates could break execution
- Migration plan: Pin all critical dependencies; implement version testing in CI

**Pickle serialization for tool IPC:**
- Issue: Subprocess communication uses pickle which is inherently unsafe
- Impact: Malicious tool code could craft pickle exploits
- Migration plan: Migrate to JSON serialization; implement strict schema validation

## Known Issues

**Frontend hardcoded API_BASE_URL:**
- Symptoms: Frontend fails when backend not on localhost:8000
- Files: `frontend/src/lib/api.ts` line 2
- Workaround: Backend must run on localhost:8000; cannot deploy to different host
- Fix approach: Make API_BASE_URL configurable; accept from environment or API discovery

**Duplicate database initialization:**
- Symptoms: DatabaseManager and get_session() both exist; unclear which should be used
- Files: `src/database/database_setup.py` (DatabaseManager), `src/database/database.py` (get_session)
- Workaround: None; risk of using wrong initialization
- Fix approach: Consolidate to single initialization pattern

---

*Concerns audit: 2026-03-15*
