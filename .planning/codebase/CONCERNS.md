# Codebase Concerns

**Analysis Date:** 2026-03-15

## Tech Debt

**Monolithic Backend API:**
- Issue: Single large file `src/api/backend.py` (917 lines) contains all FastAPI endpoints without separation of concerns. No route grouping, no middleware layering, all CRUD and execution logic inline.
- Files: `src/api/backend.py`
- Impact: Difficult to maintain, test, and extend. Adding new endpoints requires modifying a large file. No clear separation between request handling and business logic.
- Fix approach: Split into route modules (agents.py, tools.py, flows.py, users.py in an api/routes directory). Extract CRUD operations to dedicated service layer. Implement middleware for common cross-cutting concerns.

**Missing Input Validation:**
- Issue: API endpoints accept user inputs without comprehensive validation. User-provided Python code in tools, flow configs, and agent configs are executed without verification.
- Files: `src/api/backend.py` (endpoints), `src/executors/tool_executor.py`, `src/executors/flow_executor.py`
- Impact: Security risk. Malicious code injection through tool creation. No schema validation for graph_config structures before execution.
- Fix approach: Add Pydantic validators for all request models. Implement AST analysis to verify tool code safety before execution. Validate graph_config structure (required fields, valid node references, acyclic validation for non-loop edges).

**Unsafe eval() Usage:**
- Issue: `eval()` call in `src/executors/tool_executor.py:107` evaluates type strings dynamically using restricted builtins but still presents a code injection surface.
- Files: `src/executors/tool_executor.py:107` (`eval(type_str, {"__builtins__": {}}, type_namespace)`)
- Impact: Type resolution could be exploited. Although builtins are restricted, the type_namespace is user-controlled via tool scripts.
- Fix approach: Replace eval() with a safe type registry mapping. Pre-define allowed types and validate against them before resolution. Use ast.literal_eval() for literal values.

**Database Credentials in Environment:**
- Issue: Database URL stored in `DATABASE_URL` env var. No encryption at rest for sensitive connection strings. `.env` file (if used) not reviewed for secrets handling.
- Files: `src/database/database.py:48`, `src/database/database_setup.py:180-188`
- Impact: If .env is committed or deployed improperly, database credentials are exposed. No rotation mechanism.
- Fix approach: Use secure secrets management (AWS Secrets Manager, HashiCorp Vault, or environment-specific config). Document .env in .gitignore. Never commit credentials.

**No Authentication/Authorization Enforcement:**
- Issue: `user_id` is a required parameter in all endpoints but is client-provided and never verified against session/JWT. Any client can pass any user_id.
- Files: `src/api/backend.py` (all user-scoped endpoints like lines 257, 273, 478, 616, 745, etc.)
- Impact: Critical security flaw. Users can access/modify/execute agents, tools, and flows belonging to other users. Users can impersonate each other.
- Fix approach: Implement OAuth2/JWT authentication. Decode token in dependency injection to extract actual user_id. Enforce user_id validation on all DB queries. Use ownership checks before allowing modifications.

**No Pagination on List Endpoints:**
- Issue: Endpoints like `/agents/`, `/tools/`, `/flows/` return all records for a user without limit or offset parameters.
- Files: `src/api/backend.py` lines 272-278 (get_user_agents), 616-619 (get_user_tools), 768-771 (get_user_flows)
- Impact: Performance degradation with large datasets. Frontend could load thousands of records. Database query timeout risk.
- Fix approach: Add limit/offset or cursor-based pagination. Default limit of 50, max 200. Include total count in response.

**Inconsistent Error Handling:**
- Issue: Mix of try-except-pass blocks, bare except clauses, and generic HTTPException responses. Some errors logged, some swallowed. No error standardization.
- Files: `src/executors/tool_executor.py:108-109` (bare except with pass), `src/executors/agent_executor.py:133-136`, `src/api/backend.py` (inconsistent HTTPException details)
- Impact: Silent failures make debugging difficult. Inconsistent error responses break client error handling. Some user-facing errors leak internal details.
- Fix approach: Define custom exception hierarchy (ValidationError, NotFoundError, ExecutionError, InternalError). Implement error handler middleware to format all errors consistently. Log all exceptions with context before returning to client.

---

## Known Bugs

**Agent Execution with Invalid graph_config:**
- Symptoms: Agent executor crashes or hangs when graph_config is missing required fields (entry_point, nodes, edges).
- Files: `src/executors/agent_executor.py:187-188` (check happens but error may not propagate correctly)
- Trigger: Create agent with incomplete graph_config, attempt execution
- Workaround: Validate graph_config structure before saving to database

**Tool Execution Timeout Not Enforced for Non-Conda Tools:**
- Symptoms: Tool execution can hang indefinitely if tool function gets stuck (infinite loop, waiting for external resource).
- Files: `src/executors/tool_executor.py` (no timeout on subprocess for non-conda execution), vs. `src/executors/tool_executor.py:225` (conda execution has 300s timeout)
- Trigger: Create tool with blocking I/O or infinite loop, execute it
- Workaround: Manually kill the Python process. Restart backend.

**Missing Agent Type Validation:**
- Symptoms: Graph executor assumes agent_type field exists in node config. If missing, execution falls through without error.
- Files: `src/executors/agent_executor.py:260-290` (agent type detection and factory dispatch)
- Trigger: Create agent node without agent_type field in graph_config
- Workaround: Always set agent_type to 'pydanticai' or 'react' when creating agents

---

## Security Considerations

**Arbitrary Python Code Execution:**
- Risk: Tools allow users to upload arbitrary Python scripts. Executed directly in subprocess or conda env without sandboxing.
- Files: `src/executors/tool_executor.py:18-111` (parse_imports_and_classes), `src/factories/python_script_tool_factory.py` (AST parsing but no execution restrictions)
- Current mitigation: Conda environment isolation (optional). No other sandboxing.
- Recommendations: Implement code scanning before tool creation (check for dangerous imports like os, subprocess, socket). Use allowlist for permitted imports. Consider running tools in restricted containers. Add audit logging for all tool executions.

**API Keys in Environment/Config:**
- Risk: LLM API keys stored in `~/.llm_hub/config.yaml`. Frontend could receive masked keys but subprocess tool execution reads keys directly.
- Files: `src/executors/tool_executor.py:140-175` (reads ANTHROPIC_API_KEY, OPENAI_API_KEY from environment), `src/utils/llm_config.py`
- Current mitigation: Keys masked before sending to frontend. Not persisted in database.
- Recommendations: Never expose keys to subprocess. Use secure credential passing (secret files, credential helpers). Rotate keys regularly. Audit log all API key access.

**No Rate Limiting:**
- Risk: Backend endpoints have no rate limiting. Agent/tool execution could be DOS'd by rapid requests.
- Files: `src/api/backend.py` (no rate limiting middleware)
- Current mitigation: None
- Recommendations: Add rate limiting middleware (FastAPI Limiter). Per-user limits. Distinguish between user actions and tool executions.

**Pickle Deserialization:**
- Risk: `src/executors/tool_executor.py:236` uses `pickle.loads()` on subprocess output. Pickle can execute arbitrary code if crafted maliciously.
- Files: `src/executors/tool_executor.py:236` (pickle.loads on stdout)
- Current mitigation: Output comes from controlled subprocess we spawned, not untrusted network
- Recommendations: Use JSON serialization instead of pickle. If pickle required, validate source before deserialization. Use pickle with restricted loader (restrict_types=True in Python 3.8+).

**Streaming Response Error Handling:**
- Risk: SSE streaming responses in `src/api/backend.py:313-336` catch exceptions and yield JSON errors, but incomplete streams could leave client in inconsistent state.
- Files: `src/api/backend.py:313-336` (stream_generator), `src/api/backend.py:387-401` (system_prompt_stream)
- Current mitigation: Try-catch around stream, errors sent as JSON chunks
- Recommendations: Implement heartbeat mechanism to detect stream dropout. Client-side validation that stream is complete. Add stream session ID for tracing.

---

## Performance Bottlenecks

**Synchronous Database Operations in Async Context:**
- Problem: `AgentExecutor._execute_graph()` uses `self.session.query()` (synchronous SQLAlchemy) in async function. Blocks event loop.
- Files: `src/executors/agent_executor.py:99, 156, 163, 237` (get_agent_by_id calls in async functions)
- Cause: Database wrapper `src/database/database.py` uses synchronous sessions, not async sessions.
- Improvement path: Migrate to sqlalchemy async (AsyncSession). Use await in async functions. This requires Database manager refactor.

**No Query Optimization (N+1 Queries):**
- Problem: Loading agents with tools requires separate query for each tool. Loading flows with agents similar issue.
- Files: `src/database/database.py:87-110` (create_agent doesn't prefetch related data)
- Cause: SQLAlchemy relationships not eagerly loaded via joinedload/selectinload
- Improvement path: Use `selectinload()` on relationship queries. Measure with sqlalchemy echo=true.

**Large Execution Trace in Memory:**
- Problem: `execution_trace` list in FlowExecutor and AgentExecutor grows unbounded during execution. No cleanup for long-running executions.
- Files: `src/executors/agent_executor.py:208, 237-240` (execution_trace), `src/executors/flow_executor.py:36, 141-151` (execution_trace)
- Cause: All node outputs stored in memory for every execution
- Improvement path: Stream trace entries to database incrementally. Keep only recent N entries in memory.

**No Caching of Tool Schemas:**
- Problem: Tool input/output schemas parsed from database on every execution. Type resolution happens repeatedly.
- Files: `src/executors/flow_executor.py:38-100` (_prepare_tools), `src/executors/tool_executor.py:70-111` (eval_type_string)
- Cause: No schema caching layer
- Improvement path: Cache tool metadata (schemas, type namespace) at executor creation. Invalidate on tool update.

---

## Fragile Areas

**Complex Graph Executor Logic:**
- Files: `src/executors/agent_executor.py:167-290` (_execute_graph method, 123 lines)
- Why fragile: BFS traversal with cycle support, loop count tracking, multi-node graph handling. No clear state machine. Multiple branches for agent vs tool nodes. Complex input/output mapping between nodes.
- Safe modification: Add comprehensive tests for each graph topology (linear, branching, cyclic, multi-exit). Test with mock agents/tools. Add debug logging for each traversal step. Don't refactor control flow without tests.
- Test coverage: Unit tests exist in `tests/test_pydanticai_components.py` but focus on agent creation, not graph execution. Graph traversal logic largely untested.

**Type Resolution in Tool Execution:**
- Files: `src/executors/tool_executor.py:70-111` (eval_type_string), `src/factories/python_script_tool_factory.py` (type hint extraction via AST)
- Why fragile: Relies on eval() with restricted namespace. Complex type strings (generic types, custom classes) can fail silently. Namespace construction depends on successful imports of arbitrary user modules.
- Safe modification: Add comprehensive tests for type resolution edge cases (List[Dict[str, int]], Optional[CustomClass], etc.). Log all type evaluation failures. Consider using typeguard library for runtime validation instead of eval.
- Test coverage: `tests/test_python_script_tool_factory.py` has some coverage but gaps on complex types.

**Agent/Tool Factory Dispatch:**
- Files: `src/executors/agent_executor.py:260-290` (agent type detection and factory routing), `src/factories/agent_factory.py` (Google ADK), `src/factories/pydanticai_agent_factory.py` (PydanticAI)
- Why fragile: Two separate agent implementations (Google ADK ReAct, PydanticAI) with different capabilities. Graph executor must route to correct factory based on agent_type. No interface abstraction.
- Safe modification: Define AgentFactory protocol/interface that both implementations satisfy. Add factory tests with mock agents. Document expected behavior per factory. Add logging to trace factory selection.
- Test coverage: Separate test files for each agent type but no integration tests for dispatch logic.

**Streaming Response Implementation:**
- Files: `src/api/backend.py:313-336` (agent execution stream), `src/api/backend.py:387-401` (system prompt stream)
- Why fragile: Async generator pattern requires careful error handling. Stream can drop mid-chunk. Client-side stream parsing expects specific format (JSON lines with "data: " prefix).
- Safe modification: Add integration tests that consume full stream end-to-end. Add timeout to stream_generator. Log stream start/end. Test with slow/interrupted network.
- Test coverage: No tests for streaming endpoints. Manual testing only.

---

## Scaling Limits

**Database Connection Pooling:**
- Current capacity: SQLAlchemy default pool size (5 connections). Under high concurrency, connection exhaustion.
- Limit: ~20 concurrent requests before connection pool saturation. Each request takes 1 connection for entire lifetime.
- Files: `src/database/database.py:72-82` (engine creation, no pool configuration)
- Scaling path: Configure SQLAlchemy pool with larger size (20-50), add pool_recycle (3600s) for PostgreSQL. Use connection pooling service (PgBouncer for Postgres). Monitor pool usage with engine.pool.checkedout().

**Agent Execution Concurrency:**
- Current capacity: Async executor but tool execution is blocking subprocess. Conda environment execution serialized by OS.
- Limit: Can't execute >10 tools simultaneously without thread pool exhaustion or conda contention
- Files: `src/executors/flow_executor.py:273-278` (ThreadPoolExecutor for async bridges)
- Scaling path: Use asyncio.create_subprocess_exec() instead of subprocess.run(). Move tool execution to separate worker queue (Celery/RQ). Distribute conda env across multiple machines.

**Memory Usage of Execution Records:**
- Current capacity: All execution messages stored in database. No cleanup. Traces grow unbounded.
- Limit: After ~100k executions with avg 50 messages each (5M messages), database file/table size becomes unwieldy.
- Files: `src/database/database_setup.py:104-137` (Message table, no retention policy)
- Scaling path: Implement message archival (move old messages to cold storage). Add TTL on message records. Implement pagination for message retrieval.

---

## Dependencies at Risk

**Google ADK (google.adk):**
- Risk: Early-stage Google library. API may change. Limited documentation and community support. Vendor lock-in to Google models only.
- Impact: Agent execution could break on google.adk version update. PydanticAI is more mature alternative.
- Files: `src/factories/agent_factory.py` (Google ADK factory), `tests/test_google_adk.py`
- Migration plan: Prioritize PydanticAI factory as default. Document ADK as experimental. Plan migration to pure OpenAI SDK or LangChain for broader model support.

**Conda Dependency:**
- Risk: Tool execution requires conda CLI to be installed and in PATH. Many deployments may not have conda.
- Impact: Conda environment tools fail silently if conda not available. Tool executor falls back to non-isolated execution.
- Files: `src/executors/tool_executor.py:217` (conda run command)
- Migration plan: Detect conda at startup, warn if unavailable. Support Docker-based tool execution as alternative isolation. Document conda as optional optimization.

**LangChain/LangGraph:**
- Risk: Large, actively changing library. Agents module deprecated in favor of LangGraph. Breaking changes common between versions.
- Impact: Code in `src/agents.py` may become incompatible. Graph builder patterns may shift.
- Files: `src/agents.py` (imports from langchain.agents, langgraph)
- Migration plan: Reduce LangChain usage. Move to PydanticAI for agents. Use LangGraph only for graph structures if truly needed. Consider custom lightweight graph implementation.

---

## Missing Critical Features

**No Execution History Export:**
- Problem: Execution records stored in database but no way to export/audit. No CSV export, no execution report generation.
- Blocks: Compliance reporting, debugging tool behavior across runs, performance analysis

**No Tool/Agent Versioning:**
- Problem: Updating a tool updates it in-place. No way to roll back or maintain versions. Executions don't record which tool version was used.
- Blocks: Safe tool iteration, debugging regressions, A/B testing

**No Rollback on Flow Execution Failure:**
- Problem: If flow fails mid-execution, partial results are stored. No transactional semantics. No rollback capability.
- Blocks: Atomic workflow operations, data consistency guarantees

**No Agent Parameter Override:**
- Problem: Agent system prompt and tools baked into graph_config at creation. No runtime override. Can't change behavior per execution.
- Blocks: A/B testing prompts, dynamic tool selection, agent customization per user

---

## Test Coverage Gaps

**Graph Execution Logic:**
- What's not tested: Complex multi-node graphs, cyclic edges, multiple exit points, node failure handling
- Files: `src/executors/agent_executor.py:167-290` (_execute_graph)
- Risk: Changes to graph traversal could break without notice
- Priority: High

**Streaming Endpoints:**
- What's not tested: SSE stream generation, stream errors, incomplete streams, client disconnection
- Files: `src/api/backend.py:313-336`, `src/api/backend.py:387-401`
- Risk: Streaming failures not caught in CI
- Priority: High

**Authorization & Ownership Checks:**
- What's not tested: User can't access other users' resources, user_id validation, deletion prevents orphaned records
- Files: `src/database/database.py` (CRUD functions), `src/api/backend.py` (all endpoints)
- Risk: Security regression possible without tests
- Priority: Critical

**Tool Type Resolution Edge Cases:**
- What's not tested: Complex generic types (Dict[str, List[int]]), Optional nested types, custom classes in tool code
- Files: `src/executors/tool_executor.py:70-111` (eval_type_string)
- Risk: Type validation fails silently on complex schemas
- Priority: Medium

**Flow Validation (Tool Compatibility):**
- What's not tested: Connecting incompatible tool output to input, schema validation across edges, missing required inputs
- Files: `src/validate/tool_compatibility.py`
- Risk: Invalid flows created, execution fails at runtime
- Priority: Medium

---

*Concerns audit: 2026-03-15*
