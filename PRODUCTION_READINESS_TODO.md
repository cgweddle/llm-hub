# LLM Hub Production Readiness TODO

**Generated:** 2025-11-23
**Status:** Architecture Gap Analysis & Implementation Roadmap

---

## 🚨 CRITICAL PRIORITIES (Must-Have for Production)

### 1. Code Execution Sandboxing (HIGHEST PRIORITY)
**Current Risk:** Python scripts execute with full system permissions
**Location:** `src/python_script_agent.py:173`

**Tasks:**
- [ ] Implement Docker container execution for Python scripts
- [ ] Add resource limits (CPU: 1 core, Memory: 512MB, Disk: 1GB)
- [ ] Configure network isolation (no external network access)
- [ ] Mount filesystem as read-only except /tmp
- [ ] Implement output size limits (max 10MB)
- [ ] Add execution timeout per tool (configurable, default 5min)
- [ ] Create security policy documentation

**Implementation Notes:**
```python
# Replace subprocess.run with Docker execution
import docker
client = docker.from_env()
container = client.containers.run(
    image="python:3.11-slim",
    command=["python", "/script/tool.py"],
    volumes={script_dir: {'bind': '/script', 'mode': 'ro'}},
    mem_limit="512m",
    cpu_quota=100000,
    network_disabled=True,
    remove=True,
    timeout=300
)
```

---

### 2. Tool Versioning System
**Current Risk:** Breaking changes cascade to all dependent agents

**Database Changes:**
- [ ] Add `version` column to Tool model (semantic versioning: "1.2.3")
- [ ] Add `deprecated` boolean column
- [ ] Add `deprecation_notice` text column
- [ ] Add `min_compatible_version` column
- [ ] Add `changelog` JSON column
- [ ] Create `ToolVersion` table for version history
- [ ] Add migration script for existing tools (default to "1.0.0")

**API Changes:**
- [ ] Update `create_tool` to accept version parameter
- [ ] Add `update_tool_version` endpoint (creates new version)
- [ ] Add `deprecate_tool` endpoint
- [ ] Add `get_tool_versions` endpoint (list all versions)
- [ ] Add version parameter to tool queries (default: latest)

**Validation:**
- [ ] Update tool compatibility validator to check version compatibility
- [ ] Add version conflict detection in workflows
- [ ] Create tool migration guide generator

---

### 3. Error Handling & Retry Logic
**Current Risk:** Transient failures cause permanent job loss
**Location:** Silent failures in `src/python_script_agent.py:184`

**Tasks:**
- [ ] Remove all `except: pass` statements (audit entire codebase)
- [ ] Install `tenacity` library for retry logic
- [ ] Create error classification system (Transient, Permanent, UserError)
- [ ] Implement retry decorator with exponential backoff
- [ ] Add circuit breaker pattern (fail fast after N consecutive failures)
- [ ] Create error recovery strategies per tool type
- [ ] Add dead letter queue for permanently failed executions

**Implementation:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

class TransientError(Exception):
    """Errors that should be retried"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(TransientError),
    before_sleep=log_retry_attempt
)
def execute_tool(tool_id, inputs):
    # Execution logic
    pass
```

**Error Categories:**
- [ ] Define TransientError (network, timeout, rate limit)
- [ ] Define PermanentError (invalid input, missing dependency)
- [ ] Define UserError (schema validation failure)
- [ ] Add error context (tool_id, execution_id, timestamp)

---

### 4. Background Job System
**Current Risk:** Long-running tasks block API, jobs lost on restart

**Tasks:**
- [ ] Install Celery + Redis (or RQ as lightweight alternative)
- [ ] Create Celery configuration (`celeryconfig.py`)
- [ ] Convert tool execution to async tasks
- [ ] Add job persistence (Redis or database backend)
- [ ] Implement progress tracking callbacks
- [ ] Add job cancellation support
- [ ] Create job status endpoint (`GET /executions/{id}/status`)
- [ ] Add job result retrieval endpoint
- [ ] Implement job timeout handling
- [ ] Add worker monitoring dashboard

**Celery Setup:**
```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    'llm_hub',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

@celery_app.task(bind=True)
def execute_tool_async(self, tool_id, inputs, execution_id):
    self.update_state(state='PROGRESS', meta={'percent': 0})

    # Execute tool
    result = execute_tool(tool_id, inputs)

    self.update_state(state='PROGRESS', meta={'percent': 100})
    return result
```

**API Integration:**
- [ ] Add `/tools/{id}/execute_async` endpoint
- [ ] Return job_id immediately
- [ ] Add polling endpoint for job status
- [ ] Add webhook support for completion notifications

---

### 5. Workflow State Checkpointing
**Current Risk:** Workflow failures lose all progress

**Database Changes:**
- [ ] Create `ExecutionCheckpoint` table
  - execution_id (FK to Execution)
  - step_number (int)
  - node_name (str)
  - graph_state (JSON - full state snapshot)
  - timestamp (datetime)
  - is_latest (boolean)
- [ ] Add `resume_from_checkpoint_id` to Execution table
- [ ] Add `checkpoint_config` JSON column to Flow table

**Implementation:**
- [ ] Add checkpoint saving after each node execution
- [ ] Implement checkpoint restoration logic
- [ ] Add resume workflow endpoint (`POST /executions/{id}/resume`)
- [ ] Add state compression for large states (gzip)
- [ ] Implement checkpoint cleanup (keep last N checkpoints)
- [ ] Add checkpoint validation (state schema validation)

**Graph Execution Update:**
```python
def execute_node_with_checkpoint(node, state, execution_id, step_number):
    # Execute node
    result = node.run(state)

    # Save checkpoint
    checkpoint = ExecutionCheckpoint(
        execution_id=execution_id,
        step_number=step_number,
        node_name=node.name,
        graph_state=compress_state(state),
        is_latest=True
    )
    session.add(checkpoint)
    session.commit()

    return result
```

---

### 6. Monitoring & Observability
**Current Risk:** Blind to failures, no performance visibility

**Metrics (Prometheus):**
- [ ] Install `prometheus-client` library
- [ ] Add metrics endpoint (`GET /metrics`)
- [ ] Implement metrics:
  - `tool_executions_total` (counter) - labels: tool_name, status
  - `tool_execution_duration_seconds` (histogram) - labels: tool_name
  - `workflow_executions_total` (counter) - labels: flow_name, status
  - `workflow_execution_duration_seconds` (histogram) - labels: flow_name
  - `active_executions` (gauge)
  - `api_requests_total` (counter) - labels: endpoint, method, status
  - `api_request_duration_seconds` (histogram)

**Tracing (OpenTelemetry):**
- [ ] Install `opentelemetry-api` and `opentelemetry-sdk`
- [ ] Configure tracer provider
- [ ] Add spans for:
  - Tool execution
  - Workflow execution
  - Database queries
  - External API calls
- [ ] Add trace context propagation
- [ ] Implement correlation ID generation

**Logging:**
- [ ] Convert to structured JSON logging
- [ ] Add correlation IDs to all log entries
- [ ] Add log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- [ ] Configure log aggregation (ELK stack or Loki)

**Health Checks:**
- [ ] Create `/health` endpoint (basic liveness)
- [ ] Create `/health/ready` endpoint (readiness check)
  - Check database connectivity
  - Check Redis connectivity
  - Check Celery worker availability
- [ ] Add startup probe
- [ ] Add liveness probe

**Error Tracking:**
- [ ] Install Sentry SDK
- [ ] Configure Sentry integration
- [ ] Add error context (user_id, execution_id, tool_id)
- [ ] Add breadcrumbs for debugging
- [ ] Set up alerting rules

---

## 🔧 IMPORTANT (Enhances Functionality)

### 7. Advanced Workflow Features

**Loop Support:**
- [ ] Add `LoopNode` type to graph
- [ ] Implement iteration logic (for, while, do-while)
- [ ] Add loop condition evaluation
- [ ] Add max iteration limit (prevent infinite loops)
- [ ] Add loop state accumulation

**Parallel Execution:**
- [ ] Implement true parallel node execution (current: sequential)
- [ ] Add `ParallelNode` type
- [ ] Add barrier synchronization (wait for all parallel branches)
- [ ] Add partial failure handling (continue if N/M succeed)

**Error Handlers:**
- [ ] Add per-node error handler configuration
- [ ] Implement `on_error` callbacks
- [ ] Add retry configuration per node
- [ ] Add fallback node specification

**Conditional Branching:**
- [ ] Enhance conditional edge evaluation
- [ ] Add expression language (e.g., JSONPath, JMESPath)
- [ ] Add switch/case node type
- [ ] Add dynamic routing based on state

**Variables & Context:**
- [ ] Add workflow-level variables
- [ ] Implement variable scoping (global, node-local)
- [ ] Add variable interpolation in node configs
- [ ] Add secret variable support (masked in logs)

---

### 8. Tool Discovery & Cataloging

**Search & Filtering:**
- [ ] Implement full-text search on tool name/description
- [ ] Add filtering by tool type, category, tags
- [ ] Add filtering by input/output schema
- [ ] Add capability-based search ("tools that read CSV")

**Categorization:**
- [ ] Add `tags` JSON array to Tool model
- [ ] Add `category` column (data-processing, ml, api, etc.)
- [ ] Add `capabilities` JSON array (reads-csv, writes-json, etc.)
- [ ] Create category taxonomy
- [ ] Add tag management endpoints

**Recommendations:**
- [ ] Build tool recommendation engine
- [ ] Suggest compatible tools based on output→input matching
- [ ] Show "frequently used together" tools
- [ ] Add tool popularity scoring

**Analytics:**
- [ ] Track tool usage counts
- [ ] Track tool execution success rates
- [ ] Add tool rating system
- [ ] Show tool performance metrics (avg execution time)

**OpenAPI Schema:**
- [ ] Auto-generate OpenAPI schema from input/output schemas
- [ ] Add schema validation
- [ ] Generate interactive docs (Swagger UI)
- [ ] Add example requests/responses

---

### 9. Testing Infrastructure

**Test Fixtures:**
- [ ] Create mock tool factory
- [ ] Create mock agent factory
- [ ] Create test database fixtures
- [ ] Add sample workflows for testing

**Integration Tests:**
- [ ] Test complete tool creation flow
- [ ] Test workflow execution end-to-end
- [ ] Test tool chaining
- [ ] Test error propagation
- [ ] Test checkpointing and resume

**Contract Tests:**
- [ ] Validate tool schemas against executions
- [ ] Test input schema enforcement
- [ ] Test output schema validation
- [ ] Test type compatibility

**Load Tests:**
- [ ] Create k6 or Locust test scripts
- [ ] Test concurrent tool executions
- [ ] Test workflow throughput
- [ ] Identify performance bottlenecks
- [ ] Set performance benchmarks

**Test Coverage:**
- [ ] Install `pytest-cov`
- [ ] Set coverage target (80%+)
- [ ] Add coverage reports to CI
- [ ] Add coverage badges to README

---

### 10. API Documentation & Developer Experience

**OpenAPI Generation:**
- [ ] Auto-generate OpenAPI 3.0 spec from FastAPI
- [ ] Customize Swagger UI branding
- [ ] Add API authentication documentation
- [ ] Add rate limiting documentation

**SDK Generation:**
- [ ] Generate Python SDK from OpenAPI spec
- [ ] Generate JavaScript/TypeScript SDK
- [ ] Publish SDKs to package registries (PyPI, npm)
- [ ] Add SDK usage examples

**Interactive Playground:**
- [ ] Build tool testing playground UI
- [ ] Add workflow visual builder
- [ ] Add live execution logs
- [ ] Add schema validator UI

**Documentation Site:**
- [ ] Set up MkDocs or Docusaurus
- [ ] Write getting started guide
- [ ] Add tool creation tutorial
- [ ] Add workflow building guide
- [ ] Add architecture documentation
- [ ] Add API reference
- [ ] Add troubleshooting guide

**Examples:**
- [ ] Create example tool library
- [ ] Add workflow templates
- [ ] Add quickstart projects
- [ ] Add video tutorials

---

## 📅 IMPLEMENTATION ROADMAP

### **Phase 1: Security & Stability** (Weeks 1-2)
**Goal:** Make system safe and reliable

**Sprint 1 (Week 1):**
- [ ] Day 1-2: Implement Docker sandboxing for code execution
- [ ] Day 3-4: Add retry logic with exponential backoff
- [ ] Day 5: Remove silent error swallowing (audit & fix)

**Sprint 2 (Week 2):**
- [ ] Day 1-2: Add input validation for tool parameters
- [ ] Day 3-4: Implement error classification system
- [ ] Day 5: Add circuit breaker pattern

**Deliverables:**
- ✅ Sandboxed execution environment
- ✅ Robust error handling
- ✅ Retry mechanisms

---

### **Phase 2: Execution & Jobs** (Weeks 3-4)
**Goal:** Support long-running tasks and recovery

**Sprint 3 (Week 3):**
- [ ] Day 1-2: Set up Celery + Redis
- [ ] Day 3-4: Convert tool execution to async tasks
- [ ] Day 5: Add job status endpoints

**Sprint 4 (Week 4):**
- [ ] Day 1-3: Implement checkpointing system
- [ ] Day 4: Add workflow resume capability
- [ ] Day 5: Add cancellation support

**Deliverables:**
- ✅ Background job system
- ✅ Checkpointing & resume
- ✅ Job monitoring

---

### **Phase 3: Versioning & Discovery** (Weeks 5-6)
**Goal:** Manage tool evolution and improve discoverability

**Sprint 5 (Week 5):**
- [ ] Day 1-2: Add tool versioning schema
- [ ] Day 3-4: Implement version migration
- [ ] Day 5: Add deprecation workflow

**Sprint 6 (Week 6):**
- [ ] Day 1-2: Add tool search and filtering
- [ ] Day 3-4: Implement categorization system
- [ ] Day 5: Build compatibility matrix

**Deliverables:**
- ✅ Tool versioning
- ✅ Search & discovery
- ✅ Version compatibility

---

### **Phase 4: Observability** (Weeks 7-8)
**Goal:** Gain visibility into system behavior

**Sprint 7 (Week 7):**
- [ ] Day 1-2: Add Prometheus metrics
- [ ] Day 3-4: Implement health checks
- [ ] Day 5: Set up Grafana dashboards

**Sprint 8 (Week 8):**
- [ ] Day 1-2: Implement OpenTelemetry tracing
- [ ] Day 3-4: Integrate Sentry error tracking
- [ ] Day 5: Configure alerting rules

**Deliverables:**
- ✅ Metrics & dashboards
- ✅ Distributed tracing
- ✅ Error tracking & alerting

---

### **Phase 5: Testing & Documentation** (Weeks 9-10)
**Goal:** Improve quality and developer experience

**Sprint 9 (Week 9):**
- [ ] Day 1-2: Build test fixtures & factories
- [ ] Day 3-4: Write integration tests
- [ ] Day 5: Add load tests

**Sprint 10 (Week 10):**
- [ ] Day 1-2: Generate OpenAPI specs & SDK
- [ ] Day 3-4: Build documentation site
- [ ] Day 5: Create example workflows

**Deliverables:**
- ✅ Test coverage >80%
- ✅ API documentation
- ✅ Developer guides

---

## 🎯 QUICK WINS (Implement Today)

### 1. Health Check Endpoint (30 minutes)
```python
# Add to src/api/backend.py
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
```

### 2. Structured Logging (1 hour)
```python
# Add to logging setup
import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        return json.dumps(log_data)

handler.setFormatter(JsonFormatter())
```

### 3. Request Correlation IDs (1 hour)
```python
# Add middleware to FastAPI
from uuid import uuid4
from contextvars import ContextVar

correlation_id_var = ContextVar("correlation_id", default=None)

@app.middleware("http")
async def add_correlation_id(request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid4()))
    correlation_id_var.set(correlation_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response
```

### 4. Basic Retry Decorator (2 hours)
```python
# Install: pip install tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def execute_tool_with_retry(tool_id, inputs):
    return execute_tool(tool_id, inputs)
```

### 5. Tool Versioning Fields (2 hours)
```python
# Add to database_setup.py Tool model
version = Column(String(20), default="1.0.0")
deprecated = Column(Boolean, default=False)
deprecation_notice = Column(Text, nullable=True)

# Run migration
alembic revision --autogenerate -m "Add tool versioning"
alembic upgrade head
```

---

## 📊 SUCCESS METRICS

Track these metrics to measure production readiness:

**Security:**
- [ ] 100% of code execution sandboxed
- [ ] Zero privilege escalation vulnerabilities
- [ ] All inputs validated against schemas

**Reliability:**
- [ ] 99.9% uptime SLA
- [ ] <0.1% permanent job failure rate
- [ ] <5 second p95 API response time

**Observability:**
- [ ] 100% of endpoints instrumented with metrics
- [ ] <1 minute mean time to detect (MTTD) failures
- [ ] <5 minute mean time to recovery (MTTR)

**Quality:**
- [ ] >80% test coverage
- [ ] Zero critical security vulnerabilities (Snyk/Dependabot)
- [ ] <10% flaky test rate

**Developer Experience:**
- [ ] API documentation completeness: 100%
- [ ] <30 minute onboarding time
- [ ] <5 minute time to first successful tool execution

---

## 🔗 DEPENDENCIES & TOOLS

**New Dependencies to Add:**
```bash
# requirements.txt additions
tenacity==8.2.3           # Retry logic
celery==5.3.4             # Background jobs
redis==5.0.1              # Job queue backend
prometheus-client==0.19.0 # Metrics
opentelemetry-api==1.21.0 # Tracing
opentelemetry-sdk==1.21.0
sentry-sdk==1.39.1        # Error tracking
docker==7.0.0             # Container execution
```

**Infrastructure:**
- Redis (job queue + cache)
- Prometheus (metrics)
- Grafana (dashboards)
- Sentry (error tracking)
- ELK Stack or Loki (log aggregation)

---

## 📝 NOTES

**Architecture Decisions:**
- Use Docker for sandboxing (vs. restrictive subprocess)
- Use Celery over RQ (more features, better at scale)
- Use Prometheus over StatsD (better for time-series)
- Use OpenTelemetry (vendor-neutral tracing)

**Migration Strategy:**
- All changes should be backward compatible
- Use feature flags for gradual rollout
- Maintain old API endpoints during transition
- Run both sync and async execution in parallel initially

**Security Considerations:**
- All tool code treated as untrusted
- Principle of least privilege for execution
- Rate limiting per user/tool
- Audit logging for all mutations

---

**Last Updated:** 2025-11-23
**Next Review:** After Phase 1 completion
