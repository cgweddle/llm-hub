# Plan: LLM-as-a-Judge Evaluations with LangFuse Backend

## Context

LLM Hub already captures agent execution traces in LangFuse. Users need a way to **evaluate the quality** of agent outputs by defining custom evaluation types (e.g., helpfulness, factual accuracy) and running a "judge" LLM that scores executions. **LangFuse is the single source of truth for evaluation results** — our DB only tracks that an evaluation run happened and where to find it in LangFuse, avoiding data duplication.

v1 scope: on-demand evaluation of individual executions. Batch evaluation can come later.

---

## Phase 1: Database Models

**File:** `src/database/database_setup.py`

Add two new models after `Prompts` (line 150):

### `Evaluation` table — stores reusable evaluation type definitions
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| user_id | FK → users.id | not null |
| name | String(100) | not null, e.g. "Helpfulness" |
| description | Text | human-readable description |
| judge_system_prompt | Text | not null — system prompt for the judge LLM |
| scoring_rubric | Text | rubric text included in the judge's user message |
| score_type | String(20) | `NUMERIC`, `CATEGORICAL`, or `BOOLEAN` |
| score_categories | JSON | for CATEGORICAL, e.g. `["good","bad","neutral"]` |
| llm_provider | String(100) | not null — name from `~/.llm_hub/config.yaml` |
| is_public | Boolean | default False |
| created_at | DateTime | default now |
| updated_at | DateTime | auto-updated |

Relationships: `user`, `results` (→ EvaluationResult)

### `EvaluationResult` table — lightweight pointer to LangFuse
| Column | Type | Notes |
|---|---|---|
| id | Integer PK | |
| evaluation_id | FK → evaluations.id | not null |
| execution_id | FK → executions.id | not null |
| user_id | FK → users.id | not null |
| langfuse_trace_id | String(200) | the execution's trace ID (for convenience) |
| langfuse_score_id | String(200) | ID from `langfuse_client.score()` — primary reference to result in LangFuse |
| status | String(20) | `running` / `completed` / `failed` |
| error_message | Text | only populated on failure |
| created_at | DateTime | default now |
| completed_at | DateTime | |

Relationships: `evaluation`, `execution`, `user`

**No score values, reasoning, or model info stored here** — all of that lives in LangFuse. The `langfuse_score_id` is the foreign key into LangFuse's scoring system.

**File:** `src/database/database.py` — Add CRUD helpers:
- `create_evaluation()`, `get_evaluations_by_user()`, `get_evaluation_by_id()`, `update_evaluation()`, `delete_evaluation()`
- `create_evaluation_result()`, `get_evaluation_results_by_execution()`, `update_evaluation_result()`

---

## Phase 2: Evaluation Executor

**New file:** `src/executors/evaluation_executor.py`

### Design
- `EvaluationExecutor` class with `async evaluate()` method
- Reuses `_resolve_model_name()` — extract from `AgentExecutor` (line 666–701) into a shared utility
- Uses PydanticAI `Agent` with **structured output** for typed judge responses

### Judge prompt construction
- **System prompt**: `evaluation.judge_system_prompt`
- **User message**: assembled from execution input/output + rubric:
  ```
  ## Agent Input
  {execution.input_data}

  ## Agent Output
  {execution.output_data}

  ## Scoring Rubric
  {evaluation.scoring_rubric}

  Evaluate the agent's output and provide your score.
  ```

### Structured output models (per score_type)
```python
class NumericJudgeResponse(BaseModel):
    reasoning: str
    score: float = Field(ge=0, le=1)

class BooleanJudgeResponse(BaseModel):
    reasoning: str
    score: bool

class CategoricalJudgeResponse(BaseModel):
    reasoning: str
    score: str  # validated against evaluation.score_categories
```

### LangFuse score posting
```python
score = langfuse_client.score(
    trace_id=execution.langfuse_trace_id,
    name=evaluation.name,
    value=numeric_value,
    comment=judge_response.reasoning,
    data_type=evaluation.score_type,
)
langfuse_client.flush()
# Store score.id as langfuse_score_id on EvaluationResult
```

---

## Phase 3: Backend API Endpoints

**File:** `src/api/backend.py`

### Pydantic request/response models
- `EvaluationCreate`, `EvaluationUpdate`, `EvaluationResponse`
- `EvaluateRequest` (`user_id`, `evaluation_ids: List[int]`)
- `EvaluationResultResponse` (includes `langfuse_score_id` and `status`)

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/evaluations/?user_id={id}` | Create evaluation type |
| GET | `/evaluations/?user_id={id}` | List user's evaluations (+ public ones) |
| GET | `/evaluations/{id}` | Get single evaluation |
| PATCH | `/evaluations/{id}` | Update evaluation |
| DELETE | `/evaluations/{id}` | Delete evaluation |
| POST | `/executions/{execution_id}/evaluate` | Run evaluation(s) against an execution |
| GET | `/executions/{execution_id}/evaluations` | Get evaluation results for an execution |
| GET | `/executions/{execution_id}/scores` | Proxy — fetch score data from LangFuse |

The `/evaluate` endpoint:
1. Validates execution exists and has a `langfuse_trace_id`
2. Creates `EvaluationResult` rows with `status='running'`
3. For each evaluation_id, calls `EvaluationExecutor.evaluate()`
4. Posts score to LangFuse, captures `langfuse_score_id`
5. Updates `EvaluationResult` with `langfuse_score_id`, `status='completed'`

The `/scores` endpoint:
- Proxies to LangFuse to fetch score details for the execution's trace
- Returns score values, comments (reasoning), and metadata from LangFuse

---

## Phase 4: Frontend — API Client & Types

**File:** `frontend/src/lib/api.ts`

### New interfaces
```typescript
interface Evaluation {
  id: number;
  name: string;
  description: string | null;
  judge_system_prompt: string;
  scoring_rubric: string | null;
  score_type: 'NUMERIC' | 'CATEGORICAL' | 'BOOLEAN';
  score_categories: string[] | null;
  llm_provider: string;
  is_public: boolean;
  created_at: string;
  updated_at: string;
}

interface EvaluationResult {
  id: number;
  evaluation_id: number;
  execution_id: number;
  langfuse_score_id: string | null;
  status: string;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

interface LangFuseScore {
  id: string;
  name: string;
  value: number | string | boolean;
  comment: string | null;  // judge reasoning
  data_type: string;
  created_at: string;
}
```

### New API functions
- `fetchEvaluations(userId)`, `createEvaluation(userId, data)`, `updateEvaluation(id, data)`, `deleteEvaluation(id)`
- `evaluateExecution(executionId, userId, evaluationIds)`, `fetchEvaluationResults(executionId)`
- `fetchExecutionScores(executionId)` — fetches score data from LangFuse via proxy

---

## Phase 5: Frontend — Evaluation Manager

**New file:** `frontend/src/routes/EvaluationManager.svelte`

Dialog component (shadcn-svelte) for CRUD of evaluation types:
- List of existing evaluations with edit/delete
- Create/edit form: name, description, judge system prompt (textarea), scoring rubric (textarea), score type (Select), categories (conditional Input), LLM provider (Select from `llmProviders`), is_public toggle
- Follows pattern of `LLMProvidersPanel.svelte` and `CondaEnvironmentsPanel.svelte`

**File:** `frontend/src/routes/+page.svelte`
- Add "Evaluations" button to toolbar
- Import and render `EvaluationManager`

---

## Phase 6: Frontend — Evaluation in InfoPanel

**File:** `frontend/src/routes/InfoPanel.svelte`

### New state
```typescript
let scoresCache: Map<number, LangFuseScore[]> = $state(new Map());
let scoresLoading: Set<number> = $state(new Set());
let evaluatingSet: Set<number> = $state(new Set());
```

### UI additions
1. **"Evaluate" button** on agent execution nodes with `langfuse_trace_id`
2. **Evaluation selector** — list of available evaluations with checkboxes + "Run" button
3. **Score display** in expanded node details (below trace section), fetched from LangFuse:
   ```
   Evaluation Scores
     Helpfulness: 0.85 ▸ (expand for reasoning)
     Factual Accuracy: true ▸
   ```
4. **Auto-load** existing scores when expanding a traced node via `fetchExecutionScores()`
5. **Loading state** while evaluation is running

---

## Key Files Summary

| File | Action |
|------|--------|
| `src/database/database_setup.py` | Add `Evaluation` and `EvaluationResult` models |
| `src/database/database.py` | Add CRUD helpers (~8 functions) |
| `src/api/backend.py` | Add Pydantic models + 8 endpoints |
| `src/executors/evaluation_executor.py` | **New** — judge LLM orchestration + LangFuse score posting |
| `src/utils/llm_config.py` | **New** — extracted `resolve_model_name()` utility |
| `src/executors/agent_executor.py` | Refactor to use shared `resolve_model_name()` |
| `frontend/src/lib/api.ts` | Add interfaces + 7 API functions |
| `frontend/src/routes/EvaluationManager.svelte` | **New** — evaluation type CRUD dialog |
| `frontend/src/routes/InfoPanel.svelte` | Add evaluate button + score display (from LangFuse) |
| `frontend/src/routes/+page.svelte` | Add toolbar button + import |

---

## Verification

1. **Database**: `python src/database/database_setup.py setup` — verify new tables
2. **Evaluation CRUD**: Create/list/update/delete evaluations via curl
3. **Evaluation flow**: Run agent, POST to `/executions/{id}/evaluate` — verify `EvaluationResult` created with `langfuse_score_id`, score visible in LangFuse dashboard, `GET /executions/{id}/scores` returns data from LangFuse
4. **Frontend manager**: Create evaluation type, verify persistence
5. **InfoPanel**: Expand agent node, click Evaluate, select evaluations, verify scores render from LangFuse
6. **Edge cases**: No trace ID (button disabled), LangFuse down (error shown), is_public filtering
