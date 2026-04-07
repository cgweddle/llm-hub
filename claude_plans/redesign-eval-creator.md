# Plan: Redesign Evaluation Creator

## Context

The current evaluation form is a basic prompt editor. The user wants a structured evaluator builder with:
- Configurable data type, categories with descriptions, return fields, and input variable selection
- A "Generate with AI" button that produces a judge prompt from the configured fields
- Parameters stored in the DB, judge function built at runtime (not stored as a Python string)

**Approach: Parameters in DB, runtime construction.** This is more secure (no code injection), allows evolving the execution engine without migration, and aligns with the existing architecture where `evaluation_executor.py` already builds prompts from stored parameters.

---

## Part 1: Expand Database Model

**File:** `src/database/database_setup.py` — Update `Evaluation` model

Add/modify columns:
- `score_categories` — change from simple `["good","bad"]` to structured `[{"name": "good", "description": "Output is helpful"}, ...]`
- `return_fields` — new JSON column, optional list of return field names beyond score (e.g. `["reasoning", "confidence"]`)
- `input_variables` — new JSON column, list of selected inputs: `["output", "input", "tool_output"]`

**Remove `llm_provider` column** — the model is no longer stored on the evaluation. Instead, the user selects the model at runtime when triggering the evaluation (in the Evals toggle flow or manually). The evaluation definition is model-agnostic.

No new tables needed. The existing `judge_system_prompt` and `scoring_rubric` columns remain — the AI-generated prompt goes into `judge_system_prompt`.

**File:** `src/database/database.py` — No changes needed (uses `**kwargs`)

**File:** `src/api/backend.py` — Update `EvaluationCreate`, `EvaluationUpdate`, `EvaluationResponse` Pydantic models:
- Add `return_fields` and `input_variables`
- Remove `llm_provider` from create/update/response
- Update `EvaluateRequest` to accept `llm_provider: str` (model selected at runtime per evaluation run)

---

## Part 2: Update Evaluation Executor

**File:** `src/executors/evaluation_executor.py`

### Fix existing bug
- Change `langfuse_client.score()` → `langfuse_client.create_score()` (current code crashes)

### Temperature
- Use the model's default temperature (no explicit temperature setting)

### Update `_build_user_message()`
- Only include sections for selected `input_variables`:
  - `"input"` → include `## Agent Input` section
  - `"output"` → include `## Agent Output` section
  - `"tool_output"` → include `## Tool Output` section (from execution metadata or child tool_result nodes)
- If no `input_variables` set, default to current behavior (input + output)

### Update structured output models
- If `return_fields` specified, dynamically build a Pydantic model that includes those fields alongside `score`
- If no `return_fields`, use existing `NumericJudgeResponse` / `BooleanJudgeResponse` / `CategoricalJudgeResponse`

### Update category validation
- Handle new structured categories format: `[{"name": "safe", "description": "..."}]`
- Extract just the names for validation, include descriptions in the prompt

---

## Part 3: Redesign EvaluationManager.svelte Form

**File:** `frontend/src/routes/EvaluationManager.svelte`

### New form layout

**Section 1: Basics**
- Name (text input)
- Description (text input, optional)
- Is Public (checkbox)

(No LLM provider field — model is selected at runtime when running the evaluation)

**Section 2: Data Type**
- Score Type: radio or select (Numeric / Categorical / Boolean)
- If **Numeric** or **Categorical**:
  - Dynamic list of categories, each with:
    - Category name (required)
    - Category description (optional)
  - Add/remove buttons for categories
  - For Numeric: categories represent score ranges (e.g. "0.0-0.3" = "Poor")
  - For Categorical: categories are the allowed values

**Section 3: Inputs**
- Checkboxes for which execution data to include in the judge prompt:
  - `output` — the agent's output
  - `input` — the user's input to the agent
  - `tool_output` — output from tool calls
- Selected inputs appear as insertable `{output}`, `{input}`, `{tool_output}` tags in the prompt

**Section 4: Return Fields**
- Optional list of additional return fields beyond score
- Each field has a name (e.g. "reasoning", "confidence")
- Add/remove buttons

**Section 5: Judge Prompt**
- Large textarea for `judge_system_prompt`
- Above the textarea: clickable variable tags (`{output}`, `{input}`, `{tool_output}`) that insert at cursor
- **"Generate with AI" button** — generates a prompt based on all configured fields

---

## Part 4: "Generate with AI" for Evaluations

### Backend

**File:** `src/api/backend.py` — New endpoint:
- `POST /evaluations/generate-prompt` — streaming endpoint

**File:** `src/ai_integrations/generate_eval_prompt.py` — New file

**Request model:**
```python
class EvalPromptGenerateRequest(BaseModel):
    eval_name: str
    eval_description: str
    score_type: str
    score_categories: Optional[List[dict]]  # [{"name": "safe", "description": "..."}]
    return_fields: Optional[List[str]]
    input_variables: List[str]  # ["output", "input", "tool_output"]
    model: str
    additional_instructions: Optional[str]
```

**Prompt generation logic:**
- Load a template from the `prompts` DB table (name: `"eval_prompt_gen"`)
- Or hardcode a meta-prompt that instructs the LLM to generate a judge prompt given:
  - The evaluation name and description
  - The score type and categories (with descriptions)
  - The available input variables
  - The expected return format
- Stream the generated prompt back

**Example meta-prompt:**
```
You are an expert at writing LLM-as-a-judge evaluation prompts.

Generate a system prompt for a judge that evaluates: {EVAL_NAME}
Description: {EVAL_DESCRIPTION}

Score type: {SCORE_TYPE}
{CATEGORIES_SECTION}

The judge will receive these inputs: {INPUT_VARIABLES}
{RETURN_FIELDS_SECTION}

Write a clear, specific system prompt that:
- Explains what to evaluate
- Provides the scoring criteria
- Specifies the exact JSON return format with {RETURN_FORMAT}
```

### Frontend

**File:** `frontend/src/routes/EvaluationManager.svelte`
- Add "Generate with AI" button next to the judge prompt textarea
- Uses the same NDJSON streaming pattern as `generateSystemPromptStream`
- Streams generated text into the `judge_system_prompt` textarea

**File:** `frontend/src/lib/api.ts`
- Add `generateEvalPromptStream()` function following the existing streaming pattern

---

## Part 5: Update Frontend API Types

**File:** `frontend/src/lib/api.ts`

Update `Evaluation` interface:
```typescript
interface Evaluation {
  // existing fields...
  score_categories: Array<{name: string; description?: string}> | null;  // changed from string[]
  return_fields: string[] | null;  // new
  input_variables: string[] | null;  // new
}
```

---

## Key Files

| File | Changes |
|------|---------|
| `src/database/database_setup.py` | Add `return_fields`, `input_variables` columns |
| `src/api/backend.py` | Update Pydantic models, add generate-prompt endpoint |
| `src/executors/evaluation_executor.py` | Fix `create_score` bug, update prompt building for new fields |
| `src/ai_integrations/generate_eval_prompt.py` | **New** — eval prompt generation with streaming |
| `frontend/src/routes/EvaluationManager.svelte` | Redesigned form with categories, inputs, return fields, AI generation |
| `frontend/src/lib/api.ts` | Update Evaluation interface, add `generateEvalPromptStream` |

---

## Verification

1. **Create evaluation**: Fill in name, select Categorical, add categories with descriptions, select inputs, click "Generate with AI" → verify prompt is generated with correct format
2. **Save and run**: Save evaluation, assign to agent, run flow with Evals → verify executor builds correct prompt from new fields
3. **LangFuse**: Verify `create_score` posts correctly (bug fix)
4. **Backward compat**: Existing evaluations with simple `score_categories` strings still work
