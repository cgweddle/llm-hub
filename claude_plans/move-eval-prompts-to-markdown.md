# Plan: Move Evaluation Prompt Generation Templates to Markdown Files

## Context

The evaluation prompt generator (`src/ai_integrations/generate_eval_prompt.py`) hardcodes its meta system prompt and builds its user message entirely in Python code. All other prompt generators (agent system prompt, agent user prompt, code gen) follow a consistent pattern: templates live as `{name}.system.md` / `{name}.user.md` in `src/prompts/`, get uploaded to the DB via `upload_prompts.py`, and are loaded at runtime via `get_prompt_by_name()`. This change aligns the evaluation prompt generator with that pattern.

---

## Files to Modify

| File | Action |
|------|--------|
| `src/prompts/eval_prompt_gen.system.md` | **New** — system prompt (currently `META_SYSTEM_PROMPT` in Python) |
| `src/prompts/eval_prompt_gen.user.md` | **New** — user prompt template with placeholders |
| `src/ai_integrations/generate_eval_prompt.py` | Refactor to load prompts from DB instead of hardcoding |

No changes needed to `upload_prompts.py` — it already auto-discovers `*.system.md` / `*.user.md` files.

---

## Step 1: Create `src/prompts/eval_prompt_gen.system.md`

Move the `META_SYSTEM_PROMPT` content from `generate_eval_prompt.py:24-35` into this file verbatim:

```
You are an expert at writing LLM-as-a-judge evaluation prompts.
...
Output ONLY the system prompt text. Do not wrap it in quotes or markdown.
```

## Step 2: Create `src/prompts/eval_prompt_gen.user.md`

Extract the user message construction from `generate_eval_prompt.py:63-116` into a template with placeholders. The template will use these placeholders (filled at runtime):

- `{EVAL_NAME}` — evaluation name
- `{EVAL_DESCRIPTION_SECTION}` — description line (empty string if no description)
- `{SCORE_TYPE_SECTION}` — score type + constraints (NUMERIC/CATEGORICAL/BOOLEAN details)
- `{INPUT_VARIABLES_SECTION}` — what the judge will receive
- `{RETURN_FORMAT}` — JSON return format spec

Template content:
```
Generate a judge system prompt for an evaluation called "{EVAL_NAME}".
{EVAL_DESCRIPTION_SECTION}
{SCORE_TYPE_SECTION}
{INPUT_VARIABLES_SECTION}
{RETURN_FORMAT}
```

## Step 3: Refactor `generate_eval_prompt.py`

1. Remove the hardcoded `META_SYSTEM_PROMPT` constant
2. Add `from src.database.database import get_prompt_by_name` import
3. Accept a `session` parameter (matching `generate_agent_system_prompt.py` pattern)
4. Load prompts from DB: `get_prompt_by_name(session, "eval_prompt_gen")`
5. Keep the existing logic that builds score type sections, input variable descriptions, and return format — but use it to fill template placeholders instead of constructing the entire message
6. Add error handling for missing DB prompt (with `upload_prompts.py` hint)

## Step 4: Update the backend endpoint

**File:** `src/api/backend.py` — the `generate_eval_prompt_endpoint` (line 1157) needs to pass a DB session to the updated function signature.

## Step 5: Upload prompts & verify

```bash
python src/prompts/upload_prompts.py --verbose
```

Verify `eval_prompt_gen` appears in the prompts table, then test the `/evaluations/generate-prompt` endpoint still streams correctly.

---

## Verification

1. `python src/prompts/upload_prompts.py --verbose` — confirm `eval_prompt_gen` is uploaded
2. `POST /evaluations/generate-prompt` with test payload — confirm streaming works identically
3. Verify no hardcoded prompt strings remain in `generate_eval_prompt.py`
