# Generate Both System and User Prompts — Two-Pass Flow (IMPLEMENTED)

## Approach

The "Generate with AI" button now runs two sequential LLM calls:
1. **Pass 1:** Generate system prompt (existing flow, unchanged)
2. **Pass 2:** Generate user prompt, with the system prompt from pass 1 as context

The second call uses a separate prompt template (`agent_user_prompt_gen`) that instructs the LLM to write a complementary user prompt, aware of what the system prompt already covers.

## Changes Made

### Backend
- `src/prompts/agent_user_prompt_gen.system.md` — **new** — instructs LLM on how to write user prompt templates, explains `{input}` and `{message_history}` placeholders
- `src/prompts/agent_user_prompt_gen.user.md` — **new** — provides agent details + the generated system prompt via `{SYSTEM_PROMPT}` placeholder
- `src/ai_integrations/generate_agent_system_prompt.py` — updated `generate_user_prompt_stream` to accept `generated_system_prompt` param and use `agent_user_prompt_gen` template
- `src/api/backend.py` — added `UserPromptGenerateRequest` model with `generated_system_prompt` field, updated endpoint to use it

### Frontend
- `frontend/src/lib/api.ts` — added `UserPromptGenerateRequest` interface extending `SystemPromptGenerateRequest`
- `frontend/src/routes/+page.svelte` — updated `handleGeneratePrompt` to two-pass flow: system prompt streams first, then user prompt streams with awareness of the system prompt
