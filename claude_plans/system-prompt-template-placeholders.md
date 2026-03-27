# Plan: Template Placeholders in Agent System Prompts

## Context

Currently, when a user generates a system prompt via AI, the LLM bakes the agent's name, description, and tool list directly into the text. This means the system prompt becomes stale if the agent is renamed, tools are added/removed, or the description changes.

This plan makes the generated system prompt keep `{AGENT_NAME}`, `{AGENT_DESCRIPTION}`, and `{TOOLS_SECTION}` as literal placeholders, which get resolved at runtime when the agent is compiled into a PydanticAI or ReAct agent.

## Changes

### 1. Create prompt template resolver utility
**New file: `src/utils/prompt_template.py`**

A pure function `resolve_system_prompt_template(system_prompt, node_config, tool_records)` that:
- Replaces `{AGENT_NAME}` → `node_config["name"]`
- Replaces `{AGENT_DESCRIPTION}` → `node_config["description"]`
- Replaces `{TOOLS_SECTION}` → formatted tool names list from `tool_records`
- Is a no-op for old prompts without placeholders (backward compatible)

Uses the same formatting as `generate_agent_system_prompt.py` lines 83-85 for the tools section.

### 2. Integrate resolver into agent execution (3 call sites)
**File: `src/executors/agent_executor.py`**

- `_run_pydanticai_node()` (line 496): Fetch tool records, resolve template before passing to `Agent()`
- `_execute_single_node_stream()` (line 629): Same pattern
- `_run_react_node()` (line 580): Resolve before passing as `agent_description` to ReActAgent

At each site, tool records are already being fetched for registration — just reorder to fetch before Agent construction.

### 3. Integrate resolver into PydanticAI factory
**File: `src/factories/pydanticai_agent_factory.py`**

- `create_from_node_config()` (line 63): Resolve template before `Agent(system_prompt=...)`
- Tool records are already being fetched in `_register_tools_by_ids()` — collect them first.

### 4. Add `description` field to node_config
**File: `frontend/src/lib/agentGraphBuilder.ts`**

- Add `description: string` to `SubAgentConfig` interface
- In `buildAgentGraph()` line 56: add `description: node.data.description || ''`

### 5. Include `description` in node_config when saving agents

**File: `frontend/src/routes/+page.svelte`** (simple agent creation, ~line 1080):
- Add `description: newAgentDescription.trim()` to `graph_config.nodes.main`

**File: `frontend/src/routes/FullscreenNodeModal.svelte`** (agent editing, ~line 99):
- Add `description: editedAgentDescription` to the node config object in `updatedGraphConfig`

### 6. Update meta-prompt to preserve placeholders
**File: `src/prompts/agent_prompt_gen.system.md`**

Change the instruction from "Do NOT include placeholders" to explicitly require `{AGENT_NAME}`, `{AGENT_DESCRIPTION}`, and `{TOOLS_SECTION}` as literal placeholders in the generated output.

**File: `src/prompts/agent_prompt_gen.user.md`**

Add a reminder line: "Use {AGENT_NAME}, {AGENT_DESCRIPTION}, and {TOOLS_SECTION} as literal placeholders — do not substitute them."

Then run `python src/prompts/upload_prompts.py` to re-upload to DB.

### 7. Tests
**New file: `tests/test_prompt_template.py`**

- All 3 placeholders resolve correctly
- No placeholders → passthrough (backward compat)
- Partial placeholders → only those resolve
- Empty tools → "no specific tools assigned"
- Missing name/description defaults

## Verification

1. Run `pytest tests/test_prompt_template.py` for unit tests
2. Start backend, generate a new system prompt via AI — verify it contains literal `{AGENT_NAME}`, `{AGENT_DESCRIPTION}`, `{TOOLS_SECTION}`
3. Execute the agent — verify the placeholders are replaced with actual values in the PydanticAI Agent's system prompt
4. Execute an existing agent with a baked prompt — verify it still works (backward compat)
5. Run `pytest tests/` for full regression
