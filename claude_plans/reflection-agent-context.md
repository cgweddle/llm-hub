# Agents: User Prompt Templates with Message History and Input Variables

## Context

When a node executes in a complex (multi-node) agent graph, it receives **only the text output** of the previous node. It has no access to the previous node's system prompt, input, tool calls, or reasoning.

PydanticAI's `message_history` parameter can't be used to pass this context because **when `message_history` is non-empty, PydanticAI skips generating a new system prompt** — the successor node would lose its own identity/instructions.

Instead, we'll add two template variables to user prompts, resolved at runtime:
- **`{input}`** — the `node_input` text (preceding node's output, or the user's runtime input for the first node). Already available via the existing `node_outputs` mailbox.
- **`{message_history}`** — the accumulated PydanticAI message history serialized to text, giving full visibility into previous nodes' system prompts, reasoning, tool calls, and outputs.

Any agent type can use these variables. Simple agents and all complex agent node types get `{input}` by default.

## Plan

### 1. Add a message history serializer

**New file:** `src/utils/message_serializer.py`

```python
def serialize_messages(messages: List) -> str:
    """Serialize PydanticAI messages into human-readable text."""
```

Part type mapping:
- `SystemPromptPart` → `[System Prompt]`
- `UserPromptPart` → `[User]`
- `TextPart` → `[Assistant]`
- `ToolCallPart` → `[Tool Call: {name}]`
- `ToolReturnPart` → `[Tool Result: {name}]`
- `ThinkingPart` → skip
- `RetryPromptPart` → skip

Only called at template resolution time — messages are stored in native PydanticAI format throughout.

### 2. Replace `node_messages` dict with a single accumulating message list

**File:** `src/executors/agent_executor.py`

Complex agents are linear chains (no branching), so a single accumulating list represents the full conversation.

**Replace** the dict on line 298:
```python
# Before:
node_messages: Dict[str, List] = {}

# After:
messages: List = []  # Accumulating PydanticAI message history across all nodes
```

**After each node executes** (line 341-342), replace the list:
```python
# Before:
if messages is not None:
    node_messages[node_id] = messages

# After:
if node_result_messages is not None:
    messages = node_result_messages
```

**Rename the return variable** from `_run_sub_agent` to `node_result_messages` to avoid shadowing the accumulating `messages`.

**Pass accumulated messages to `_run_sub_agent`** (lines 332, 339):
```python
await self._run_sub_agent(node_id, nodes_config[node_id], node_input,
    predecessor_messages=messages if messages else None)
```

**Drop `message_history` from PydanticAI calls.** Each node always gets its own system prompt. Prior context is available via `{message_history}` in the user prompt template.

### 3. Add user prompt template resolution

**File:** `src/utils/prompt_template.py`

Add a new function alongside the existing `resolve_system_prompt_template`:

```python
def resolve_user_prompt_template(
    user_prompt: str,
    node_input: str,
    predecessor_messages: Optional[List] = None,
) -> str:
```

Resolves two placeholders from **different sources**:
- `{input}` → `node_input` (the text already being passed between nodes — preceding node's output, or user's runtime input for the first/simple node)
- `{message_history}` → `serialize_messages(predecessor_messages)` if messages exist, else empty string

If neither placeholder is present, returns `user_prompt` unchanged (backward compatible).

### 4. Call user prompt template resolution in the executor

**File:** `src/executors/agent_executor.py`

Update `_apply_user_prompt` (line 462):

```python
@staticmethod
def _apply_user_prompt(node_config: Dict, node_input: str,
                       predecessor_messages: Optional[List] = None) -> str:
    """Resolve user_prompt templates and prepend to node_input."""
    user_prompt = node_config.get("user_prompt", "").strip()
    if user_prompt:
        user_prompt = resolve_user_prompt_template(
            user_prompt, node_input, predecessor_messages
        )
        return f"{user_prompt}\n\n{node_input}"
    return node_input
```

Note: `{input}` resolves from `node_input` directly — always available, no dependency on messages. `{message_history}` resolves from `predecessor_messages` — `None` for simple agents and first nodes (resolves to empty string).

**Simplified `_run_sub_agent` signature:**
```python
async def _run_sub_agent(
    self, node_id: str, node_config: Dict, node_input: str,
    predecessor_messages: Optional[List] = None,
) -> Tuple[str, Optional[List], Optional[str]]:
```

Thread `predecessor_messages` through to `_run_pydanticai_node` → `_apply_user_prompt`.

### 5. Add `defaultUserPrompt` to agent templates

**File:** `frontend/src/lib/agentTemplates.ts`

Add `defaultUserPrompt` to the `AgentTemplate` interface:
```typescript
export interface AgentTemplate {
  type: AgentTypeKey;
  name: string;
  description: string;
  color: string;
  borderColor: string;
  icon: string;
  defaultSystemPrompt: string;
  defaultUserPrompt: string;  // new
}
```

Default values:
- **planning:** `defaultUserPrompt: '{input}'`
- **react:** `defaultUserPrompt: '{input}'`
- **custom:** `defaultUserPrompt: '{input}'`
- **reflection:** uses the fuller default with both variables:
```
The previous agent received this input:
{input}

Here is the full conversation from the previous agent:
{message_history}

Evaluate the output above based on the previous agent's instructions and the original input.
```

Update the reflection **system prompt**:
```
You are a Reflection Agent that critically evaluates and improves outputs.

You will receive context about the previous agent node, including:
- The original input the previous agent received
- The full conversation history showing the agent's instructions, reasoning, tool usage, and output

Your responsibilities:
1. Evaluate whether the output fulfills the previous agent's instructions
2. Check if the output properly addresses the original input
3. Identify potential issues, gaps, or areas for improvement
4. Suggest specific improvements or corrections
5. When appropriate, provide an improved version

Evaluation criteria:
- Fulfillment: Does the output satisfy the previous agent's instructions?
- Relevance: Does it properly address the original input?
- Accuracy: Is the information correct?
- Completeness: Are there missing elements?
- Clarity: Is it easy to understand?
- Consistency: Are there contradictions?
- Quality: Does it meet the expected standards?

Always provide constructive feedback with specific suggestions for improvement.
```

### 6. Add User Prompt field to the agent creation modal

The agent creation modal in `+page.svelte` is shared between simple agent creation and complex agent node configuration. It currently has no User Prompt field.

**File:** `frontend/src/routes/+page.svelte`

**Add state variable** (after line 133):
```typescript
let newAgentUserPrompt = '{input}';
```

**Add User Prompt textarea** to the modal (after the System Prompt section, ~line 1872):
```svelte
<!-- User Prompt Section -->
<div class="create-tool-section">
  <div class="create-tool-section-label">User Prompt</div>
  <div class="create-tool-helper-text" style="margin-bottom: 8px;">
    Optional. Prepended to runtime input. Use {input} for the preceding node's output and {message_history} for the full conversation history.
  </div>
  <textarea
    class="create-tool-textarea"
    bind:value={newAgentUserPrompt}
    placeholder="e.g. {input}"
    rows="4"
    style="min-height: 80px;"
  ></textarea>
</div>
```

**Include `user_prompt` in `handleCreateAgent`** (line 1105 area, inside the `"main"` node config):
```typescript
system_prompt: newAgentSystemPrompt.trim(),
user_prompt: newAgentUserPrompt.trim(),  // new
```

**Include `user_prompt` in complex node configuration** — when the modal is used for `isConfiguringComplexNode`, pass `newAgentUserPrompt` to the AgentBuilder.

**Pre-fill from template** — when opening the modal for a complex node with a template:
```typescript
newAgentSystemPrompt = template.defaultSystemPrompt;
newAgentUserPrompt = template.defaultUserPrompt;  // new
```

**Reset on close** — in `closeCreateAgentModal` (line 1267):
```typescript
newAgentUserPrompt = '{input}';
```

### 7. Wire `defaultUserPrompt` into complex agent node creation

**File:** `frontend/src/routes/AgentBuilder.svelte`

When creating a new agent node from a template on the canvas:
```typescript
node.data.user_prompt = template.defaultUserPrompt;
```

**File:** `frontend/src/lib/agentGraphBuilder.ts` — already handles `user_prompt: node.data.user_prompt || ''` at line 141, no change needed.

## Data Flow

```
{input} = node_input (text, always available)
{message_history} = serialized from messages list (PydanticAI native format)

messages: List  (single accumulating PydanticAI message list)
        │
        │  starts empty
        ▼
Node A executes → messages = result.all_messages()
        │
        │  node_input for B = A's output text (via node_outputs mailbox)
        │  predecessor_messages for B = messages (accumulated PydanticAI history)
        ▼
Node B executes:
  1. _apply_user_prompt resolves {input} from node_input, {message_history} from messages
  2. PydanticAI runs with B's OWN system prompt + resolved user prompt + node_input
  3. messages = result.all_messages()
        │
        │  if loop back, messages contains full history
        ▼
```

## Files to Modify

- `src/utils/message_serializer.py` — **new** — serialize PydanticAI messages to text
- `src/utils/prompt_template.py` — add `resolve_user_prompt_template`
- `src/executors/agent_executor.py` — replace `node_messages` dict with single `messages` list, thread `predecessor_messages` through to `_apply_user_prompt`, drop `message_history` param from PydanticAI calls
- `frontend/src/lib/agentTemplates.ts` — add `defaultUserPrompt` field to all templates
- `frontend/src/routes/+page.svelte` — add User Prompt field to agent creation modal, include in `graph_config`, pre-fill from template, reset on close
- `frontend/src/routes/AgentBuilder.svelte` — populate `user_prompt` from template on canvas node creation

## Verification

1. Run existing tests: `pytest tests/` — no regressions
2. Create a **simple agent** with default `{input}` user prompt — verify it appears in the modal and is saved in `graph_config.nodes.main.user_prompt`
3. Create a **complex agent** node from a reflection template — verify user prompt is pre-filled with the full default including `{message_history}` and `{input}`
4. Execute a 2-node complex agent (e.g. writer → reviewer with a loop edge back). Verify in LangFuse that the second node's user message contains the serialized conversation from the first node.
5. Verify agents without template variables in their user_prompt behave as before.
