# PydanticAI Integration

This document describes the PydanticAI integration into the LLM Hub platform.

## Overview

PydanticAI has been integrated as an **alternative agent type** alongside the existing Google ADK ReAct agents. Users can now choose between:

- **`react`**: Google ADK ReAct-style agents (existing)
- **`pydanticai`**: PydanticAI agents with structured outputs and streaming (new)

## Key Features

### ✅ Dual Agent Support
- Both Google ADK and PydanticAI agents work side-by-side
- Automatic routing based on `agent_type` field
- No breaking changes to existing functionality

### ✅ Dynamic Tool Conversion
- Database tools automatically work with both agent types
- JSON Schema → Pydantic model conversion at runtime
- No database schema changes required

### ✅ Structured Outputs
- Define `result_schema` in agent metadata
- Automatic Pydantic validation of agent responses
- Type-safe outputs with compile-time guarantees

### ✅ Streaming Support
- Native streaming for PydanticAI agents
- Server-Sent Events (SSE) API
- Real-time response delivery

### ✅ Multi-Provider Support
- Anthropic (Claude models)
- OpenAI (GPT models)
- Google Gemini
- LM Studio (local models)
- Azure OpenAI

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Backend                     │
│                 (src/api/backend.py)                 │
└─────────────────────┬───────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │   AgentExecutor       │
          │ (unified routing)     │
          └───────┬───────┬───────┘
                  │       │
      ┌───────────▼       ▼───────────┐
      │                               │
┌─────▼─────┐                   ┌────▼──────┐
│  ReactAgent│                   │PydanticAI │
│  Factory   │                   │  Factory  │
└─────┬─────┘                   └────┬──────┘
      │                               │
      │                         ┌─────▼─────┐
      │                         │   Tool    │
      │                         │ Converter │
      │                         └─────┬─────┘
      │                               │
      └───────────┬───────────────────┘
                  │
          ┌───────▼────────┐
          │  Database      │
          │  (SQLite/PG)   │
          └────────────────┘
```

## New Components

### 1. **Agent Factory** (`src/factories/pydanticai_agent_factory.py`)
- Creates PydanticAI agents from database records
- Maps `~/.llm_hub/config.yaml` providers to PydanticAI model classes
- Handles provider-specific authentication and custom base URLs
- Loads and registers tools automatically
- Structured output configuration
- Validation before execution

### 2. **Tool Converter** (`src/converters/pydanticai_tool_converter.py`)
- Converts database Tool records to PydanticAI tool functions
- Dynamic Pydantic model generation from JSON schemas
- Handles sync/async functions
- Caching for performance

### 3. **Agent Executor** (`src/executors/agent_executor.py`)
- Unified execution service for both agent types
- Manages execution lifecycle (create → run → store)
- Message history storage
- Cost tracking for PydanticAI
- Streaming support

## API Endpoints

### Execute Agent (Updated)
```http
POST /agents/{agent_id}/execute
Content-Type: application/json

{
  "user_id": 1,
  "input_data": "What is 2+2?",
  "stream": false
}
```

**Response:**
```json
{
  "execution_id": 42,
  "status": "completed",
  "result": "The answer is 4.",
  "messages": [...],
  "cost": {
    "total_tokens": 150,
    "input_tokens": 50,
    "output_tokens": 100
  },
  "agent_type": "pydanticai"
}
```

### Create PydanticAI Agent (New)
```http
POST /agents/pydanticai/create?user_id=1
Content-Type: application/json

{
  "name": "Math Assistant",
  "description": "Helps with calculations",
  "agent_type": "pydanticai",
  "system_prompt": "You are a math assistant.",
  "llm_config": {
    "model_name": "My Anthropic Config"
  },
  "tools_config": {
    "tool_ids": [1, 2, 3]
  }
}
```

**With Structured Output:**
```http
POST /agents/pydanticai/create?user_id=1&result_schema={"type":"object","properties":{"answer":{"type":"string"},"confidence":{"type":"number"}}}
```

## Usage Examples

### 1. Create a PydanticAI Agent via API

```python
import requests

# Create agent
response = requests.post(
    "http://localhost:8000/agents/pydanticai/create",
    params={"user_id": 1},
    json={
        "name": "Research Assistant",
        "description": "Helps with research",
        "agent_type": "pydanticai",
        "system_prompt": "You are a helpful research assistant.",
        "llm_config": {"model_name": "My Anthropic Config"},
        "tools_config": {"tool_ids": [1, 2]}
    }
)

agent_id = response.json()["id"]
```

### 2. Execute Agent

```python
# Execute agent
response = requests.post(
    f"http://localhost:8000/agents/{agent_id}/execute",
    json={
        "user_id": 1,
        "input_data": "What is the capital of France?",
        "stream": False
    }
)

result = response.json()
print(result["result"])  # Agent's answer
```

### 3. Execute with Streaming

```python
import sseclient

response = requests.post(
    f"http://localhost:8000/agents/{agent_id}/execute",
    json={
        "user_id": 1,
        "input_data": "Write a poem about AI",
        "stream": True
    },
    stream=True
)

client = sseclient.SSEClient(response)
for event in client.events():
    chunk = json.loads(event.data)
    if chunk["type"] == "message":
        print(chunk["content"], end="", flush=True)
    elif chunk["type"] == "complete":
        print(f"\n\nCost: {chunk['cost']}")
```

### 4. Direct Agent Usage (Python)

```python
from database.database import get_session
from factories.pydanticai_agent_factory import PydanticAIAgentFactory

# Create agent from database
session = get_session()
factory = PydanticAIAgentFactory(session)
agent = factory.create_from_database(agent_id=5)

# Run agent
result = await agent.run("What is 2+2?")
print(result.data)  # Agent's answer
print(result.cost())  # Token usage
```

## Configuration

### LLM Providers

Configure providers in `~/.llm_hub/config.yaml`:

```yaml
models:
  - name: "My Anthropic Config"
    provider: "anthropic"
    model: "claude-3-5-sonnet-20241022"
    api_key: "sk-ant-..."

  - name: "My OpenAI Config"
    provider: "openai"
    model: "gpt-4"
    api_key: "sk-..."

  - name: "Local LM Studio"
    provider: "lmstudio"
    model: "local-model"
    base_url: "http://localhost:1234/v1"
```

### Structured Outputs

Define structured output schema in `agent_metadata`:

```python
result_schema = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The main answer"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score 0-1"
        },
        "sources": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of sources used"
        }
    },
    "required": ["answer", "confidence"]
}
```

## Testing

Run the integration test:

```bash
cd /Users/chris/Documents/repos/llm-hub
python test_pydanticai_integration.py
```

This will:
1. Create a test user
2. Create a simple calculator tool
3. Create a PydanticAI agent
4. Execute the agent
5. Verify results

## Database Schema

No schema changes required! PydanticAI integration uses existing tables:

- **`agents`**: `agent_type` field supports `"pydanticai"` value
- **`agents.agent_metadata`**: Stores `result_schema` for structured outputs
- **`tools`**: All existing tools work with PydanticAI
- **`executions`**: Stores execution records for both agent types
- **`messages`**: Stores conversation history

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing Google ADK agents continue working
- No API changes for existing endpoints
- Database schema unchanged
- Frontend changes optional

## Performance

### Tool Conversion Caching
Tool conversions are cached per tool ID to avoid repeated:
- JSON Schema → Pydantic model creation
- Function code compilation

### In-Process Execution
PydanticAI tools run in-process (unlike subprocess-based Google ADK), providing:
- Lower latency
- No pickle/unpickle overhead
- Better memory efficiency

## Troubleshooting

### "LLM config not found"
**Problem**: `LLM config 'X' not found in ~/.llm_hub/config.yaml`

**Solution**: Configure the LLM provider in `~/.llm_hub/config.yaml` with matching name.

### "Agent is not a PydanticAI agent"
**Problem**: Trying to execute non-PydanticAI agent with PydanticAI factory

**Solution**: Use unified `AgentExecutor` which automatically routes to correct factory.

### "Tool conversion failed"
**Problem**: Complex JSON schema can't be converted to Pydantic model

**Solution**: Simplify the schema or use `Any` type as fallback. Check tool's `input_schema` format.

### Streaming not working
**Problem**: Streaming responses not appearing

**Solution**:
- Ensure `stream: true` in request
- Use proper SSE client
- Check CORS headers
- Verify PydanticAI agent type

## Future Enhancements

- [ ] Agent composition (agent calling agent as tool)
- [ ] Custom Pydantic validators for outputs
- [ ] Automatic retry logic with exponential backoff
- [ ] Detailed metrics dashboard (token usage, cost, latency)
- [ ] Tool result caching within execution session
- [ ] Batch execution support
- [ ] Multi-agent collaboration patterns

## Contributing

When adding new features:
1. Maintain backward compatibility with Google ADK agents
2. Add tests to `test_pydanticai_integration.py`
3. Update this documentation
4. Follow existing code patterns (Factory, Executor, Converter)

## Support

- **Issues**: https://github.com/your-repo/llm-hub/issues
- **Documentation**: See `/docs` directory
- **Examples**: See `test_pydanticai_integration.py`
