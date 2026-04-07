# PydanticAI Agent Factory

## Overview

`pydanticai_agent_factory.py` creates complete PydanticAI agents from database configurations. It handles:
- Loading agent records from the database
- Creating PydanticAI model instances from LLM configurations
- Converting and registering database tools
- Configuring structured outputs
- Validating agent configurations

The factory encapsulates all the complexity of bridging database-stored configurations with PydanticAI's runtime requirements.

## Configuration Source

LLM configurations are stored in `~/.llm_hub/config.yaml` with this structure:

```yaml
models:
  - name: "Production Claude"
    provider: "anthropic"
    model: "claude-3-5-sonnet-20241022"
    api_key: "sk-ant-..."
  - name: "Local LM Studio"
    provider: "lmstudio"
    model: "llama-3"
    base_url: "http://localhost:1234/v1"
```

The function `get_llm_config_by_name()` in `src/utils.py` loads a config by its `name` field and returns it as a dictionary with keys: `provider`, `model`, `api_key`, `base_url`.

## Class Structure

### `PROVIDER_MAP`

A module-level dict that maps provider strings to PydanticAI model classes:

```python
PROVIDER_MAP = {
    "anthropic": AnthropicModel,
    "openai": OpenAIModel,
    "lmstudio": OpenAIModel,
}
```

Adding a new OpenAI-compatible provider is a single line addition to this dict.

### `PydanticAIAgentFactory`

The main factory class that orchestrates agent creation with these key methods:

**`create_from_database(agent_id)`**
1. Loads agent record from database
2. Validates agent type is "pydanticai"
3. Gets LLM configuration
4. Creates model instance
5. Configures structured output (if specified)
6. Creates PydanticAI Agent
7. Registers tools

**`_create_model(llm_config)` (private)**
Creates PydanticAI model instances:
1. Reads `provider`, `model`, `api_key`, `base_url` from the config dict
2. Looks up the model class from `PROVIDER_MAP`
3. Applies LM Studio defaults if needed (dummy API key, default base URL)
4. Builds `kwargs` conditionally — only includes values that are set
5. Returns `model_class(**kwargs)`

**`validate_agent_config(agent_id)`**
Pre-flight validation without creating the agent, useful for UI validation.

## Design Decisions

### Single Method with Provider Map

All providers follow the same pattern: look up a class, build `kwargs`, call the constructor. A dict lookup replaces what was previously separate methods per provider. The only special case (LM Studio defaults) is a simple `if` block within the single method.

### Why Static Methods

The factory holds no state. Every call to `create_model()` is self-contained — it takes a config dict and returns a model. Making the method static reflects this and avoids unnecessary instantiation.

### Why LM Studio Uses `OpenAIModel`

LM Studio exposes an OpenAI-compatible REST API on `localhost:1234/v1`. Rather than creating a custom model class, the factory reuses `OpenAIModel` with:
- A dummy API key (`"lm-studio"`) since LM Studio doesn't validate keys
- A default `base_url` of `http://localhost:1234/v1` if none is configured

This works because PydanticAI's `OpenAIModel` communicates via the OpenAI HTTP API format, which LM Studio implements.

### Why Conditional `kwargs` Instead of Passing `None`

The method builds a `kwargs` dict and only adds keys when values are present:

```python
kwargs = {"model_name": model_name}
if api_key:
    kwargs["api_key"] = api_key
```

This avoids passing `None` to PydanticAI constructors, which may not accept `None` for optional parameters. It lets PydanticAI use its own defaults (e.g., reading `ANTHROPIC_API_KEY` from environment variables) when no explicit key is provided.

## How It Fits Into the System

```
~/.llm_hub/config.yaml
        │
        ▼
  get_llm_config_by_name()          (src/utils.py)
        │
        ▼
  PydanticAIAgentFactory            (src/factories/pydanticai_agent_factory.py)
    ├── _get_llm_config()           Loads config from yaml
    ├── _create_model()             Creates model instance
    ├── _get_result_type()          Configures structured output
    └── _register_tools()           Converts & registers tools
        │
        ▼
  Agent(model=model, ...)           Complete PydanticAI agent
```

The `PydanticAIAgentFactory` handles everything in one place:
1. Loads agent record from database
2. Resolves LLM config via `get_llm_config_by_name()`
3. Creates model instance directly using `_create_model()`
4. Registers tools via `PydanticAIToolConverter`
5. Returns fully configured agent

## Convenience Function

`create_pydanticai_agent_from_database(agent_id)` is a module-level shortcut that creates a factory instance and returns a configured agent in one call, useful for scripts and testing.

## Supported Providers

| Provider | PydanticAI Class | API Key Required | Base URL Required |
|----------|-----------------|------------------|-------------------|
| `anthropic` | `AnthropicModel` | Optional (env fallback) | No |
| `openai` | `OpenAIModel` | Optional (env fallback) | No |
| `lmstudio` | `OpenAIModel` | No (uses dummy) | Optional (defaults to localhost:1234) |
