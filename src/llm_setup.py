
from langchain_litellm import ChatLiteLLM
from typing import Optional


def create_llm(
    provider="openai",
    model="gpt-3.5-turbo",
    temperature: float = 0.0,
    config_path: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> ChatLiteLLM:
    """
    Create an agnostic LLM based on provider and model

    Args:
        provider (str): The provider name (e.g., 'anthropic', 'openai', 'gemini', 'lmstudio')
        model (str): The specific model name
        temperature (float): Model temperature (default: 0.0)
        config_path (str): Path to config.yaml file (unused, kept for compatibility)
        api_key (str): Optional API key for the provider
        base_url (str): Optional base URL for custom endpoints (e.g., LM Studio)

    Returns:
        ChatLiteLLM: Configured LLM instance
    """
    # Build kwargs for ChatLiteLLM
    kwargs = {
        "model": model,
        "temperature": temperature
    }

    # Add API key if provided
    if api_key:
        kwargs["api_key"] = api_key

    # Add base URL if provided
    if base_url:
        kwargs["api_base"] = base_url

    return ChatLiteLLM(**kwargs)

