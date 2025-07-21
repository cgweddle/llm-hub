
from langchain_litellm import ChatLiteLLM
from utils import load_model_config
from typing import Optional


def create_llm(provider="openai", model="gpt-3.5-turbo", temperature: float = 0.0, config_path: Optional[str] = None) -> ChatLiteLLM:
    """
    Create an agnostic LLM based on provider and model, validated against config.yaml
    
    Args:
        provider (str): The provider name (e.g., 'bedrock', 'mistral')
        model (str): The specific model name
        temperature (float): Model temperature (default: 0.7)
        config_path (str): Path to config.yaml file (default: "config.yaml")
    
    Returns:
        ChatLiteLLM: Configured LLM instance
        
    Raises:
        ValueError: If provider or model is not found in config
    """
    config = load_model_config()
    
    providers = config['models']['provider']
    if provider not in providers:
        available_providers = list(providers.keys())
        raise ValueError(f"Provider '{provider}' not found in config. Available providers: {available_providers}")


    available_models = providers[provider]
    if model not in available_models:
        raise ValueError(f"Model '{model}' not found for provider '{provider}'. Available models: {available_models}")

    
    return ChatLiteLLM(
        model=model,
        temperature=temperature
    )

