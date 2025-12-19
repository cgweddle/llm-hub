import yaml
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

def load_model_config(config_path: Optional[str] = None) -> dict:
    """Load model configuration from YAML file"""
    try:
        if config_path is None:
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            config_path = os.path.join(project_root, "config.yaml")
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing config file: {e}")

def get_llm_hub_config_path() -> Path:
    """Get the path to the LLM Hub config directory

    Works cross-platform on Linux, macOS, and Windows.
    Returns path to ~/.llm_hub/config.yaml
    """
    # Path.home() works on all platforms (Linux, Mac, Windows)
    home = Path.home()
    llm_hub_dir = home / ".llm_hub"

    # Create directory if it doesn't exist (cross-platform)
    llm_hub_dir.mkdir(parents=True, exist_ok=True)

    return llm_hub_dir / "config.yaml"

def load_llm_provider_config() -> Dict[str, Any]:
    """Load LLM provider configuration from ~/.llm_hub/config.yaml

    Works cross-platform on Linux, macOS, and Windows.

    Returns:
        dict with key:
            - models: List of LLM provider configurations
    """
    config_path = get_llm_hub_config_path()

    # Return empty config if file doesn't exist
    if not config_path.exists():
        return {
            "models": []
        }

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        # Ensure config has the expected structure
        if config is None:
            config = {}

        if "models" not in config:
            config["models"] = []

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing LLM config file: {e}")

def save_llm_provider_config(models: List[Dict[str, Any]]) -> None:
    """Save LLM provider configuration to ~/.llm_hub/config.yaml

    Works cross-platform on Linux, macOS, and Windows.

    Args:
        models: List of LLM provider configurations (each with name, provider, api_key, base_url, model)
    """
    config_path = get_llm_hub_config_path()

    config = {
        "models": models
    }

    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise IOError(f"Failed to save LLM config file: {e}")

def get_llm_config_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific LLM configuration by its name from ~/.llm_hub/config.yaml

    Args:
        model_name: Name of the LLM configuration to retrieve (e.g., "Production Claude")

    Returns:
        Dict containing the LLM configuration with keys:
            - name: str
            - provider: str (e.g., "anthropic", "openai", "gemini", "lmstudio")
            - model: str
            - api_key: Optional[str]
            - base_url: Optional[str]
        Returns None if not found

    Example:
        >>> config = get_llm_config_by_name("Production Claude")
        >>> print(config)
        {
            "name": "Production Claude",
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20241022",
            "api_key": "sk-ant-...",
            "base_url": None
        }
    """
    try:
        config = load_llm_provider_config()
        models = config.get("models", [])

        # Search for the model by name
        for model_config in models:
            if model_config.get("name") == model_name:
                return model_config

        # Not found
        return None

    except Exception as e:
        raise ValueError(f"Failed to get LLM config for '{model_name}': {e}")

MASKED_VALUE = "********"

def mask_credentials(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mask sensitive credential fields in LLM configurations

    Args:
        models: List of LLM configurations

    Returns:
        List of LLM configurations with masked credentials
    """
    masked_models = []
    for model in models:
        masked_model = model.copy()

        # Mask API key if present
        if masked_model.get('api_key'):
            masked_model['api_key'] = MASKED_VALUE

        # Note: base_url is not sensitive, so we don't mask it

        masked_models.append(masked_model)

    return masked_models

def restore_masked_credentials(new_models: List[Dict[str, Any]], existing_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restore masked credentials from existing configuration

    When saving, if api_key is the masked value "********", replace it with the
    existing credential from the file. This allows editing other fields without
    exposing or requiring re-entry of credentials.

    Args:
        new_models: New models list (may contain masked values)
        existing_models: Existing models list (with real credentials)

    Returns:
        Models list with masked values replaced by real credentials
    """
    # Create a lookup map of existing models by name
    existing_by_name = {model.get('name'): model for model in existing_models}

    restored_models = []
    for new_model in new_models:
        restored_model = new_model.copy()
        model_name = new_model.get('name')

        # If api_key is masked and we have an existing model with same name
        if restored_model.get('api_key') == MASKED_VALUE and model_name in existing_by_name:
            # Restore the real api_key from existing config
            existing_model = existing_by_name[model_name]
            if existing_model.get('api_key'):
                restored_model['api_key'] = existing_model['api_key']

        restored_models.append(restored_model)

    return restored_models