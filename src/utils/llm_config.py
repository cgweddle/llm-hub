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
    home = Path.home()
    llm_hub_dir = home / ".llm_hub"
    llm_hub_dir.mkdir(parents=True, exist_ok=True)
    return llm_hub_dir / "config.yaml"


# --- Private YAML helpers (local mode) ---

def _load_from_yaml() -> Dict[str, Any]:
    """Load LLM provider configuration from ~/.llm_hub/config.yaml"""
    config_path = get_llm_hub_config_path()

    if not config_path.exists():
        return {"models": []}

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        if config is None:
            config = {}
        if "models" not in config:
            config["models"] = []

        return config

    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing LLM config file: {e}")

def _save_to_yaml(models: List[Dict[str, Any]]) -> None:
    """Save LLM provider configuration to ~/.llm_hub/config.yaml"""
    config_path = get_llm_hub_config_path()
    config = {"models": models}
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.dump(config, file, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise IOError(f"Failed to save LLM config file: {e}")


# --- Public API (YAML-only; DB-backed per-user configs live in src/database/database.py) ---

def load_llm_provider_config() -> Dict[str, Any]:
    """Load LLM provider configuration from ~/.llm_hub/config.yaml."""
    return _load_from_yaml()

def save_llm_provider_config(models: List[Dict[str, Any]]) -> None:
    """Save LLM provider configuration to ~/.llm_hub/config.yaml."""
    _save_to_yaml(models)

def get_llm_config_by_name(model_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific LLM configuration by its name from ~/.llm_hub/config.yaml.

    Returns:
        Dict with name, provider, model, api_key, base_url — or None if not found.
    """
    try:
        config = load_llm_provider_config()
        for model_config in config.get("models", []):
            if model_config.get("name") == model_name:
                return model_config
        return None
    except Exception as e:
        raise ValueError(f"Failed to get LLM config for '{model_name}': {e}")

def resolve_model_name(llm_provider: str) -> str:
    """Resolve an LLM provider name to a 'provider:model' string for PydanticAI.
    Also sets api_key/base_url as env vars so PydanticAI can pick them up."""
    model_config = get_llm_config_by_name(llm_provider)
    if not model_config:
        raise ValueError(f"LLM provider '{llm_provider}' not found in config")

    provider = model_config.get("provider")
    model = model_config.get("model")
    api_key = model_config.get("api_key")
    base_url = model_config.get("base_url")

    if provider == "lmstudio":
        api_key = api_key or "lm-studio"
        base_url = base_url or "http://localhost:1234/v1"
        provider = "openai"

    if api_key:
        if provider == "anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url

    return f"{provider}:{model}"


MASKED_VALUE = "********"

def mask_credentials(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mask sensitive credential fields in LLM configurations"""
    masked_models = []
    for model in models:
        masked_model = model.copy()
        if masked_model.get('api_key'):
            masked_model['api_key'] = MASKED_VALUE
        masked_models.append(masked_model)
    return masked_models

def restore_masked_credentials(new_models: List[Dict[str, Any]], existing_models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restore masked credentials from existing configuration.

    When saving, if api_key is the masked value "********", replace it with the
    existing credential. This allows editing other fields without re-entering credentials.
    """
    existing_by_name = {model.get('name'): model for model in existing_models}

    restored_models = []
    for new_model in new_models:
        restored_model = new_model.copy()
        model_name = new_model.get('name')

        if restored_model.get('api_key') == MASKED_VALUE and model_name in existing_by_name:
            existing_model = existing_by_name[model_name]
            if existing_model.get('api_key'):
                restored_model['api_key'] = existing_model['api_key']

        restored_models.append(restored_model)

    return restored_models
