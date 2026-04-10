import yaml
from typing import Optional, List, Dict, Any
import os
from pathlib import Path

from .environment import is_hosted


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


# --- Private DB helpers (hosted mode) ---

def _get_db_session():
    """Get a database session for hosted mode."""
    from src.database.database import get_session
    return get_session()

def _load_from_database(user_id: int) -> Dict[str, Any]:
    """Load LLM provider configs from database for a specific user."""
    from src.database.database import get_user_llm_provider_configs
    session = _get_db_session()
    try:
        models = get_user_llm_provider_configs(session, user_id)
        return {"models": models}
    finally:
        session.close()

def _save_to_database(user_id: int, models: List[Dict[str, Any]]) -> None:
    """Save LLM provider configs to database for a specific user.

    Uses a sync strategy: update existing configs by name, create new ones,
    delete configs that are no longer in the list.
    """
    from src.database.database import (
        get_user_llm_provider_configs,
        create_llm_provider_config,
        update_llm_provider_config,
        delete_llm_provider_config,
    )
    session = _get_db_session()
    try:
        existing = get_user_llm_provider_configs(session, user_id)
        existing_by_name = {m["name"]: m for m in existing}
        new_names = {m.get("name") for m in models}

        # Delete configs no longer in the list
        for existing_model in existing:
            if existing_model["name"] not in new_names:
                delete_llm_provider_config(session, existing_model["id"])

        # Create or update
        for model in models:
            name = model.get("name")
            if name in existing_by_name:
                # Update existing
                config_id = existing_by_name[name]["id"]
                update_llm_provider_config(session, config_id,
                    name=model.get("name"),
                    provider=model.get("provider"),
                    model=model.get("model"),
                    api_key=model.get("api_key"),
                    base_url=model.get("base_url"),
                )
            else:
                # Create new
                create_llm_provider_config(session, user_id,
                    name=model.get("name"),
                    provider=model.get("provider"),
                    model=model.get("model"),
                    api_key=model.get("api_key"),
                    base_url=model.get("base_url"),
                )
    finally:
        session.close()


# --- Public API (routes to YAML or DB based on environment) ---

def load_llm_provider_config(user_id: Optional[int] = None) -> Dict[str, Any]:
    """Load LLM provider configuration.

    In hosted mode with a user_id, loads from the database.
    Otherwise, loads from ~/.llm_hub/config.yaml.
    """
    if is_hosted() and user_id is not None:
        return _load_from_database(user_id)
    return _load_from_yaml()

def save_llm_provider_config(models: List[Dict[str, Any]], user_id: Optional[int] = None) -> None:
    """Save LLM provider configuration.

    In hosted mode with a user_id, saves to the database.
    Otherwise, saves to ~/.llm_hub/config.yaml.
    """
    if is_hosted() and user_id is not None:
        _save_to_database(user_id, models)
    else:
        _save_to_yaml(models)

def get_llm_config_by_name(model_name: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Get a specific LLM configuration by its name.

    Args:
        model_name: Name of the LLM configuration (e.g., "Production Claude")
        user_id: User ID for hosted mode DB lookup. Ignored in local mode.

    Returns:
        Dict with name, provider, model, api_key, base_url — or None if not found.
    """
    try:
        config = load_llm_provider_config(user_id=user_id)
        models = config.get("models", [])

        for model_config in models:
            if model_config.get("name") == model_name:
                return model_config

        return None

    except Exception as e:
        raise ValueError(f"Failed to get LLM config for '{model_name}': {e}")

def resolve_model_name(llm_provider: str, user_id: Optional[int] = None) -> str:
    """Resolve an LLM provider name to a 'provider:model' string for PydanticAI.
    Also sets api_key/base_url as env vars so PydanticAI can pick them up."""
    model_config = get_llm_config_by_name(llm_provider, user_id=user_id)
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
