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