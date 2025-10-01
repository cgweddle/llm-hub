import yaml
from typing import Optional
import os

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