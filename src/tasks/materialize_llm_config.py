"""Materialize FLOW_RUNNER_LLM_CONFIG into ~/.llm_hub/config.yaml.

Invoked by deploy/flow-runner/entrypoint.sh at container startup, before the
flow runner is exec'd. Decouples the credential-transport mechanism (env var)
from the flow runner itself, which always reads the YAML like LOCAL mode does.
"""
import json
import os
import sys

from src.utils.llm_config import get_llm_hub_config_path, save_llm_provider_config


def main() -> int:
    raw = os.environ.get("FLOW_RUNNER_LLM_CONFIG")
    if not raw:
        return 0

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        print(
            f"materialize_llm_config: invalid JSON in FLOW_RUNNER_LLM_CONFIG: {e}",
            file=sys.stderr,
        )
        return 1

    if not isinstance(config, dict) or "models" not in config:
        print(
            "materialize_llm_config: FLOW_RUNNER_LLM_CONFIG must be a JSON object with a 'models' key",
            file=sys.stderr,
        )
        return 1

    save_llm_provider_config(config["models"])

    try:
        os.chmod(get_llm_hub_config_path(), 0o600)
    except OSError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
