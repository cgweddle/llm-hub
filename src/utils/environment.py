import os


def is_hosted() -> bool:
    """Check if running in hosted/deployed environment.

    Returns True when ENVIRONMENT=HOSTED (set in deploy/podman-compose.yml).
    Defaults to LOCAL when not set.
    """
    return os.getenv("ENVIRONMENT", "LOCAL").upper() == "HOSTED"


def is_local() -> bool:
    """Check if running in local development environment."""
    return not is_hosted()
