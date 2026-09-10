"""Small configuration helpers shared by services."""

import os
from pathlib import Path


def read_secret(name: str) -> str | None:
    """Read a secret from NAME or from the file referenced by NAME_FILE."""
    value = os.environ.get(name)
    if value:
        return value
    file_path = os.environ.get(f"{name}_FILE")
    if not file_path:
        return None
    try:
        return Path(file_path).read_text().strip() or None
    except OSError as exc:
        raise RuntimeError(f"Could not read {name} from {file_path}") from exc
