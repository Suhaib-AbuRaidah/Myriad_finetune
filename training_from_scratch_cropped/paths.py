"""Keep generated artifacts separate from all previous implementations."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def artifact_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Output/cache must be inside {ROOT}: {path}")
    return path
