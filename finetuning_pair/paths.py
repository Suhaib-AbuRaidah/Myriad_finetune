"""Keep all generated project artifacts inside the requested isolated folder."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def artifact_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(PACKAGE_ROOT):
        raise ValueError(f'Output/cache paths must be inside {PACKAGE_ROOT}: {path}')
    return path
