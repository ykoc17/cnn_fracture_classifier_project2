"""Repository-rooted paths independent of the caller's working directory."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_ROOT = REPO_ROOT / "data" / "processed"


def resolve_repo_path(path: str | Path, *, create_parent: bool = False) -> Path:
    """Resolve relative paths from the repository root, never from ``cwd``."""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved
