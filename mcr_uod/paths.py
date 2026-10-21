"""Path resolution helpers.

The upstream MCR-UOD code hard-codes absolute paths to several lab machines
(e.g. ``/home/linianxin/aaai2026/confunder.txt``). This module resolves all
research assets relative to the repository root so every machine can run the
same code.

Repository layout (root)::

    UAV/
    |-- confunder.txt          # confounder prompt bank (36 descriptions)
    |-- mcr_uod/               # this package
    |-- ultralytics/           # vendored, patched ultralytics
"""

from __future__ import annotations

from pathlib import Path

# mcr_uod/paths.py -> mcr_uod/ -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

CONFOUNDER_FILE: Path = REPO_ROOT / "confunder.txt"
CONFIGS_DIR: Path = Path(__file__).resolve().parent / "configs"
DEFAULT_CAUSAL_CONFIG: Path = CONFIGS_DIR / "mcr_causal.yaml"


def resolve_asset(path: str | Path) -> Path:
    """Resolve ``path`` relative to the repository root.

    Absolute paths are returned unchanged; relative paths are anchored at the
    repository root so behaviour does not depend on the current working dir.

    Args:
        path: Absolute or repo-relative path to a research asset.

    Returns:
        The resolved absolute :class:`pathlib.Path`.

    Raises:
        FileNotFoundError: If the resolved path does not exist.
    """
    p = Path(path)
    resolved = p if p.is_absolute() else (REPO_ROOT / p)
    if not resolved.exists():
        raise FileNotFoundError(f"Asset not found: {resolved}")
    return resolved


def confounder_file() -> Path:
    """Return the path to ``confunder.txt`` (confounder prompt bank)."""
    return resolve_asset(CONFOUNDER_FILE)
