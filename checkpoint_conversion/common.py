"""Shared helpers for TensorFlow-to-PyTorch checkpoint conversion."""

from __future__ import annotations

import sys
from pathlib import Path


TORCH_FALLBACK_PATHS = [
    Path("/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages"),
    Path("/Users/ricky/Library/Python/3.11/lib/python/site-packages"),
]


def ensure_repo_on_path() -> Path:
    """Add the repo root to sys.path and return it."""
    repo_root = Path(__file__).resolve().parents[1]
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root


def import_torch():
    """Import torch, falling back to known Python 3.11 site-packages if needed.

    TensorFlow is installed into a temporary Python 3.11 virtualenv for the
    conversion scripts. In this repo, torch already exists in the system
    Python 3.11 site-packages. We append those paths only if torch is not
    already importable.
    """
    try:
        import torch  # type: ignore

        return torch
    except ModuleNotFoundError:
        for path in TORCH_FALLBACK_PATHS:
            if path.exists():
                sys.path.append(str(path))
        import torch  # type: ignore

        return torch

