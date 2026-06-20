"""
optimizations/paths.py
=======================
Unified output prefix -- redirects ALL artifacts (results, logs, models,
profiles) to a persistent folder (Google Drive on Colab) without changing
the writing code.

Prefix logic:
  * Local       : empty prefix -> paths stay relative to the project.
  * On Colab    : "/content/drive/MyDrive/ProjectDSAI2026_2" -> everything is
                  written directly on the mounted Drive (created if needed).
  * Override    : PROJECT_OUTPUT_PREFIX env variable, or set_prefix().

Usage:
    from optimizations.paths import out_path, ensure_dir, project_prefix
    run_dir = ensure_dir("results", "optimization", run_id)   # creates + returns Path
    csv     = out_path("results", "x.csv")                     # just the prefixed Path

[!] The user mounts Drive themselves (drive.mount). This module does not mount
  anything; it only prefixes paths and creates folders on demand.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default prefix on Colab (Drive mounted by the user)
_COLAB_DEFAULT = "/content/drive/MyDrive/ProjectDSAI2026_2"

# Current prefix (modifiable via set_prefix)
_PREFIX: str | None = None


def _detect_prefix() -> str:
    """Detect the prefix: env > Colab > local (empty)."""
    env = os.environ.get("PROJECT_OUTPUT_PREFIX")
    if env is not None:
        return env
    if "google.colab" in sys.modules:
        return _COLAB_DEFAULT
    return ""


def project_prefix() -> str:
    """Return the current prefix (detected on first use)."""
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = _detect_prefix()
    return _PREFIX


def set_prefix(prefix: str) -> str:
    """Force the prefix (e.g. in the notebook after mounting Drive)."""
    global _PREFIX
    _PREFIX = prefix
    if prefix:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    return _PREFIX


def out_path(*parts: str | os.PathLike) -> Path:
    """Prefixed path (without creating any folder)."""
    prefix = project_prefix()
    return (Path(prefix) / Path(*parts)) if prefix else Path(*parts)


def ensure_dir(*parts: str | os.PathLike) -> Path:
    """Prefixed path + recursive folder creation. Returns the Path."""
    p = out_path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(*parts: str | os.PathLike) -> Path:
    """Prefixed file path + creation of its parent folder."""
    p = out_path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def describe() -> str:
    prefix = project_prefix()
    where = prefix if prefix else "(local, empty prefix)"
    return f"Output prefix: {where}"
