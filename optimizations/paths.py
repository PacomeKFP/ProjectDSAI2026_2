"""
optimizations/paths.py
═══════════════════════
Préfixe de sortie unifié — permet de rediriger TOUS les artefacts (résultats,
logs, modèles, profils) vers un dossier persistant (Google Drive sur Colab),
sans changer le code qui écrit.

Logique du préfixe :
  • En local      : préfixe vide → les chemins restent relatifs au projet.
  • Sur Colab     : "/content/drive/MyDrive/ProjectDSAI2026_2" → tout est écrit
                    directement sur le Drive monté (créé si besoin).
  • Override      : variable d'env PROJECT_OUTPUT_PREFIX, ou set_prefix().

Usage :
    from optimizations.paths import out_path, ensure_dir, project_prefix
    run_dir = ensure_dir("results", "optimization", run_id)   # crée + retourne Path
    csv     = out_path("results", "x.csv")                     # juste le Path préfixé

⚠ L'utilisateur monte lui-même le Drive (drive.mount). Ce module ne monte rien ;
  il se contente de préfixer les chemins et de créer les dossiers à la demande.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Préfixe par défaut sur Colab (Drive monté par l'utilisateur)
_COLAB_DEFAULT = "/content/drive/MyDrive/ProjectDSAI2026_2"

# Préfixe courant (modifiable via set_prefix)
_PREFIX: str | None = None


def _detect_prefix() -> str:
    """Détecte le préfixe : env > Colab > local (vide)."""
    env = os.environ.get("PROJECT_OUTPUT_PREFIX")
    if env is not None:
        return env
    if "google.colab" in sys.modules:
        return _COLAB_DEFAULT
    return ""


def project_prefix() -> str:
    """Retourne le préfixe courant (détecté à la première utilisation)."""
    global _PREFIX
    if _PREFIX is None:
        _PREFIX = _detect_prefix()
    return _PREFIX


def set_prefix(prefix: str) -> str:
    """Force le préfixe (ex. dans le notebook après avoir monté le Drive)."""
    global _PREFIX
    _PREFIX = prefix
    if prefix:
        Path(prefix).mkdir(parents=True, exist_ok=True)
    return _PREFIX


def out_path(*parts: str | os.PathLike) -> Path:
    """Chemin préfixé (sans créer de dossier)."""
    prefix = project_prefix()
    return (Path(prefix) / Path(*parts)) if prefix else Path(*parts)


def ensure_dir(*parts: str | os.PathLike) -> Path:
    """Chemin préfixé + création récursive du dossier. Retourne le Path."""
    p = out_path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(*parts: str | os.PathLike) -> Path:
    """Chemin préfixé d'un fichier + création de son dossier parent."""
    p = out_path(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def describe() -> str:
    prefix = project_prefix()
    where = prefix if prefix else "(local, préfixe vide)"
    return f"Préfixe de sortie : {where}"
