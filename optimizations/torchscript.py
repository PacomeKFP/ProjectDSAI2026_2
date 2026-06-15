"""
optimizations/torchscript.py
═════════════════════════════
Compilation de graphe via TorchScript — fonctionne sur Windows ET Colab.

Dépendances :
  torch uniquement (natif, aucune dépendance externe, pas de Triton).

Ce que ça fait :
  TorchScript transforme un nn.Module Python dynamique en un graphe statique
  sérialisable et optimisable. C'est la voie de « compilation de graphe »
  disponible partout (contrairement à torch.compile/inductor qui exige Triton).

  Trois étapes :
    1. CAPTURE du graphe
       • torch.jit.script  : analyse le code Python source → graphe (gère le
         contrôle de flux : if/for/while). Voie privilégiée pour les modèles
         torchvision (écrits pour être scriptables).
       • torch.jit.trace   : exécute le modèle sur un exemple et enregistre les
         ops rencontrées. Plus simple mais ne capture pas le contrôle de flux
         dépendant des données (les branches non prises sont perdues).

    2. FREEZE (torch.jit.freeze)
       Inline les poids comme constantes, supprime les attributs inutiles, et
       permet des optimisations inter-procédurales (constant folding).

    3. OPTIMIZE_FOR_INFERENCE (torch.jit.optimize_for_inference)
       Applique des passes spécifiques inférence :
         • fusion Conv+BatchNorm (les stats BN sont figées → repliées dans la conv)
         • fusion Conv+ReLU via le fuser NNC
         • élimination de code mort, peephole optimizations

  Différence avec TensorRT :
    TorchScript fusionne au niveau du graphe PyTorch (NNC/nvFuser) mais reste
    dans le runtime PyTorch. TRT recompile en kernels CUDA natifs. TorchScript
    donne des gains plus modestes (×1.1–1.4) mais sans dépendance et partout.

  Fusion Conv+BN :
    C'est LE gain principal de optimize_for_inference. En inférence, BatchNorm
    devient une transformation affine constante (y = γ·(x−μ)/σ + β) qui peut être
    repliée dans les poids de la conv précédente → une couche au lieu de deux.
    Visible dans le rapport : les modules BatchNorm2d « disparaissent » du graphe.

Sauvegarde :
  Le graphe TorchScript est sérialisable (.ts) et rechargeable sans le code
  source Python original → portable entre machines (contrairement aux engines TRT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def optimize_with_torchscript(
    model: nn.Module,
    example_input=None,
    prefer: str = "script",
    freeze: bool = True,
    optimize: bool = True,
) -> nn.Module:
    """
    Compile le modèle en TorchScript + freeze + optimize_for_inference.

    Parameters
    ----------
    model         : nn.Module en mode eval
    example_input : entrée exemple pour le tracing (requis si prefer="trace"
                    ou si le scripting échoue). Format attendu par le modèle :
                    List[Tensor] pour RetinaNet, Tensor[B,C,H,W] pour EfficientDet.
    prefer        : "script" (défaut, robuste pour torchvision) ou "trace"
    freeze        : appliquer torch.jit.freeze (recommandé)
    optimize      : appliquer optimize_for_inference (fusion Conv+BN, etc.)

    Returns
    -------
    nn.Module TorchScript — même API que l'original.
    Si la compilation échoue, retourne le modèle original avec un avertissement.
    """
    model.eval()
    ts_model = None

    # ── 1. Capture du graphe ──────────────────────────────────────────────────
    if prefer == "script":
        try:
            ts_model = torch.jit.script(model)
            print("[TorchScript] [OK] Graphe capturé via torch.jit.script")
        except Exception as e:
            print(f"[TorchScript] script a échoué ({type(e).__name__}) — fallback trace")
            ts_model = _try_trace(model, example_input)
    else:
        ts_model = _try_trace(model, example_input)

    if ts_model is None:
        print("[TorchScript] [X] Compilation impossible — retour du modèle original.")
        return model

    # ── 2. Freeze ─────────────────────────────────────────────────────────────
    if freeze:
        try:
            ts_model = torch.jit.freeze(ts_model)
            print("[TorchScript] [OK] Freeze appliqué (poids inlinés, constant folding)")
        except Exception as e:
            print(f"[TorchScript] freeze ignoré ({type(e).__name__})")

    # ── 3. Optimize for inference ─────────────────────────────────────────────
    if optimize:
        try:
            ts_model = torch.jit.optimize_for_inference(ts_model)
            print("[TorchScript] [OK] optimize_for_inference (fusion Conv+BN+ReLU)")
        except Exception as e:
            print(f"[TorchScript] optimize_for_inference ignoré ({type(e).__name__})")

    print("  -> Le premier appel forward finalise les optimisations JIT (warmup).")
    return ts_model


def _try_trace(model: nn.Module, example_input) -> Optional[nn.Module]:
    """Tente un torch.jit.trace avec l'exemple fourni."""
    if example_input is None:
        print("[TorchScript] trace impossible : example_input manquant.")
        return None
    try:
        # strict=False : tolère les sorties dict/list (détection)
        traced = torch.jit.trace(model, example_input, strict=False)
        print("[TorchScript] [OK] Graphe capturé via torch.jit.trace")
        return traced
    except Exception as e:
        print(f"[TorchScript] trace a échoué ({type(e).__name__}: {e})")
        return None


def save_torchscript(model: nn.Module, path: str) -> str:
    """
    Sauvegarde un modèle TorchScript (.ts) — portable entre machines.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, torch.jit.ScriptModule) and not hasattr(model, "save"):
        raise TypeError("Le modèle n'est pas un ScriptModule — compiler d'abord.")
    torch.jit.save(model, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"[TorchScript] Sauvegardé -> {path}  ({size_mb:.1f} MB)")
    return path


def load_torchscript(path: str, device: str = "cuda") -> nn.Module:
    """Charge un modèle TorchScript sauvegardé (.ts)."""
    model = torch.jit.load(path, map_location=device)
    model.eval()
    print(f"[TorchScript] Chargé <- {path}")
    return model
