"""
optimizations/torch_compile.py
═══════════════════════════════
Compilation de graphe via torch.compile (PyTorch 2.x).

Dépendances :
  torch >= 2.0
  Pour le backend "inductor" (le plus rapide) : triton
    • Linux / Colab : Triton est inclus avec PyTorch CUDA → backend inductor OK.
    • Windows       : Triton n'est PAS packagé par défaut → inductor lève
                      TritonMissing. On bascule automatiquement sur "cudagraphs".

Ce que ça fait — deux backends pertinents :

  ┌─ inductor (Linux/Colab) ────────────────────────────────────────────────┐
  │ Génère des kernels Triton fusionnés à partir du graphe FX.               │
  │ Fusion verticale (Conv→BN→ReLU) + horizontale (ops parallèles).          │
  │ Gain typique : ×1.3 à ×2.0. Premier appel lent (codegen + autotune).     │
  └──────────────────────────────────────────────────────────────────────────┘

  ┌─ cudagraphs (Windows + Colab) ──────────────────────────────────────────┐
  │ Capture la séquence de kernels CUDA en un « graphe CUDA » réexécutable.  │
  │ N'élimine PAS le coût de calcul, mais supprime l'overhead de lancement   │
  │ CPU de chaque kernel (dispatch Python + driver). Très efficace quand le  │
  │ modèle lance beaucoup de petits kernels (ResNet/EfficientNet).           │
  │ Aucune dépendance Triton. Gain typique : ×1.1 à ×1.5.                     │
  │ Contrainte : shapes d'entrée fixes (OK ici, benchmark à 640×640).        │
  └──────────────────────────────────────────────────────────────────────────┘

Le modèle retourné garde exactement la même API que l'original
(List[Tensor] pour RetinaNet, Tensor[B,C,H,W] pour EfficientDet)
→ utilisable directement avec benchmark_model() et run_map_evaluation().

Le premier appel forward déclenche la compilation JIT (warmup plus long) :
prévoir au moins quelques itérations supplémentaires dans n_warmup.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .capability import default_compile_backend, has_triton

_INDUCTOR_MODES = {"default", "reduce-overhead", "max-autotune"}


def compile_model(
    model: nn.Module,
    backend: str | None = None,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> nn.Module:
    """
    Compile le modèle avec torch.compile, en choisissant le backend adapté.

    Parameters
    ----------
    model     : nn.Module en mode eval — issu de load_model()
    backend   : None  → auto ("inductor" si Triton, sinon "cudagraphs")
                "inductor" | "cudagraphs" | "eager" | ...
    mode      : utilisé uniquement par le backend inductor
                "default" recommandé pour la détection (PAS "reduce-overhead",
                qui active cudagraphs → incompatible avec le NMS à shapes dynamiques).
    fullgraph : exiger un graphe complet sans graph break (False recommandé
                pour la détection — le NMS n'est pas traçable)
    dynamic   : False (défaut) fige les shapes → bien moins de guards symboliques
                → compilation BEAUCOUP plus rapide. Crucial pour EfficientDet/BiFPN,
                où dynamic=True fait exploser le solveur sympy (compilation > 45 min).
                Sûr ici car le benchmark est à résolution fixe 640×640.

    Returns
    -------
    nn.Module avec même API que l'original. Premier forward = compilation JIT.
    """
    if backend is None:
        backend = default_compile_backend()

    model.eval()

    if backend == "inductor" and not has_triton():
        print("[torch.compile] ⚠ backend 'inductor' demandé mais Triton absent.")
        print("  → Bascule automatique sur 'cudagraphs'.")
        backend = "cudagraphs"

    # ── Backend inductor : utilise `mode` (Triton codegen) ────────────────────
    if backend == "inductor":
        if mode not in _INDUCTOR_MODES:
            raise ValueError(f"mode doit être parmi {_INDUCTOR_MODES}, reçu {mode!r}")
        if mode == "reduce-overhead":
            print("[torch.compile] ⚠ mode='reduce-overhead' active cudagraphs →")
            print("  risque d'incompatibilité avec le NMS (shapes dynamiques).")
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        print(f"[torch.compile] backend=inductor  mode={mode!r}  dynamic={dynamic}")
        print("  Fusion de kernels Triton. Premier appel lent (codegen).")

    # ── Backend cudagraphs : capture de graphe CUDA, sans Triton ──────────────
    elif backend == "cudagraphs":
        compiled = torch.compile(model, backend="cudagraphs", fullgraph=fullgraph, dynamic=dynamic)
        print("[torch.compile] backend=cudagraphs")
        print("  ⚠ Capture de graphe à shapes FIXES — incompatible avec le NMS dynamique")
        print("    des détecteurs complets. À réserver au backbone seul.")

    # ── Autres backends (eager, aot_eager, …) ─────────────────────────────────
    else:
        compiled = torch.compile(model, backend=backend, fullgraph=fullgraph, dynamic=dynamic)
        print(f"[torch.compile] backend={backend!r}  dynamic={dynamic}")

    print("  → Inclure le premier appel dans n_warmup (≥ 5 itérations de marge).")
    return compiled


def save_compiled(model: nn.Module, path: str) -> None:
    """
    Sauvegarde le state_dict du modèle sous-jacent.
    torch.compile ne modifie pas les poids — on sérialise le module original.
    Pour recharger : recréer le modèle PyTorch puis rappeler compile_model().
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    underlying = getattr(model, "_orig_mod", model)
    torch.save(underlying.state_dict(), path)
    print(f"[torch.compile] state_dict sauvegardé → {path}")
    print("  Recharger : load_model() puis compile_model().")
