"""
optimizations/tensorrt_fp16.py
═══════════════════════════════
TensorRT FP16 via torch_tensorrt — fusion de kernels + demi-précision.

Dépendances :
  torch >= 2.1
  torch-tensorrt >= 2.1    pip install torch-tensorrt
  tensorrt >= 8.6          pre-installé sur Colab GPU / NGC containers
                           (inclus dans torch-tensorrt sur Colab)

  ⚠ Windows : TensorRT n'est PAS disponible sur Windows via pip standard.
    Utiliser impérativement sur Colab, un container NVIDIA, ou WSL2.

Ce que ça fait :
  torch_tensorrt compile le modèle PyTorch en engine TensorRT optimisé.
  L'approche utilisée ici est le backend torch.compile (ir="torch_compile")
  qui est le plus robuste pour les modèles de détection :
    - Garde le NMS et les ops Python complexes dans PyTorch (pas exporté vers TRT)
    - Envoie les blocs Conv/BN/ReLU/FPN dans des sous-graphes TRT optimisés
    - Retourne un modèle avec EXACTEMENT la même API que l'original
      → fonctionne directement avec benchmark_model() et run_map_evaluation()

  Optimisations appliquées par TRT :
    • Fusion Conv + BN + activation → 1 seul kernel (Conv-BN-ReLU → CBR kernel)
    • Tensor Cores FP16 (2× throughput vs FP32 sur Ampere+)
    • Optimisation du plan d'exécution des kernels CUDA
    • Réutilisation des buffers mémoire intermédiaires

  min_block_size : nombre minimum d'ops consécutives pour former un bloc TRT.
    Trop petit → trop de transitions PyTorch↔TRT (overhead).
    Recommandé : 5 pour modèles de détection (blocs résiduels = 6+ ops).

Sauvegarde :
  TRT avec torch.compile ne peut pas être sérialisé directement (l'engine
  est compilé à la volée au premier appel). Pour sauvegarder un engine TRT
  persistant, utiliser la variante ExportedProgram (voir save_trt_model()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def build_trt_fp16(
    model: nn.Module,
    min_block_size: int = 5,
    workspace_gb: float = 4.0,
    debug: bool = False,
) -> nn.Module:
    """
    Compile le modèle avec TensorRT FP16 via torch.compile.

    Parameters
    ----------
    model          : nn.Module en mode eval — issu de load_model()
    min_block_size : minimum d'ops par bloc TRT (5 recommandé pour détection)
    workspace_gb   : taille max du workspace TRT en Go
    debug          : afficher les logs TRT (verbose)

    Returns
    -------
    nn.Module avec même API que l'original (drop-in replacement).
    Le premier appel forward déclenche la compilation TRT (warmup long).
    """
    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError(
            "torch-tensorrt non installé.\n"
            "  pip install torch-tensorrt\n"
            "  (Colab : !pip install torch-tensorrt)"
        )

    _check_cuda()
    model.eval()

    workspace_bytes = int(workspace_gb * (1 << 30))

    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float16},
            "min_block_size":     min_block_size,
            "workspace_size":     workspace_bytes,
            "debug":              debug,
            "truncate_long_and_double": True,
        },
    )

    print(f"[TRT FP16] Modèle compilé avec torch_tensorrt backend.")
    print(f"  min_block_size={min_block_size}  workspace={workspace_gb:.1f} GB")
    print("  → Premier appel forward déclenche la compilation TRT.")
    print("  → Inclure dans n_warmup (minimum 3 appels supplémentaires).")
    return trt_model


def save_trt_model(
    model: nn.Module,
    save_path: str,
    sample_input: torch.Tensor,
    fp16: bool = True,
    min_block_size: int = 5,
) -> str:
    """
    Compile et sauvegarde un engine TRT persistant via ExportedProgram.
    Plus long à construire, mais peut être rechargé sans recompilation.

    sample_input : Tensor[1, 3, H, W] sur CUDA — définit les shapes figées.

    Returns
    -------
    str : chemin du fichier .ts sauvegardé
    """
    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError("pip install torch-tensorrt")

    _check_cuda()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        exp = torch.export.export(model, (sample_input,))

    precision = {torch.float16} if fp16 else {torch.float32}
    trt_ep = torch_tensorrt.dynamo.compile(
        exp,
        inputs=[sample_input],
        enabled_precisions=precision,
        min_block_size=min_block_size,
    )

    torch_tensorrt.save(trt_ep, save_path, inputs=[sample_input])
    size_mb = Path(save_path).stat().st_size / 1e6
    print(f"[TRT FP16] Engine sauvegardé → {save_path}  ({size_mb:.1f} MB)")
    return save_path


def load_trt_model(path: str) -> nn.Module:
    """
    Charge un engine TRT sauvegardé avec save_trt_model().
    Nécessite torch_tensorrt importé pour désérialiser.
    """
    try:
        import torch_tensorrt  # noqa: F401 — nécessaire pour le désérialiseur
    except ImportError:
        raise ImportError("pip install torch-tensorrt")

    model = torch.export.load(path)
    print(f"[TRT FP16] Engine chargé ← {path}")
    return model


# ── Utilitaire ────────────────────────────────────────────────────────────────

def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponible. TensorRT nécessite un GPU NVIDIA."
        )
    device_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"[TRT FP16] GPU : {device_name}  (Compute Capability {cc_major}.{cc_minor})")
    if cc_major < 7:
        print("  ⚠ Tensor Cores FP16 disponibles à partir de Volta (CC 7.0+).")
        print("  Le modèle compilera mais les gains FP16 seront limités.")
