"""
optimizations/fp16_half.py
═══════════════════════════
Inférence en demi-précision (FP16) via torch.autocast — sans compilation.

Dépendances :
  torch + GPU CUDA avec Tensor Cores (Volta/Turing/Ampere/Ada/Blackwell)
  Aucune dépendance externe — fonctionne sur Windows ET Colab.

Ce que ça fait :
  Deux approches de FP16 :

  A. AUTOCAST (recommandé, robuste) :
     Les poids restent en FP32, mais les opérations matricielles (conv, matmul)
     sont calculées en FP16 sur les Tensor Cores. PyTorch choisit automatiquement
     quelles ops passent en FP16 (conv, linear) et lesquelles restent en FP32
     (batchnorm stats, softmax, loss) pour préserver la stabilité numérique.
     → Aucune modification des poids, aucune calibration, MAP quasi-identique.

  B. HALF pur (model.half()) :
     Convertit TOUS les poids et activations en FP16. Plus rapide encore mais
     risque d'instabilité (overflow dans certaines couches). Moins robuste.
     Non utilisé par défaut ici.

Pourquoi c'est important pour le rapport :
  C'est la MÊME idée que TensorRT FP16, mais isolée : on mesure le gain pur du
  FP16 (Tensor Cores) SANS la fusion de kernels. En comparant :
    baseline (FP32)  →  FP16 autocast  →  TensorRT FP16
  on décompose le speedup TRT en deux contributions :
    • gain FP16        = (baseline → autocast)
    • gain fusion+graph = (autocast → TRT)
  C'est exactement la décomposition « d'où viennent les speed-ups » demandée.

Gain attendu :
  ×1.3 à ×2.0 sur les modèles riches en convolutions (ResNet, EfficientNet),
  selon la proportion de temps passé dans les conv vs le post-processing CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AutocastModel(nn.Module):
    """
    Wrapper qui exécute le forward du modèle sous torch.autocast(FP16).

    Préserve exactement l'API du modèle original :
      - même signature forward (List[Tensor] ou Tensor)
      - même format de sortie
    → drop-in pour benchmark_model() et run_map_evaluation().

    Les poids ne sont PAS modifiés (restent en FP32 en mémoire) : autocast
    fait la conversion à la volée pendant le forward.
    """

    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        with torch.autocast(device_type="cuda", dtype=self.dtype):
            return self.model(*args, **kwargs)


def to_fp16_autocast(model: nn.Module, dtype: torch.dtype = torch.float16) -> nn.Module:
    """
    Enveloppe le modèle pour une inférence FP16 via autocast.

    Parameters
    ----------
    model : nn.Module en mode eval — issu de load_model()
    dtype : torch.float16 (défaut) ou torch.bfloat16
            bfloat16 = plus stable numériquement, dispo Ampere+ (CC 8.0+)

    Returns
    -------
    AutocastModel — même API que l'original.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 autocast nécessite un GPU CUDA.")

    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major < 7:
        print("  [!] Tensor Cores FP16 disponibles à partir de Volta (CC 7.0+).")
        print("    Le wrapper fonctionnera mais sans accélération matérielle FP16.")

    model.eval()
    print(f"[FP16] Wrapper autocast créé (dtype={dtype}).")
    print("  Poids conservés en FP32, calcul conv/matmul en FP16 sur Tensor Cores.")
    return AutocastModel(model, dtype=dtype).eval()


def to_fp16_half(model: nn.Module) -> nn.Module:
    """
    Convertit le modèle entièrement en FP16 (model.half()).
    Plus agressif que autocast, mais peut être numériquement instable.
    À utiliser seulement si autocast ne donne pas assez de gain.

    ⚠ L'input doit aussi être converti en .half() avant le forward.
       Ne fonctionne PAS directement avec les preprocess_fn FP32 du projet.
       Préférer to_fp16_autocast() pour la compatibilité.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 half nécessite un GPU CUDA.")
    model.eval().half()
    print("[FP16] Modèle converti en half() — penser à passer des inputs .half().")
    return model
