"""
optimizations/ort_inference.py
═══════════════════════════════
ONNX Runtime (ORT) avec CUDA Execution Provider.

Dépendances :
  onnxruntime-gpu >= 1.16    pip install onnxruntime-gpu
  onnx >= 1.14               pip install onnx

Ce que ça fait :
  ONNX Runtime est un runtime d'inférence cross-platform qui optimise les
  graphes ONNX sans nécessiter TensorRT. Il utilise le "CUDA Execution Provider"
  (EP) pour exécuter les opérations sur GPU via cuDNN/cuBLAS.

  Avantages vs TensorRT :
    ✓ Plus simple (pas de compilation d'engine longue)
    ✓ Cross-platform (Windows inclus)
    ✓ Supporte les formes dynamiques nativement
    ✓ Bonne intégration avec les pipelines ML standard

  Inconvénients vs TensorRT :
    ✗ Moins de fusion de kernels (pas de CBR kernel, fusion limitée)
    ✗ Pas de FP16 aussi agressif
    ✗ Gains typiques 20-40% vs 60-100% pour TRT FP16

  Utilisation dans ce projet :
    ORT opère sur l'export backbone_only (onnx_export.py). La tête de
    détection (NMS, décodage des boîtes) reste en PyTorch.
    Pour l'évaluation MAP complète, utiliser le modèle PyTorch original
    ou TRT (qui préserve l'API complète).

Architecture de ORTModel :
  ORTModel wraps une session ORT et expose une API compatible avec
  benchmark_model() pour mesurer le throughput backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ORTModel:
    """
    Wrapper ONNX Runtime pour backbone de détection.

    Expose __call__(images) → List[Tensor] (feature maps par FPN level).
    Compatible avec benchmark_model() pour le throughput backbone.

    Pour la MAP complète, il faut ajouter la tête de détection et le NMS
    en PyTorch par-dessus les features ORT — voir ORTDetectionModel.
    """

    def __init__(self, onnx_path: str, device: str = "cuda"):
        import onnxruntime as ort

        providers = (
            [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads      = 1

        self.session  = ort.InferenceSession(onnx_path, sess_options=opts, providers=providers)
        self.inp_name = self.session.get_inputs()[0].name
        self.device   = device
        self._active_provider = self.session.get_providers()[0]

        print(f"[ORT] Session créée — provider actif : {self._active_provider}")
        print(f"  inputs  : {[i.name for i in self.session.get_inputs()]}")
        print(f"  outputs : {[o.name for o in self.session.get_outputs()]}")

    def __call__(self, images) -> List[torch.Tensor]:
        """
        images : Tensor[B, 3, H, W] ou List[Tensor[3, H, W]]
        Retourne une liste de feature maps (une par FPN level).
        """
        if isinstance(images, (list, tuple)):
            x = torch.stack(images)
        else:
            x = images

        x_np   = x.cpu().numpy()
        ort_out = self.session.run(None, {self.inp_name: x_np})
        return [torch.from_numpy(o).to(self.device) for o in ort_out]

    def eval(self):
        return self

    def benchmark_forward(self, images) -> List[torch.Tensor]:
        """Alias pour compatibilité avec benchmark_model (qui appelle model(gpu))."""
        return self(images)


class ORTDetectionModel:
    """
    Modèle de détection complet avec ORT backbone + tête PyTorch.

    Architecture :
      images → [ORT backbone] → features → [PyTorch head + NMS] → List[Dict]

    Utile pour comparer ORT vs PyTorch vs TRT sur MAP complète.
    Nécessite accès aux attributs internes du modèle (head, anchor_generator, etc.)
    → implémenté seulement pour RetinaNet torchvision.
    """

    def __init__(self, pytorch_model: nn.Module, ort_model: ORTModel):
        self.head             = pytorch_model.head
        self.anchor_generator = pytorch_model.anchor_generator
        self.postprocess      = pytorch_model.postprocess_detections
        self.transform        = pytorch_model.transform
        self.ort              = ort_model
        self.device           = ort_model.device

    def __call__(self, images: List[torch.Tensor]) -> List[dict]:
        # Transform (normalisation torchvision)
        imgs, targets = self.transform(images, None)

        # Backbone via ORT
        features_list = self.ort(imgs.tensors)
        # Reconstituer l'OrderedDict attendu par le head RetinaNet
        from collections import OrderedDict
        features = OrderedDict(
            {str(i): f for i, f in enumerate(features_list)}
        )

        # Head PyTorch (classification + regression)
        with torch.no_grad():
            head_out = self.head(features)
            anchors  = self.anchor_generator(imgs, list(features.values()))
            detections = self.postprocess(
                head_out, anchors,
                imgs.image_sizes, [img.shape[-2:] for img in images],
            )
        return detections

    def eval(self):
        self.head.eval()
        return self


# ── Fonction de construction ──────────────────────────────────────────────────

def build_ort_model(
    onnx_path: str,
    device: str = "cuda",
) -> ORTModel:
    """
    Construit une session ONNX Runtime à partir d'un fichier .onnx.

    Parameters
    ----------
    onnx_path : chemin vers le .onnx exporté par onnx_export.export_backbone_only()
    device    : "cuda" ou "cpu"

    Returns
    -------
    ORTModel — callable compatible avec benchmark_model() pour le backbone.
    """
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"Fichier ONNX non trouvé : {onnx_path}")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ImportError(
            "onnxruntime-gpu non installé.\n"
            "  pip install onnxruntime-gpu"
        )

    return ORTModel(onnx_path, device=device)


def build_ort_detection_model(
    pytorch_model: nn.Module,
    onnx_path: str,
    device: str = "cuda",
) -> ORTDetectionModel:
    """
    Construit un modèle de détection complet ORT + tête PyTorch (RetinaNet uniquement).

    Parameters
    ----------
    pytorch_model : modèle PyTorch original (pour la tête + NMS)
    onnx_path     : .onnx backbone exporté avec export_backbone_only()
    device        : "cuda" ou "cpu"

    Returns
    -------
    ORTDetectionModel — callable qui retourne List[Dict] (même format que PyTorch).
    """
    ort_backbone = build_ort_model(onnx_path, device)
    return ORTDetectionModel(pytorch_model, ort_backbone)


# ── Analyse des providers disponibles ────────────────────────────────────────

def list_ort_providers() -> List[str]:
    """Liste les Execution Providers disponibles dans cette installation ORT."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("[ORT] Providers disponibles :")
        for p in providers:
            print(f"  {p}")
        return providers
    except ImportError:
        print("[ORT] onnxruntime non installé — pip install onnxruntime-gpu")
        return []
