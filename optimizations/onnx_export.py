"""
optimizations/onnx_export.py
═════════════════════════════
Export ONNX — porte d'entrée vers TensorRT et ONNX Runtime.

Dépendances :
  onnx >= 1.14          pip install onnx
  onnxruntime-gpu       pip install onnxruntime-gpu
  onnxsim (optionnel)   pip install onnxsim   ← simplifie le graphe exporté

Ce que ça fait :
  torch.onnx.export trace le graphe PyTorch et le sérialise au format ONNX.
  Le graphe ONNX est ensuite utilisé par :
    - TensorRT (via onnx_parser) pour compiler un engine optimisé
    - ONNX Runtime (ort_inference.py) pour une inférence cross-platform
    - Netron (netron.app) pour visualiser l'architecture

Subtilités détection :
  Les modèles de détection ont des sorties à taille variable (nombre de
  détections). Deux stratégies :

  A. export_backbone_only() — exporte uniquement backbone + FPN + tête
     (sorties tenseurs fixes par FPN level). Post-processing en Python.
     ✓ Compatible ONNX, TRT, ORT sans modifications.
     Utilisé par ort_inference.py pour l'inférence complète.

  B. export_full_detection() — tente d'exporter le modèle complet avec NMS.
     ✓ Fonctionne pour EfficientDet (sortie [B, N, 6]).
     ✗ Difficile pour RetinaNet torchvision (sortie List[Dict]).
     Utilisé seulement si le modèle expose une API batched.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# ── Export backbone uniquement (RetinaNet + EfficientDet) ─────────────────────

class _BackboneFPNWrapper(nn.Module):
    """Extrait backbone + FPN de RetinaNet torchvision pour export ONNX propre."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # RetinaNet torchvision : model.backbone inclut backbone + FPN
        if hasattr(model, "backbone"):
            self.backbone = model.backbone
            self._mode = "retinanet"
        elif hasattr(model, "model") and hasattr(model.model, "backbone"):
            # effdet wrap : model.model.backbone
            self.backbone = model.model.backbone
            self._mode = "effdet"
        else:
            raise ValueError("Architecture non reconnue — backbone introuvable.")

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        if isinstance(feats, dict):
            return tuple(feats.values())
        if isinstance(feats, (list, tuple)):
            return tuple(feats)
        return (feats,)


def export_backbone_only(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
    device: str = "cuda",
    simplify: bool = False,
) -> str:
    """
    Exporte uniquement le backbone + FPN vers ONNX (sorties fixes — TRT-ready).

    Parameters
    ----------
    model       : nn.Module (load_model())
    output_path : chemin .onnx
    image_size  : (H, W) — résolution d'entrée fixe pour l'export
    opset       : version opset ONNX (17 recommandé)
    device      : device sur lequel faire le trace
    simplify    : simplifier le graphe avec onnxsim (recommandé)

    Returns
    -------
    str : chemin du fichier .onnx
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    wrapper = _BackboneFPNWrapper(model).to(device).eval()
    h, w    = image_size
    dummy   = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        out = wrapper(dummy)
    output_names = [f"feat_{i}" for i in range(len(out))]

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            **{n: {0: "batch"} for n in output_names},
        },
    )

    if simplify:
        _simplify(output_path)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[ONNX] backbone exporté -> {output_path}  ({size_mb:.1f} MB)")
    print(f"  outputs : {output_names}")
    return output_path


# ── Export modèle complet (EfficientDet — sortie [B, N, 6]) ──────────────────

def export_full_detection(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
    device: str = "cuda",
    simplify: bool = False,
) -> str:
    """
    Exporte le modèle complet de détection (NMS inclus) vers ONNX.
    Fonctionne pour EfficientDet (sortie Tensor [B, N, 6]).
    Pour RetinaNet torchvision, utiliser export_backbone_only().

    Returns
    -------
    str : chemin du fichier .onnx
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    h, w  = image_size
    dummy = torch.zeros(1, 3, h, w, device=device)

    with torch.no_grad():
        out = model(dummy)

    if isinstance(out, (list, tuple)):
        n_out = len(out) if not isinstance(out[0], dict) else 1
    else:
        n_out = 1
    output_names = [f"output_{i}" for i in range(n_out)]

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
        },
    )

    if simplify:
        _simplify(output_path)

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"[ONNX] modèle complet exporté -> {output_path}  ({size_mb:.1f} MB)")
    return output_path


# ── Alias générique ───────────────────────────────────────────────────────────

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = (640, 640),
    backbone_only: bool = True,
    **kwargs,
) -> str:
    """
    Point d'entrée unifié. backbone_only=True est le défaut recommandé
    (compatible avec tous les modèles).
    """
    if backbone_only:
        return export_backbone_only(model, output_path, image_size, **kwargs)
    return export_full_detection(model, output_path, image_size, **kwargs)


# ── Validation ────────────────────────────────────────────────────────────────

def check_onnx(onnx_path: str) -> bool:
    """Vérifie l'intégrité structurelle du graphe ONNX."""
    import onnx
    m = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(m)
        print(f"[ONNX] [OK] Validation OK — {onnx_path}")
        return True
    except onnx.checker.ValidationError as e:
        print(f"[ONNX] [X] Validation FAILED — {e}")
        return False


def validate_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    image_size: Tuple[int, int] = (640, 640),
    device: str = "cuda",
    atol: float = 1e-3,
) -> bool:
    """
    Compare numériquement les sorties PyTorch et ONNX Runtime sur backbone_only.
    Utile pour détecter des régressions numériques à l'export.
    """
    import onnxruntime as ort
    import numpy as np

    h, w    = image_size
    dummy   = torch.zeros(1, 3, h, w, device=device)
    wrapper = _BackboneFPNWrapper(pytorch_model).to(device).eval()

    with torch.no_grad():
        pt_outs = wrapper(dummy)
    pt_outs = [o.cpu().numpy() for o in pt_outs]

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess      = ort.InferenceSession(onnx_path, providers=providers)
    inp_name  = sess.get_inputs()[0].name
    ort_outs  = sess.run(None, {inp_name: dummy.cpu().numpy()})

    ok = True
    for i, (pt, ort_o) in enumerate(zip(pt_outs, ort_outs)):
        diff = np.abs(pt - ort_o).max()
        status = "✓" if diff <= atol else "✗"
        print(f"  feat_{i}: max_diff={diff:.2e}  {status}")
        if diff > atol:
            ok = False

    print(f"[ONNX] Validation numérique : {'OK' if ok else 'FAILED'}")
    return ok


# ── Utilitaire privé ─────────────────────────────────────────────────────────

def _simplify(path: str) -> None:
    try:
        import onnxsim, onnx
        m, ok = onnxsim.simplify(onnx.load(path))
        if ok:
            onnx.save(m, path)
            print("[ONNX] Graphe simplifié avec onnxsim [OK]")
        else:
            print("[ONNX] onnxsim n'a pas pu simplifier le graphe")
    except ImportError:
        print("[ONNX] onnxsim non installé — skip")
