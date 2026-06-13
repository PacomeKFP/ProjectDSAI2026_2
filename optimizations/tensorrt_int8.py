"""
optimizations/tensorrt_int8.py
═══════════════════════════════
TensorRT INT8 avec calibration PTQ (Post-Training Quantization).

Dépendances :
  torch >= 2.1
  torch-tensorrt >= 2.1    pip install torch-tensorrt
  tensorrt >= 8.6          pre-installé sur Colab GPU
  numpy                    pip install numpy   (inclus)

Ce que ça fait :
  La quantification INT8 réduit les poids et activations de FP32 (32 bits)
  à INT8 (8 bits). Gain : 4× moins de bande passante mémoire, 2-4× plus
  rapide sur les INT8 Tensor Cores (Turing/Ampere/Hopper).

  Protocole PTQ (Post-Training Quantization) :
    1. Calibration : on fait tourner le modèle sur ~200-500 images COCO
       avec un "calibrateur" qui collecte les histogrammes d'activation.
    2. Calcul des facteurs d'échelle : par couche, TRT choisit le facteur
       qui minimise l'erreur de quantification (méthode entropy ou percentile).
    3. Compilation : TRT compile les couches INT8-compatibles en INT8,
       garde en FP16 les couches sensibles (softmax, sigmoid, NMS).

  Impact MAP :
    Typiquement -0.5% à -2% vs FP32 selon le modèle et la taille du
    dataset de calibration. À vérifier systématiquement après optimisation.

  Calibration dataset :
    Utiliser un sous-ensemble du VAL set COCO (pas le train set pour éviter
    le data leakage). 200-500 images suffisent.

  ⚠ INT8 nécessite des données de calibration représentatives.
    Des images non représentatives (mauvaise distribution des valeurs)
    peuvent dégrader la MAP significativement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# ── Dataset de calibration ────────────────────────────────────────────────────

class CalibrationDataset(torch.utils.data.Dataset):
    """
    Dataset minimal pour la calibration INT8.
    Wraps une liste de samples (dicts avec 'path') et les prétraite pour TRT.
    """

    def __init__(
        self,
        samples: list,
        preprocess_fn: Callable,
        collate_fn: Callable,
        device: str = "cuda",
        n_images: int = 500,
    ):
        self.samples      = samples[:n_images]
        self.preprocess   = preprocess_fn
        self.collate      = collate_fn
        self.device       = device

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp = self.preprocess(self.samples[idx])
        # Normaliser au format [3, H, W] float32 (pas de batch dim)
        if inp.dim() == 4:
            inp = inp.squeeze(0)
        return inp


def build_calibration_loader(
    samples: list,
    preprocess_fn: Callable,
    n_images: int = 500,
    batch_size: int = 8,
) -> torch.utils.data.DataLoader:
    """
    Crée un DataLoader pour la calibration INT8.

    Parameters
    ----------
    samples       : liste de dicts {'path', 'image_id', ...}
    preprocess_fn : fonction de prétraitement du modèle (ex: r50.preprocess)
    n_images      : nombre d'images de calibration (200-500 recommandé)
    batch_size    : batch pour la calibration (8-16, ne pas dépasser la VRAM)

    Returns
    -------
    DataLoader prêt pour torch_tensorrt.DataLoaderCalibrator
    """
    ds = CalibrationDataset(samples, preprocess_fn, collate_fn=None, n_images=n_images)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


# ── Compilation INT8 ──────────────────────────────────────────────────────────

def build_trt_int8(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    save_path: Optional[str] = None,
    workspace_gb: float = 4.0,
    calibration_algo: str = "entropy",
    min_block_size: int = 5,
) -> nn.Module:
    """
    Compile le modèle avec TensorRT INT8 via torch_tensorrt + calibration PTQ.

    Parameters
    ----------
    model               : nn.Module en mode eval
    calibration_loader  : DataLoader de calibration (build_calibration_loader())
    save_path           : si fourni, sauvegarde l'engine calibré (.ts)
    workspace_gb        : taille workspace TRT en Go
    calibration_algo    : "entropy" (recommandé) | "minmax" | "percentile"
    min_block_size      : minimum d'ops par bloc TRT

    Returns
    -------
    nn.Module avec même API que l'original.
    """
    try:
        import torch_tensorrt
        from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo
    except ImportError:
        raise ImportError(
            "torch-tensorrt non installé.\n"
            "  pip install torch-tensorrt"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponible. TRT nécessite un GPU NVIDIA.")

    model.eval()
    workspace_bytes = int(workspace_gb * (1 << 30))

    _algo_map = {
        "entropy":    CalibrationAlgo.ENTROPY_CALIBRATION_2,
        "minmax":     CalibrationAlgo.MINMAX_CALIBRATION,
        "percentile": CalibrationAlgo.PERCENTILE_CALIBRATION,
    }
    if calibration_algo not in _algo_map:
        raise ValueError(f"calibration_algo doit être parmi {list(_algo_map)}")

    print(f"[TRT INT8] Calibration en cours ({len(calibration_loader.dataset)} images)...")

    calibrator = DataLoaderCalibrator(
        calibration_loader,
        use_cache=False,
        algo_type=_algo_map[calibration_algo],
        device=torch.device("cuda:0"),
    )

    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.int8},
            "calibrator":         calibrator,
            "min_block_size":     min_block_size,
            "workspace_size":     workspace_bytes,
            "truncate_long_and_double": True,
        },
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        # Forcer la compilation en faisant un premier appel
        sample_batch = next(iter(calibration_loader)).cuda().unsqueeze(0)
        with torch.no_grad():
            _ = trt_model([sample_batch[0]])
        torch_tensorrt.save(trt_model, save_path)
        size_mb = Path(save_path).stat().st_size / 1e6
        print(f"[TRT INT8] Engine sauvegardé → {save_path}  ({size_mb:.1f} MB)")

    print(f"[TRT INT8] Compilation INT8 prête (algo={calibration_algo}).")
    print("  → Vérifier la MAP après optimisation (dégradation typique : -0.5% à -2%).")
    return trt_model


# ── Analyse de la sensibilité par couche ─────────────────────────────────────

def layer_sensitivity_report(
    model_fp16: nn.Module,
    model_int8: nn.Module,
    data: list,
    preprocess_fn: Callable,
    collate_fn: Callable,
    n_samples: int = 50,
    device: str = "cuda",
) -> None:
    """
    Compare les sorties FP16 et INT8 couche par couche pour identifier
    les couches sensibles à la quantification (grande divergence).

    Utiliser ModuleBenchmark pour le timing, cette fonction pour la précision.
    """
    from utils.benchmark import ModuleBenchmark
    import pandas as pd

    print("[INT8 Sensitivity] Comparaison FP16 vs INT8 sur", n_samples, "images")

    errors = []
    for s in data[:n_samples]:
        inp = preprocess_fn(s)
        gpu = collate_fn([inp], device)
        with torch.no_grad():
            out_fp = model_fp16(gpu)
            out_i8 = model_int8(gpu)

        if isinstance(out_fp, list) and isinstance(out_fp[0], dict):
            boxes_fp = out_fp[0].get("boxes", torch.zeros(0, 4))
            boxes_i8 = out_i8[0].get("boxes", torch.zeros(0, 4))
            n_common  = min(len(boxes_fp), len(boxes_i8))
            if n_common > 0:
                err = (boxes_fp[:n_common] - boxes_i8[:n_common]).abs().mean().item()
                errors.append(err)

    if errors:
        mean_err = np.mean(errors)
        print(f"  Erreur moyenne boxes FP16↔INT8 : {mean_err:.4f} pixels")
        if mean_err > 2.0:
            print("  ⚠ Dégradation significative — considérer mixed precision (INT8 backbone seulement)")
        else:
            print("  ✓ Dégradation acceptable")
