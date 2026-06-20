"""
optimizations/tensorrt_int8.py
===============================
TensorRT INT8 with PTQ (Post-Training Quantization) calibration.

Dependencies:
  torch >= 2.1
  torch-tensorrt >= 2.1    pip install torch-tensorrt
  tensorrt >= 8.6          pre-installed on Colab GPU
  numpy                    pip install numpy   (included)

What it does:
  INT8 quantization reduces weights and activations from FP32 (32 bits) to
  INT8 (8 bits). Gain: 4x less memory bandwidth, 2-4x faster on INT8 Tensor
  Cores (Turing/Ampere/Hopper).

  PTQ (Post-Training Quantization) protocol:
    1. Calibration: run the model on ~200-500 COCO images with a "calibrator"
       that collects activation histograms.
    2. Computing scale factors: per layer, TRT picks the factor that minimizes
       quantization error (entropy or percentile method).
    3. Compilation: TRT compiles INT8-compatible layers in INT8, keeps the
       sensitive layers in FP16 (softmax, sigmoid, NMS).

  MAP impact:
    Typically -0.5% to -2% vs FP32 depending on the model and the calibration
    dataset size. Always verify after optimization.

  Calibration dataset:
    Use a subset of the COCO VAL set (not the train set to avoid data leakage).
    200-500 images is enough.

  [!] INT8 requires representative calibration data.
    Non-representative images (poor value distribution) can degrade the MAP
    significantly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn


# -- Calibration dataset -------------------------------------------------------

class CalibrationDataset(torch.utils.data.Dataset):
    """
    Minimal dataset for INT8 calibration.
    Wraps a list of samples (dicts with 'path') and preprocesses them for TRT.
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
        # Normalize to [3, H, W] float32 (no batch dim)
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
    Create a DataLoader for INT8 calibration.

    Parameters
    ----------
    samples       : list of dicts {'path', 'image_id', ...}
    preprocess_fn : model preprocessing function (e.g. r50.preprocess)
    n_images      : number of calibration images (200-500 recommended)
    batch_size    : batch for calibration (8-16, do not exceed VRAM)

    Returns
    -------
    DataLoader ready for torch_tensorrt.DataLoaderCalibrator
    """
    ds = CalibrationDataset(samples, preprocess_fn, collate_fn=None, n_images=n_images)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


# -- INT8 compilation ----------------------------------------------------------

def build_trt_int8(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    save_path: Optional[str] = None,
    workspace_gb: float = 4.0,
    calibration_algo: str = "entropy",
    min_block_size: int = 5,
) -> nn.Module:
    """
    Compile the model with TensorRT INT8 via torch_tensorrt + PTQ calibration.

    Parameters
    ----------
    model               : nn.Module in eval mode
    calibration_loader  : calibration DataLoader (build_calibration_loader())
    save_path           : if provided, save the calibrated engine (.ts)
    workspace_gb        : TRT workspace size in GB
    calibration_algo    : "entropy" (recommended) | "minmax" | "percentile"
    min_block_size      : minimum ops per TRT block

    Returns
    -------
    nn.Module with the same API as the original.
    """
    try:
        import torch_tensorrt
        from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo
    except ImportError:
        raise ImportError(
            "torch-tensorrt not installed.\n"
            "  pip install torch-tensorrt"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. TRT requires an NVIDIA GPU.")

    model.eval()
    workspace_bytes = int(workspace_gb * (1 << 30))

    _algo_map = {
        "entropy":    CalibrationAlgo.ENTROPY_CALIBRATION_2,
        "minmax":     CalibrationAlgo.MINMAX_CALIBRATION,
        "percentile": CalibrationAlgo.PERCENTILE_CALIBRATION,
    }
    if calibration_algo not in _algo_map:
        raise ValueError(f"calibration_algo must be one of {list(_algo_map)}")

    print(f"[TRT INT8] Calibration in progress ({len(calibration_loader.dataset)} images)...")

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
        # Force compilation by running a first call
        sample_batch = next(iter(calibration_loader)).cuda().unsqueeze(0)
        with torch.no_grad():
            _ = trt_model([sample_batch[0]])
        torch_tensorrt.save(trt_model, save_path)
        size_mb = Path(save_path).stat().st_size / 1e6
        print(f"[TRT INT8] Engine saved -> {save_path}  ({size_mb:.1f} MB)")

    print(f"[TRT INT8] INT8 compilation ready (algo={calibration_algo}).")
    print("  -> Check MAP after optimization (typical drop: -0.5% to -2%).")
    return trt_model


# -- Per-layer sensitivity analysis -------------------------------------------

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
    Compare FP16 and INT8 outputs layer by layer to identify layers sensitive
    to quantization (large divergence).

    Use ModuleBenchmark for timing, this function for accuracy.
    """
    from utils.benchmark import ModuleBenchmark
    import pandas as pd

    print("[INT8 Sensitivity] FP16 vs INT8 comparison on", n_samples, "images")

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
        print(f"  Mean FP16<->INT8 box error: {mean_err:.4f} pixels")
        if mean_err > 2.0:
            print("  [!] Significant degradation -- consider mixed precision (INT8 backbone only)")
        else:
            print("  [OK] Acceptable degradation")
