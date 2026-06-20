"""
optimizations/ort_inference.py
===============================
ONNX Runtime (ORT) with the CUDA Execution Provider.

Dependencies:
  onnxruntime-gpu >= 1.16    pip install onnxruntime-gpu
  onnx >= 1.14               pip install onnx

What it does:
  ONNX Runtime is a cross-platform inference runtime that optimizes ONNX
  graphs without requiring TensorRT. It uses the "CUDA Execution Provider"
  (EP) to run operations on the GPU via cuDNN/cuBLAS.

  Advantages vs TensorRT:
    [OK] Simpler (no long engine compilation)
    [OK] Cross-platform (Windows included)
    [OK] Supports dynamic shapes natively
    [OK] Nice integration with standard ML pipelines

  Disadvantages vs TensorRT:
    [X] Less kernel fusion (no CBR kernel, limited fusion)
    [X] FP16 not as aggressive
    [X] Typical gains: 20-40% vs 60-100% for TRT FP16

  Usage in this project:
    ORT operates on the backbone_only export (onnx_export.py). The detection
    head (NMS, box decoding) stays in PyTorch. For full MAP evaluation, use
    the original PyTorch model or TRT (which preserves the full API).

ORTModel architecture:
  ORTModel wraps an ORT session and exposes an API compatible with
  benchmark_model() to measure backbone throughput.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class ORTModel:
    """
    ONNX Runtime wrapper for a detection backbone.

    Exposes __call__(images) -> List[Tensor] (feature maps per FPN level).
    Compatible with benchmark_model() for backbone throughput.

    For full MAP, add the detection head and NMS in PyTorch on top of the
    ORT features -- see ORTDetectionModel.
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

        print(f"[ORT] Session created -- active provider: {self._active_provider}")
        print(f"  inputs : {[i.name for i in self.session.get_inputs()]}")
        print(f"  outputs: {[o.name for o in self.session.get_outputs()]}")

    def __call__(self, images) -> List[torch.Tensor]:
        """
        images: Tensor[B, 3, H, W] or List[Tensor[3, H, W]]
        Returns a list of feature maps (one per FPN level).
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
        """Alias for compatibility with benchmark_model (which calls model(gpu))."""
        return self(images)


class ORTDetectionModel:
    """
    Full detection model with an ORT backbone + a PyTorch head.

    Architecture:
      images -> [ORT backbone] -> features -> [PyTorch head + NMS] -> List[Dict]

    Useful to compare ORT vs PyTorch vs TRT on full MAP.
    Requires access to the model's internal attributes (head, anchor_generator, etc.)
    -> implemented only for torchvision RetinaNet.
    """

    def __init__(self, pytorch_model: nn.Module, ort_model: ORTModel):
        self.head             = pytorch_model.head
        self.anchor_generator = pytorch_model.anchor_generator
        self.postprocess      = pytorch_model.postprocess_detections
        self.transform        = pytorch_model.transform
        self.ort              = ort_model
        self.device           = ort_model.device

    def __call__(self, images: List[torch.Tensor]) -> List[dict]:
        # Transform (torchvision normalization)
        imgs, targets = self.transform(images, None)

        # Backbone via ORT
        features_list = self.ort(imgs.tensors)
        # Rebuild the OrderedDict expected by the RetinaNet head
        from collections import OrderedDict
        features = OrderedDict(
            {str(i): f for i, f in enumerate(features_list)}
        )

        # PyTorch head (classification + regression)
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


# -- Build function ------------------------------------------------------------

def build_ort_model(
    onnx_path: str,
    device: str = "cuda",
) -> ORTModel:
    """
    Build an ONNX Runtime session from an .onnx file.

    Parameters
    ----------
    onnx_path : path to the .onnx exported by onnx_export.export_backbone_only()
    device    : "cuda" or "cpu"

    Returns
    -------
    ORTModel -- callable compatible with benchmark_model() for the backbone.
    """
    if not Path(onnx_path).exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ImportError(
            "onnxruntime-gpu not installed.\n"
            "  pip install onnxruntime-gpu"
        )

    return ORTModel(onnx_path, device=device)


def build_ort_detection_model(
    pytorch_model: nn.Module,
    onnx_path: str,
    device: str = "cuda",
) -> ORTDetectionModel:
    """
    Build a full ORT + PyTorch head detection model (RetinaNet only).

    Parameters
    ----------
    pytorch_model : original PyTorch model (for the head + NMS)
    onnx_path     : backbone .onnx exported with export_backbone_only()
    device        : "cuda" or "cpu"

    Returns
    -------
    ORTDetectionModel -- callable returning List[Dict] (same format as PyTorch).
    """
    ort_backbone = build_ort_model(onnx_path, device)
    return ORTDetectionModel(pytorch_model, ort_backbone)


# -- Available providers analysis ---------------------------------------------

def list_ort_providers() -> List[str]:
    """List the Execution Providers available in this ORT install."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("[ORT] Available providers:")
        for p in providers:
            print(f"  {p}")
        return providers
    except ImportError:
        print("[ORT] onnxruntime not installed -- pip install onnxruntime-gpu")
        return []
