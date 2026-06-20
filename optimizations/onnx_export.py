"""
optimizations/onnx_export.py
=============================
ONNX export -- gateway to TensorRT and ONNX Runtime.

Dependencies:
  onnx >= 1.14          pip install onnx
  onnxruntime-gpu       pip install onnxruntime-gpu
  onnxsim (optional)    pip install onnxsim   <- simplifies the exported graph

What it does:
  torch.onnx.export traces the PyTorch graph and serializes it as ONNX.
  The ONNX graph is then used by:
    - TensorRT (via onnx_parser) to compile an optimized engine
    - ONNX Runtime (ort_inference.py) for cross-platform inference
    - Netron (netron.app) to visualize the architecture

Detection subtleties:
  Detection models have variable-sized outputs (number of detections). Two
  strategies:

  A. export_backbone_only() -- exports only backbone + FPN + head
     (fixed-tensor outputs per FPN level). Post-processing stays in Python.
     [OK] Compatible with ONNX, TRT, ORT without modification.
     Used by ort_inference.py for full inference.

  B. export_full_detection() -- tries to export the full model with NMS.
     [OK] Works for EfficientDet (output [B, N, 6]).
     [X] Hard for torchvision RetinaNet (output List[Dict]).
     Used only if the model exposes a batched API.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


# -- Backbone-only export (RetinaNet + EfficientDet) ---------------------------

class _BackboneFPNWrapper(nn.Module):
    """Extract backbone + FPN from torchvision RetinaNet for a clean ONNX export."""

    def __init__(self, model: nn.Module):
        super().__init__()
        # torchvision RetinaNet: model.backbone includes backbone + FPN
        if hasattr(model, "backbone"):
            self.backbone = model.backbone
            self._mode = "retinanet"
        elif hasattr(model, "model") and hasattr(model.model, "backbone"):
            # effdet wrap: model.model.backbone
            self.backbone = model.model.backbone
            self._mode = "effdet"
        else:
            raise ValueError("Unrecognized architecture -- backbone not found.")

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
    Export only the backbone + FPN to ONNX (fixed outputs -- TRT-ready).

    Parameters
    ----------
    model       : nn.Module (load_model())
    output_path : .onnx path
    image_size  : (H, W) -- fixed input resolution for the export
    opset       : ONNX opset version (17 recommended)
    device      : device used for tracing
    simplify    : simplify the graph with onnxsim (recommended)

    Returns
    -------
    str: path of the .onnx file
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
    print(f"[ONNX] backbone exported -> {output_path}  ({size_mb:.1f} MB)")
    print(f"  outputs: {output_names}")
    return output_path


# -- Full-model export (EfficientDet -- output [B, N, 6]) ----------------------

def export_full_detection(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = (640, 640),
    opset: int = 17,
    device: str = "cuda",
    simplify: bool = False,
) -> str:
    """
    Export the full detection model (NMS included) to ONNX.
    Works for EfficientDet (Tensor[B, N, 6] output).
    For torchvision RetinaNet, use export_backbone_only().

    Returns
    -------
    str: path of the .onnx file
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
    print(f"[ONNX] full model exported -> {output_path}  ({size_mb:.1f} MB)")
    return output_path


# -- Generic alias -------------------------------------------------------------

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = (640, 640),
    backbone_only: bool = True,
    **kwargs,
) -> str:
    """
    Unified entry point. backbone_only=True is the recommended default
    (compatible with every model).
    """
    if backbone_only:
        return export_backbone_only(model, output_path, image_size, **kwargs)
    return export_full_detection(model, output_path, image_size, **kwargs)


# -- Validation ----------------------------------------------------------------

def check_onnx(onnx_path: str) -> bool:
    """Check the structural integrity of the ONNX graph."""
    import onnx
    m = onnx.load(onnx_path)
    try:
        onnx.checker.check_model(m)
        print(f"[ONNX] [OK] Validation OK -- {onnx_path}")
        return True
    except onnx.checker.ValidationError as e:
        print(f"[ONNX] [X] Validation FAILED -- {e}")
        return False


def validate_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    image_size: Tuple[int, int] = (640, 640),
    device: str = "cuda",
    atol: float = 1e-3,
) -> bool:
    """
    Numerically compare PyTorch and ONNX Runtime outputs on backbone_only.
    Useful to detect numerical regressions at export time.
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
        status = "[OK]" if diff <= atol else "[X]"
        print(f"  feat_{i}: max_diff={diff:.2e}  {status}")
        if diff > atol:
            ok = False

    print(f"[ONNX] Numerical validation: {'OK' if ok else 'FAILED'}")
    return ok


# -- Private utility ----------------------------------------------------------

def _simplify(path: str) -> None:
    try:
        import onnxsim, onnx
        m, ok = onnxsim.simplify(onnx.load(path))
        if ok:
            onnx.save(m, path)
            print("[ONNX] Graph simplified with onnxsim [OK]")
        else:
            print("[ONNX] onnxsim could not simplify the graph")
    except ImportError:
        print("[ONNX] onnxsim not installed -- skipping")
