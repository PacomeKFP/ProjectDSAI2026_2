"""
optimizations/capability.py
============================
Centralized detection of environment capabilities.

Each optimization technique has different dependencies and is not available
everywhere. This module centralizes detection so that the notebook and the
optimization modules share a single source of truth.

Typical availability matrix:

  Technique              Windows local   Colab (Linux+GPU)   Depends on
  ---------------------  --------------  ------------------  ------------------
  FP16 autocast          [OK]               [OK]                   CUDA (Tensor Cores)
  TorchScript            [OK]               [OK]                   torch (built-in)
  torch.compile cudagraphs [OK]             [OK]                   CUDA
  torch.compile inductor  [X] (no Triton)  [OK]                   triton
  ONNX export            [OK]               [OK]                   onnx
  ONNX Runtime           [OK]               [OK]                   onnxruntime-gpu
  TensorRT FP16/INT8     [X]               [OK]                   torch-tensorrt

Note: torch.compile has two relevant backends here:
  - inductor  : generates fused Triton kernels (the fastest, but Triton is
                not packaged for Windows by default)
  - cudagraphs: captures the CUDA graph to remove the kernel-launch overhead
                (no codegen -> no Triton dependency)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Dict

import torch


# -- Encoding-tolerant status marks --------------------------------------------
# Jupyter is UTF-8 ([OK]/[X] OK). Some Windows terminals (cp1252) cannot encode
# these characters -> we detect and fall back to ASCII marks.

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "[OK][X]".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


_UNI = _supports_unicode()
_OK  = "[OK]" if _UNI else "[OK]"
_NO  = "[X]" if _UNI else "[--]"


def has_triton() -> bool:
    """True if Triton is installed and usable (inductor backend of torch.compile)."""
    try:
        import triton  # noqa: F401
        # torch also exposes a check that verifies GPU compatibility
        from torch.utils._triton import has_triton as _torch_has_triton
        return bool(_torch_has_triton())
    except Exception:
        try:
            import triton  # noqa: F401
            return True
        except Exception:
            return False


def has_torch_tensorrt() -> bool:
    """True if torch-tensorrt is importable AND a CUDA GPU is present."""
    if not torch.cuda.is_available():
        return False
    try:
        import torch_tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def has_onnx() -> bool:
    try:
        import onnx  # noqa: F401
        return True
    except Exception:
        return False


def has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def default_compile_backend() -> str:
    """
    Recommended torch.compile backend depending on the environment.
      - "inductor"   if Triton is available (full Triton fusion)
      - "cudagraphs" otherwise, if CUDA is available (graph capture, no Triton)
      - "eager"      as a last resort (no real acceleration)
    """
    if has_triton():
        return "inductor"
    if torch.cuda.is_available():
        return "cudagraphs"
    return "eager"


@dataclass
class Capabilities:
    """Snapshot of the current environment's capabilities."""
    platform:        str
    is_colab:        bool
    cuda:            bool
    gpu_name:        str
    torch_version:   str
    triton:          bool
    torch_tensorrt:  bool
    onnx:            bool
    onnxruntime:     bool
    compile_backend: str
    flags: Dict[str, bool] = field(default_factory=dict)

    def matrix(self) -> str:
        """Readable table of available techniques."""
        def mark(ok): return f"{_OK} available" if ok else f"{_NO} unavailable"
        inductor_label = "  - inductor (Triton)" if not _UNI else "  + inductor (Triton)"
        rows = [
            ("FP16 autocast",          self.cuda),
            ("TorchScript",            True),
            (f"torch.compile ({self.compile_backend})", self.cuda),
            (inductor_label,           self.triton),
            ("ONNX export",            self.onnx),
            ("ONNX Runtime",           self.onnxruntime),
            ("TensorRT FP16/INT8",     self.torch_tensorrt),
        ]
        width = max(len(r[0]) for r in rows)
        lines = [f"  {name:<{width}}  {mark(ok)}" for name, ok in rows]
        return "\n".join(lines)


def detect() -> Capabilities:
    """Build a snapshot of the current environment's capabilities."""
    cuda = torch.cuda.is_available()
    caps = Capabilities(
        platform        = platform.system(),
        is_colab        = "google.colab" in sys.modules,
        cuda            = cuda,
        gpu_name        = torch.cuda.get_device_name(0) if cuda else "--",
        torch_version   = torch.__version__,
        triton          = has_triton(),
        torch_tensorrt  = has_torch_tensorrt(),
        onnx            = has_onnx(),
        onnxruntime     = has_onnxruntime(),
        compile_backend = default_compile_backend(),
    )
    caps.flags = {
        "fp16":          caps.cuda,
        "torchscript":   True,
        "torch_compile": caps.cuda,
        "inductor":      caps.triton,
        "onnx":          caps.onnx,
        "ort":           caps.onnxruntime,
        "tensorrt":      caps.torch_tensorrt,
    }
    return caps


def print_report() -> Capabilities:
    """Print a full report and return the Capabilities object."""
    caps = detect()
    print(f"Platform     : {caps.platform} ({'Colab' if caps.is_colab else 'local'})")
    print(f"PyTorch      : {caps.torch_version}")
    print(f"CUDA         : {_OK + ' ' + caps.gpu_name if caps.cuda else _NO + ' (CPU)'}")
    print(f"Recommended compile backend: {caps.compile_backend}")
    print()
    print("Available optimization techniques:")
    print(caps.matrix())
    return caps
