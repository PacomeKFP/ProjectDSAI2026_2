"""
optimizations/fp16_half.py
===========================
Half-precision (FP16) inference via torch.autocast -- without compilation.

Dependencies:
  torch + CUDA GPU with Tensor Cores (Volta/Turing/Ampere/Ada/Blackwell)
  No external dependencies -- works on Windows AND Colab.

What it does:
  Two FP16 approaches:

  A. AUTOCAST (recommended, robust):
     Weights stay in FP32, but matrix operations (conv, matmul) are computed
     in FP16 on Tensor Cores. PyTorch automatically chooses which ops go to
     FP16 (conv, linear) and which stay in FP32 (batchnorm stats, softmax,
     loss) to preserve numerical stability.
     -> No weight modification, no calibration, MAP nearly identical.

  B. Pure HALF (model.half()):
     Converts ALL weights and activations to FP16. Even faster but with a
     risk of instability (overflow in some layers). Less robust.
     Not used by default here.

Why this matters for the report:
  This is the SAME idea as TensorRT FP16, but isolated: we measure the pure
  gain of FP16 (Tensor Cores) WITHOUT kernel fusion. By comparing:
    baseline (FP32)  ->  FP16 autocast  ->  TensorRT FP16
  we decompose the TRT speedup into two contributions:
    * FP16 gain        = (baseline -> autocast)
    * fusion+graph gain = (autocast -> TRT)
  This is exactly the "where do speed-ups come from" decomposition we want.

Expected gain:
  x1.3 to x2.0 on convolution-heavy models (ResNet, EfficientNet), depending
  on the share of time spent in conv vs CPU post-processing.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AutocastModel(nn.Module):
    """
    Wrapper that runs the model's forward under torch.autocast(FP16).

    Preserves the original model's API exactly:
      - same forward signature (List[Tensor] or Tensor)
      - same output format
    -> drop-in for benchmark_model() and run_map_evaluation().

    Weights are NOT modified (stay in FP32 in memory): autocast performs the
    conversion on the fly during the forward.
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
    Wrap the model for FP16 inference via autocast.

    Parameters
    ----------
    model : nn.Module in eval mode -- from load_model()
    dtype : torch.float16 (default) or torch.bfloat16
            bfloat16 = more numerically stable, available on Ampere+ (CC 8.0+)

    Returns
    -------
    AutocastModel -- same API as the original.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 autocast requires a CUDA GPU.")

    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major < 7:
        print("  [!] FP16 Tensor Cores available from Volta (CC 7.0+).")
        print("    The wrapper will work but without hardware FP16 acceleration.")

    model.eval()
    print(f"[FP16] Autocast wrapper created (dtype={dtype}).")
    print("  Weights kept in FP32, conv/matmul computed in FP16 on Tensor Cores.")
    return AutocastModel(model, dtype=dtype).eval()


def to_fp16_half(model: nn.Module) -> nn.Module:
    """
    Convert the model entirely to FP16 (model.half()).
    More aggressive than autocast but can be numerically unstable.
    Use only if autocast does not give enough gain.

    [!] The input must also be converted to .half() before the forward.
       Does NOT work directly with the project's FP32 preprocess_fn.
       Prefer to_fp16_autocast() for compatibility.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("FP16 half requires a CUDA GPU.")
    model.eval().half()
    print("[FP16] Model converted with half() -- remember to pass .half() inputs.")
    return model
