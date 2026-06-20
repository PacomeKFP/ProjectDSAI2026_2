"""
optimizations/torchscript.py
=============================
Graph compilation via TorchScript -- works on Windows AND Colab.

Dependencies:
  torch only (built-in, no external dependency, no Triton).

What it does:
  TorchScript turns a dynamic Python nn.Module into a static, serializable,
  optimizable graph. This is the "graph compilation" path available
  everywhere (unlike torch.compile/inductor which requires Triton).

  Three steps:
    1. GRAPH CAPTURE
       * torch.jit.script  : analyzes the Python source code -> graph (handles
         control flow: if/for/while). Preferred path for torchvision models
         (written to be scriptable).
       * torch.jit.trace   : runs the model on a sample input and records the
         ops encountered. Simpler but does not capture data-dependent control
         flow (untaken branches are lost).

    2. FREEZE (torch.jit.freeze)
       Inlines weights as constants, removes useless attributes, and enables
       inter-procedural optimizations (constant folding).

    3. OPTIMIZE_FOR_INFERENCE (torch.jit.optimize_for_inference)
       Applies inference-specific passes:
         * Conv+BatchNorm fusion (BN stats are frozen -> folded into the conv)
         * Conv+ReLU fusion via the NNC fuser
         * dead-code elimination, peephole optimizations

  Difference vs TensorRT:
    TorchScript fuses at the PyTorch graph level (NNC/nvFuser) but stays in
    the PyTorch runtime. TRT recompiles into native CUDA kernels.
    TorchScript yields more modest gains (x1.1-1.4) but with no dependency
    and works everywhere.

  Conv+BN fusion:
    This is THE main gain of optimize_for_inference. In inference, BatchNorm
    becomes a constant affine transform (y = gamma*(x-mu)/sigma + beta) that can be
    folded into the preceding conv's weights -> one layer instead of two.
    Visible in the report: BatchNorm2d modules "disappear" from the graph.

Saving:
  The TorchScript graph is serializable (.ts) and reloadable without the
  original Python source code -> portable across machines (unlike TRT engines).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def optimize_with_torchscript(
    model: nn.Module,
    example_input=None,
    prefer: str = "script",
    freeze: bool = True,
    optimize: bool = True,
) -> nn.Module:
    """
    Compile the model into TorchScript + freeze + optimize_for_inference.

    Parameters
    ----------
    model         : nn.Module in eval mode
    example_input : example input for tracing (required if prefer="trace"
                    or if scripting fails). Format expected by the model:
                    List[Tensor] for RetinaNet, Tensor[B,C,H,W] for EfficientDet.
    prefer        : "script" (default, robust for torchvision) or "trace"
    freeze        : apply torch.jit.freeze (recommended)
    optimize      : apply optimize_for_inference (Conv+BN fusion, etc.)

    Returns
    -------
    TorchScript nn.Module -- same API as the original.
    If compilation fails, raises RuntimeError (fail-loud).
    """
    model.eval()
    ts_model = None

    # -- 1. Graph capture ------------------------------------------------------
    if prefer == "script":
        try:
            ts_model = torch.jit.script(model)
            print("[TorchScript] [OK] Graph captured via torch.jit.script")
        except Exception as e:
            print(f"[TorchScript] script failed ({type(e).__name__}) -- falling back to trace")
            ts_model = _try_trace(model, example_input)
    else:
        ts_model = _try_trace(model, example_input)

    if ts_model is None:
        # FAIL-LOUD: do not silently fall back to eager -- otherwise the runner
        # would report a fake gain (the bench would measure the non-compiled
        # model while believing it measures TorchScript). The runner's
        # try/except will mark this FAILED.
        raise RuntimeError(
            "TorchScript: neither script nor trace succeeded -- compilation impossible."
        )

    # -- 2. Freeze -------------------------------------------------------------
    if freeze:
        try:
            ts_model = torch.jit.freeze(ts_model)
            print("[TorchScript] [OK] Freeze applied (weights inlined, constant folding)")
        except Exception as e:
            print(f"[TorchScript] freeze skipped ({type(e).__name__})")

    # -- 3. Optimize for inference ---------------------------------------------
    if optimize:
        try:
            ts_model = torch.jit.optimize_for_inference(ts_model)
            print("[TorchScript] [OK] optimize_for_inference (Conv+BN+ReLU fusion)")
        except Exception as e:
            print(f"[TorchScript] optimize_for_inference skipped ({type(e).__name__})")

    print("  -> The first forward call finalizes JIT optimizations (warmup).")
    return ts_model


def _try_trace(model: nn.Module, example_input) -> Optional[nn.Module]:
    """Attempt a torch.jit.trace with the provided example."""
    if example_input is None:
        print("[TorchScript] trace impossible: example_input missing.")
        return None
    try:
        # strict=False: tolerates dict/list outputs (detection)
        traced = torch.jit.trace(model, example_input, strict=False)
        print("[TorchScript] [OK] Graph captured via torch.jit.trace")
        return traced
    except Exception as e:
        print(f"[TorchScript] trace failed ({type(e).__name__}: {e})")
        return None


def save_torchscript(model: nn.Module, path: str) -> str:
    """
    Save a TorchScript model (.ts) -- portable across machines.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(model, torch.jit.ScriptModule) and not hasattr(model, "save"):
        raise TypeError("The model is not a ScriptModule -- compile it first.")
    torch.jit.save(model, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"[TorchScript] Saved -> {path}  ({size_mb:.1f} MB)")
    return path


def load_torchscript(path: str, device: str = "cuda") -> nn.Module:
    """Load a saved TorchScript model (.ts)."""
    model = torch.jit.load(path, map_location=device)
    model.eval()
    print(f"[TorchScript] Loaded <- {path}")
    return model
