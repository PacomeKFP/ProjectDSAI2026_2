"""
optimizations/torch_compile.py
===============================
Graph compilation via torch.compile (PyTorch 2.x).

Dependencies:
  torch >= 2.0
  For the "inductor" backend (fastest): triton
    * Linux / Colab: Triton ships with PyTorch CUDA -> inductor backend works.
    * Windows      : Triton is NOT packaged by default -> inductor raises
                     TritonMissing. We automatically fall back to "cudagraphs".

What it does -- two relevant backends:

  +- inductor (Linux/Colab) ------------------------------------------------+
  | Generates fused Triton kernels from the FX graph.                        |
  | Vertical fusion (Conv->BN->ReLU) + horizontal (parallel ops).              |
  | Typical gain: x1.3 to x2.0. First call is slow (codegen + autotune).     |
  +--------------------------------------------------------------------------+

  +- cudagraphs (Windows + Colab) ------------------------------------------+
  | Captures the sequence of CUDA kernels as a replayable "CUDA graph".      |
  | Does NOT remove compute cost, but removes the CPU launch overhead of     |
  | each kernel (Python dispatch + driver). Very effective when the model    |
  | launches many small kernels (ResNet/EfficientNet).                       |
  | No Triton dependency. Typical gain: x1.1 to x1.5.                         |
  | Constraint: fixed input shapes (OK here, benchmark at 640x640).          |
  +--------------------------------------------------------------------------+

The returned model keeps exactly the original API
(List[Tensor] for RetinaNet, Tensor[B,C,H,W] for EfficientDet)
-> directly usable with benchmark_model() and run_map_evaluation().

The first forward call triggers JIT compilation (longer warmup):
budget at least a few extra iterations in n_warmup.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .capability import default_compile_backend, has_triton

_INDUCTOR_MODES = {"default", "reduce-overhead", "max-autotune"}


def compile_model(
    model: nn.Module,
    backend: str | None = None,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> nn.Module:
    """
    Compile the model with torch.compile, choosing the appropriate backend.

    Parameters
    ----------
    model     : nn.Module in eval mode -- from load_model()
    backend   : None  -> auto ("inductor" if Triton, else "cudagraphs")
                "inductor" | "cudagraphs" | "eager" | ...
    mode      : used only by the inductor backend
                "default" recommended for detection (NOT "reduce-overhead",
                which enables cudagraphs -> incompatible with NMS dynamic shapes).
    fullgraph : require a full graph without graph break (False recommended
                for detection -- the NMS is not traceable)
    dynamic   : False (default) freezes shapes -> far fewer symbolic guards
                -> compilation MUCH faster. Crucial for EfficientDet/BiFPN,
                where dynamic=True blows up the sympy solver (compilation > 45 min).
                Safe here because the benchmark runs at a fixed 640x640 resolution.

    Returns
    -------
    nn.Module with the same API as the original. First forward = JIT compilation.
    """
    if backend is None:
        backend = default_compile_backend()

    model.eval()

    if backend == "inductor" and not has_triton():
        print("[torch.compile] [!] 'inductor' backend requested but Triton missing.")
        print("  -> Auto-falling back to 'cudagraphs'.")
        backend = "cudagraphs"

    # -- inductor backend: uses `mode` (Triton codegen) ------------------------
    if backend == "inductor":
        if mode not in _INDUCTOR_MODES:
            raise ValueError(f"mode must be in {_INDUCTOR_MODES}, got {mode!r}")
        if mode == "reduce-overhead":
            print("[torch.compile] [!] mode='reduce-overhead' enables cudagraphs ->")
            print("  risk of incompatibility with NMS (dynamic shapes).")
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        print(f"[torch.compile] backend=inductor  mode={mode!r}  dynamic={dynamic}")
        print("  Triton kernel fusion. First call is slow (codegen).")

    # -- cudagraphs backend: CUDA graph capture, no Triton ---------------------
    elif backend == "cudagraphs":
        compiled = torch.compile(model, backend="cudagraphs", fullgraph=fullgraph, dynamic=dynamic)
        print("[torch.compile] backend=cudagraphs")
        print("  [!] FIXED-shape graph capture -- incompatible with the dynamic NMS")
        print("    of full detectors. Reserve for backbone-only.")

    # -- Other backends (eager, aot_eager, ...) ----------------------------------
    else:
        compiled = torch.compile(model, backend=backend, fullgraph=fullgraph, dynamic=dynamic)
        print(f"[torch.compile] backend={backend!r}  dynamic={dynamic}")

    print("  -> Include the first call in n_warmup (>= 5 extra iterations margin).")
    return compiled


def save_compiled(model: nn.Module, path: str) -> None:
    """
    Save the state_dict of the underlying model.
    torch.compile does not modify the weights -- we serialize the original module.
    To reload: rebuild the PyTorch model then call compile_model() again.
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    underlying = getattr(model, "_orig_mod", model)
    torch.save(underlying.state_dict(), path)
    print(f"[torch.compile] state_dict saved -> {path}")
    print("  To reload: load_model() then compile_model().")
