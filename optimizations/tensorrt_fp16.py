"""
optimizations/tensorrt_fp16.py
===============================
TensorRT FP16 via torch_tensorrt -- kernel fusion + half precision.

Dependencies:
  torch >= 2.1
  torch-tensorrt >= 2.1    pip install torch-tensorrt
  tensorrt >= 8.6          pre-installed on Colab GPU / NGC containers
                           (bundled with torch-tensorrt on Colab)

  [!] Windows: TensorRT is NOT available on Windows via standard pip.
    Use Colab, an NVIDIA container, or WSL2.

What it does:
  torch_tensorrt compiles the PyTorch model into an optimized TensorRT engine.
  The approach used here is the torch.compile backend (ir="torch_compile"),
  which is the most robust for detection models:
    - Keeps the NMS and complex Python ops inside PyTorch (not exported to TRT)
    - Sends the Conv/BN/ReLU/FPN blocks into optimized TRT sub-graphs
    - Returns a model with EXACTLY the same API as the original
      -> works directly with benchmark_model() and run_map_evaluation()

  Optimizations applied by TRT:
    * Conv + BN + activation fusion -> single kernel (Conv-BN-ReLU -> CBR kernel)
    * FP16 Tensor Cores (2x throughput vs FP32 on Ampere+)
    * CUDA kernel execution-plan optimization
    * Reuse of intermediate memory buffers

  min_block_size: minimum number of consecutive ops to form a TRT block.
    Too small -> too many PyTorch<->TRT transitions (overhead).
    Recommended: 5 for detection models (residual blocks = 6+ ops).

Saving:
  TRT via torch.compile cannot be serialized directly (the engine is compiled
  on the fly at the first call). To save a persistent TRT engine, use the
  ExportedProgram variant (see save_trt_model()).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def build_trt_fp16(
    model: nn.Module,
    min_block_size: int = 5,
    workspace_gb: float = 4.0,
    debug: bool = False,
) -> nn.Module:
    """
    Compile the model with TensorRT FP16 via torch.compile.

    Parameters
    ----------
    model          : nn.Module in eval mode -- from load_model()
    min_block_size : minimum ops per TRT block (5 recommended for detection)
    workspace_gb   : max TRT workspace size in GB
    debug          : print TRT logs (verbose)

    Returns
    -------
    nn.Module with the same API as the original (drop-in replacement).
    The first forward call triggers TRT compilation (long warmup).
    """
    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError(
            "torch-tensorrt not installed.\n"
            "  pip install torch-tensorrt\n"
            "  (Colab: !pip install torch-tensorrt)"
        )

    _check_cuda()
    model.eval()

    workspace_bytes = int(workspace_gb * (1 << 30))

    trt_model = torch.compile(
        model,
        backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float16},
            "min_block_size":     min_block_size,
            "workspace_size":     workspace_bytes,
            "debug":              debug,
            "truncate_long_and_double": True,
        },
    )

    print(f"[TRT FP16] Model compiled with torch_tensorrt backend.")
    print(f"  min_block_size={min_block_size}  workspace={workspace_gb:.1f} GB")
    print("  -> First forward call triggers TRT compilation.")
    print("  -> Include in n_warmup (at least 3 extra calls).")
    return trt_model


def save_trt_model(
    model: nn.Module,
    save_path: str,
    sample_input: torch.Tensor,
    fp16: bool = True,
    min_block_size: int = 5,
) -> str:
    """
    Compile and save a persistent TRT engine via ExportedProgram.
    Longer to build, but can be reloaded without recompilation.

    sample_input: Tensor[1, 3, H, W] on CUDA -- defines the frozen shapes.

    Returns
    -------
    str: path of the saved .ts file
    """
    try:
        import torch_tensorrt
    except ImportError:
        raise ImportError("pip install torch-tensorrt")

    _check_cuda()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    with torch.no_grad():
        exp = torch.export.export(model, (sample_input,))

    precision = {torch.float16} if fp16 else {torch.float32}
    trt_ep = torch_tensorrt.dynamo.compile(
        exp,
        inputs=[sample_input],
        enabled_precisions=precision,
        min_block_size=min_block_size,
    )

    torch_tensorrt.save(trt_ep, save_path, inputs=[sample_input])
    size_mb = Path(save_path).stat().st_size / 1e6
    print(f"[TRT FP16] Engine saved -> {save_path}  ({size_mb:.1f} MB)")
    return save_path


def load_trt_model(path: str) -> nn.Module:
    """
    Load a TRT engine saved with save_trt_model().
    Requires torch_tensorrt to be imported for deserialization.
    """
    try:
        import torch_tensorrt  # noqa: F401 -- required for the deserializer
    except ImportError:
        raise ImportError("pip install torch-tensorrt")

    model = torch.export.load(path)
    print(f"[TRT FP16] Engine loaded <- {path}")
    return model


# -- Utility -------------------------------------------------------------------

def _check_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. TensorRT requires an NVIDIA GPU."
        )
    device_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    print(f"[TRT FP16] GPU: {device_name}  (Compute Capability {cc_major}.{cc_minor})")
    if cc_major < 7:
        print("  [!] FP16 Tensor Cores available from Volta (CC 7.0+).")
        print("  The model will compile but FP16 gains will be limited.")
