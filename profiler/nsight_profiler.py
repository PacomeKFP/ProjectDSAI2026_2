"""
profiler/nsight_profiler.py
---------------------------
Low-level profiling via NVIDIA Nsight Systems.

How it works:
-------------
Nsight Systems (nsys) is an external system-level profiler -- it hooks into the
Python process from the outside and records every CUDA, cuDNN, cuBLAS, NVTX,
OS event, etc.

This file plays two roles:
  1. Callable function : `profile_with_nsight(model, data, ...)`
     Adds NVTX annotations at several levels and bounds the capture window
     with cudaProfilerStart / cudaProfilerStop.
     -> To be used when the Python process is launched under nsys (see command).

  2. Standalone script : `python -m profiler.nsight_profiler --model <name> ...`
     Loads the model and the data, then calls profile_with_nsight.
     This is the script that nsys must wrap (see print_nsys_command).

Capture mechanics:
------------------
  * cudaProfilerStart / cudaProfilerStop  ->  official CUDA API to bound the
    capture window. With `--capture-range=cudaProfilerApi`, nsys only records
    what happens between these two calls (= active phase). The warmup phase
    runs but is NOT captured -> compact .nsys-rep file.

  * NVTX ranges  ->  hierarchical annotations visible in the Nsight timeline:
      Level 0: "WARMUP" / "ACTIVE"            (global phases)
      Level 1: "iter_N"                        (each iteration)
      Level 2: "preprocess" / "H2D" / "forward"  (internal steps)

Data collected (recommended nsys flags):
----------------------------------------
  --trace=cuda,nvtx,cuDNN,cublas,cusparse
      cuda    : GPU kernels, H2D/D2H copies, synchronizations
      nvtx    : our annotations + internal PyTorch/cuDNN annotations
      cuDNN   : cuDNN calls (conv, BN, pooling) with shapes and algorithms
      cublas  : cuBLAS calls (matmul, gemm) with shapes
      cusparse: sparse operations (if used)

  --cuda-memory-usage=true
      GPU allocations / deallocations with call stacks

  --gpu-metrics-device=0
      Hardware counters: SM occupancy, L1/L2 hit rate, memory bandwidth,
      IPC -- data not accessible from PyTorch

Usage:
------
  # 1. Generate the full nsys command
  python -m profiler.nsight_profiler --model retinanet_r50 --print-command

  # 2. Launch the profiling
  nsys profile \\
      --capture-range=cudaProfilerApi \\
      --trace=cuda,nvtx,cuDNN,cublas,cusparse \\
      --cuda-memory-usage=true \\
      --gpu-metrics-device=0 \\
      --output=results/profiler/nsight/retinanet_r50 \\
      python -m profiler.nsight_profiler \\
          --model retinanet_r50 \\
          --img-dir datasets/coco/val2017 \\
          --ann-file datasets/coco/annotations/instances_val2017.json \\
          --n-warmup 50 --n-active 1000

  # 3. Open the result in the Nsight Systems GUI
  #    File -> Open -> results/profiler/nsight/retinanet_r50.nsys-rep
"""

import gc
import sys
from pathlib import Path

import torch


# -- NVTX helpers ---------------------------------------------------------------
# torch.cuda.nvtx is always available with PyTorch CUDA.
# The 'nvtx' package (pip install nvtx) adds color support.

try:
    import nvtx as _nvtx_pkg
    def _push(label, color=None):
        _nvtx_pkg.push_range(label, color=color)
    def _pop():
        _nvtx_pkg.pop_range()
except ImportError:
    # Fallback: torch.cuda.nvtx (without colors)
    def _push(label, color=None):
        torch.cuda.nvtx.range_push(label)
    def _pop():
        torch.cuda.nvtx.range_pop()


class _NvtxRange:
    """NVTX context manager -- works with or without the nvtx package."""
    def __init__(self, label, color=None):
        self.label = label
        self.color = color
    def __enter__(self):
        _push(self.label, self.color)
        return self
    def __exit__(self, *_):
        _pop()


# -- Main profiler --------------------------------------------------------------

def profile_with_nsight(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=50,
    n_active=1000,
    model_name="model",
    device="cuda",
):
    """
    Annotates the forward pass with NVTX and bounds the capture with
    cudaProfilerStart / cudaProfilerStop.

    This code must be launched under nsys (see module docstring).
    In a regular execution (without nsys), the annotations are no-ops and
    cudaProfilerStart/Stop have no effect.

    Parameters
    ----------
    model         : nn.Module in eval mode
    data          : LazySampleList (at least n_warmup + n_active items)
    preprocess_fn : model.preprocess
    collate_fn    : model.collate
    n_warmup      : iterations outside the capture window (GPU warmup)
    n_active      : iterations inside the capture window
    model_name    : label used in the NVTX annotations
    device        : 'cuda' or 'cpu'
    """
    n_total = n_warmup + n_active
    if len(data) < n_total:
        raise ValueError(
            f"data contains {len(data)} samples, need {n_total}."
        )

    model.eval()
    cudart = torch.cuda.cudart()

    # ==========================================================================
    # WARMUP PHASE -- outside the nsys capture window
    # ==========================================================================
    with _NvtxRange("WARMUP", color="gray"):
        with torch.no_grad():
            for i, s in enumerate(data[:n_warmup]):
                with _NvtxRange(f"warmup_iter_{i}", color="gray"):

                    with _NvtxRange("preprocess", color="blue"):
                        inp = preprocess_fn(s)

                    with _NvtxRange("H2D", color="orange"):
                        gpu = collate_fn([inp], device)
                        del inp
                        torch.cuda.synchronize()

                    with _NvtxRange("forward", color="gray"):
                        model(gpu)

                    del gpu

    torch.cuda.synchronize()

    # ==========================================================================
    # ACTIVE PHASE -- nsys capture window (cudaProfilerApi)
    # ==========================================================================
    cudart.cudaProfilerStart()

    with _NvtxRange("ACTIVE", color="green"):
        with torch.no_grad():
            for i, s in enumerate(data[n_warmup:n_total]):
                with _NvtxRange(f"{model_name}_iter_{i}", color="white"):

                    with _NvtxRange("preprocess", color="blue"):
                        inp = preprocess_fn(s)

                    with _NvtxRange("H2D", color="orange"):
                        gpu = collate_fn([inp], device)
                        del inp
                        torch.cuda.synchronize()   # H2D done before forward

                    with _NvtxRange("forward", color="red"):
                        model(gpu)

                    torch.cuda.synchronize()       # forward done
                    del gpu

    cudart.cudaProfilerStop()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# -- nsys command generator ----------------------------------------------------

def print_nsys_command(
    model_name,
    img_dir="datasets/coco/val2017",
    ann_file="datasets/coco/annotations/instances_val2017.json",
    n_warmup=50,
    n_active=1000,
    output_dir="results/profiler/nsight",
    device="cuda",
):
    """Print the full nsys command ready to copy-paste."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = str(Path(output_dir) / model_name)
    cmd = (
        f"nsys profile \\\n"
        f"    --capture-range=cudaProfilerApi \\\n"
        f"    --trace=cuda,nvtx,cuDNN,cublas,cusparse \\\n"
        f"    --cuda-memory-usage=true \\\n"
        f"    --output={out} \\\n"
        f"    python -m profiler.nsight_profiler \\\n"
        f"        --model {model_name} \\\n"
        f"        --img-dir {img_dir} \\\n"
        f"        --ann-file {ann_file} \\\n"
        f"        --n-warmup {n_warmup} \\\n"
        f"        --n-active {n_active} \\\n"
        f"        --device {device}"
    )
    print("\n-- Nsight Systems command ------------------------------------")
    print(cmd)
    print("--------------------------------------------------------------\n")
    return cmd


# -- Standalone script (launched by nsys) --------------------------------------

_MODEL_MAP = {
    "retinanet_r50":    "models.retinanet_r50",
    "retinanet_r101":   "models.retinanet_r101",
    "fcos_r50":         "models.fcos_r50",
    "efficientdet_d4":  "models.efficientdet_d4",
    "efficientdet_d5":  "models.efficientdet_d5",
    "efficientdet_d6":  "models.efficientdet_d6",
}


if __name__ == "__main__":
    import argparse
    import importlib

    # Add the project root to the path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from utils.data_loader import load_profiling_data

    parser = argparse.ArgumentParser(description="Nsight Systems profiling script")
    parser.add_argument("--model",    required=True, choices=list(_MODEL_MAP))
    parser.add_argument("--img-dir",  default="datasets/coco/val2017")
    parser.add_argument("--ann-file", default="datasets/coco/annotations/instances_val2017.json")
    parser.add_argument("--n-warmup", type=int, default=50)
    parser.add_argument("--n-active", type=int, default=1000)
    parser.add_argument("--device",   default="cuda")
    parser.add_argument("--print-command", action="store_true",
                        help="Print the nsys command and exit")
    args = parser.parse_args()

    if args.print_command:
        print_nsys_command(
            model_name=args.model,
            img_dir=args.img_dir,
            ann_file=args.ann_file,
            n_warmup=args.n_warmup,
            n_active=args.n_active,
            device=args.device,
        )
        sys.exit(0)

    # Dynamic loading of the model module
    mod = importlib.import_module(_MODEL_MAP[args.model])

    # Create the output directory if needed (nsys does not do it)
    Path("results/profiler/nsight").mkdir(parents=True, exist_ok=True)

    print(f"[nsight] Loading model: {args.model}")
    model = mod.load_model(args.device)

    print(f"[nsight] Loading data: {args.n_warmup + args.n_active} images")
    data = load_profiling_data(
        args.img_dir, args.ann_file,
        n=args.n_warmup + args.n_active,
    )

    print(f"[nsight] Starting -- warmup={args.n_warmup}  active={args.n_active}")
    profile_with_nsight(
        model=model,
        data=data,
        preprocess_fn=mod.preprocess,
        collate_fn=mod.collate,
        n_warmup=args.n_warmup,
        n_active=args.n_active,
        model_name=args.model,
        device=args.device,
    )
    print("[nsight] Done. Open the .nsys-rep file in the Nsight Systems GUI.")
