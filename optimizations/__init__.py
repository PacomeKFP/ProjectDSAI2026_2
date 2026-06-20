"""
optimizations/
==============
Optimization toolbox for GPU inference of detection models.

Availability per environment (see capability.py for auto-detection):

  Module / technique      Windows local   Colab (Linux+GPU)
  ----------------------  --------------  -----------------
  capability.py           [OK]               [OK]     (environment detection)
  fp16_half.py            [OK]               [OK]     (FP16 autocast, Tensor Cores)
  torchscript.py          [OK]               [OK]     (TorchScript graph + Conv-BN fusion)
  torch_compile.py        [OK] (cudagraphs)  [OK] (inductor)
  onnx_export.py          [OK]               [OK]     (ONNX export)
  ort_inference.py        [OK]               [OK]     (ONNX Runtime CUDA EP)
  tensorrt_fp16.py        [X]               [OK]     (TensorRT FP16)
  tensorrt_int8.py        [X]               [OK]     (TensorRT INT8 + calibration)

Each function returns a model that preserves the original API (same forward
signature) -> drop-in for benchmark_model() and run_map_evaluation().
"""

# Environment detection -- always available
from .capability import (
    detect, print_report, Capabilities,
    has_triton, has_torch_tensorrt, has_onnx, has_onnxruntime,
    default_compile_backend,
)

# Cross-platform techniques (Windows + Colab)
from .fp16_half     import to_fp16_autocast, to_fp16_half, AutocastModel
from .torchscript   import (
    optimize_with_torchscript, save_torchscript, load_torchscript,
)
from .torch_compile import compile_model, save_compiled
from .onnx_export   import export_to_onnx, check_onnx, validate_outputs
from .ort_inference import build_ort_model, ORTModel

# Colab/Linux-only techniques -- lightweight imports (deps loaded on call)
from .tensorrt_fp16 import build_trt_fp16, load_trt_model
from .tensorrt_int8 import build_trt_int8

# Output paths (Drive prefix on Colab)
from .paths import out_path, ensure_dir, set_prefix, project_prefix, describe as describe_paths

# Architecture-aware zone and sub-zone optimization
from .zones import (
    apply_zone_optimization, apply_subzone_plan, get_static_zone, get_subzone,
    get_coarse_zones, capture_subzone_inputs, SUBZONES,
    opt_torchscript, opt_compile, opt_cudagraphs,
    opt_trt_fp16, opt_trt_fp16_folded, opt_trt_int8,
)

# Orchestrator
from .runner import (
    OptimizationRunner, RunConfig, ModelSpec, VariantSpec,
    DEFAULT_VARIANTS, FULL_VARIANTS,
)

__all__ = [
    # capability
    "detect", "print_report", "Capabilities",
    "has_triton", "has_torch_tensorrt", "has_onnx", "has_onnxruntime",
    "default_compile_backend",
    # fp16
    "to_fp16_autocast", "to_fp16_half", "AutocastModel",
    # torchscript
    "optimize_with_torchscript", "save_torchscript", "load_torchscript",
    # torch.compile
    "compile_model", "save_compiled",
    # onnx / ort
    "export_to_onnx", "check_onnx", "validate_outputs",
    "build_ort_model", "ORTModel",
    # tensorrt
    "build_trt_fp16", "load_trt_model", "build_trt_int8",
    # paths
    "out_path", "ensure_dir", "set_prefix", "project_prefix", "describe_paths",
    # zones
    "apply_zone_optimization", "apply_subzone_plan", "get_static_zone", "get_subzone",
    "get_coarse_zones", "capture_subzone_inputs", "SUBZONES",
    "opt_torchscript", "opt_compile", "opt_cudagraphs",
    "opt_trt_fp16", "opt_trt_fp16_folded", "opt_trt_int8",
    # runner
    "OptimizationRunner", "RunConfig", "ModelSpec", "VariantSpec",
    "DEFAULT_VARIANTS", "FULL_VARIANTS",
]
