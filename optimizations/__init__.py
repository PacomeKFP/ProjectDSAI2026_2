"""
optimizations/
══════════════
Boîte à outils d'optimisation pour l'inférence GPU des modèles de détection.

Disponibilité par environnement (voir capability.py pour la détection auto) :

  Module / technique      Windows local   Colab (Linux+GPU)
  ──────────────────────  ──────────────  ─────────────────
  capability.py           ✓               ✓     (détection d'environnement)
  fp16_half.py            ✓               ✓     (FP16 autocast, Tensor Cores)
  torchscript.py          ✓               ✓     (graphe TorchScript + fusion Conv-BN)
  torch_compile.py        ✓ (cudagraphs)  ✓ (inductor)
  onnx_export.py          ✓               ✓     (export ONNX)
  ort_inference.py        ✓               ✓     (ONNX Runtime CUDA EP)
  tensorrt_fp16.py        ✗               ✓     (TensorRT FP16)
  tensorrt_int8.py        ✗               ✓     (TensorRT INT8 + calibration)

Chaque fonction retourne un modèle qui préserve l'API d'origine
(même signature forward) → drop-in pour benchmark_model() et run_map_evaluation().
"""

# Détection d'environnement — toujours disponible
from .capability import (
    detect, print_report, Capabilities,
    has_triton, has_torch_tensorrt, has_onnx, has_onnxruntime,
    default_compile_backend,
)

# Techniques cross-platform (Windows + Colab)
from .fp16_half     import to_fp16_autocast, to_fp16_half, AutocastModel
from .torchscript   import (
    optimize_with_torchscript, save_torchscript, load_torchscript,
)
from .torch_compile import compile_model, save_compiled
from .onnx_export   import export_to_onnx, check_onnx, validate_outputs
from .ort_inference import build_ort_model, ORTModel

# Techniques Colab/Linux uniquement — imports légers (deps chargées à l'appel)
from .tensorrt_fp16 import build_trt_fp16, load_trt_model
from .tensorrt_int8 import build_trt_int8

# Orchestrateur (stdlib uniquement au niveau module → import sûr)
from .runner import (
    OptimizationRunner, RunConfig, ModelSpec, VariantSpec, DEFAULT_VARIANTS,
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
    # runner
    "OptimizationRunner", "RunConfig", "ModelSpec", "VariantSpec", "DEFAULT_VARIANTS",
]
