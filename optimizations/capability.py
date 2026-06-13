"""
optimizations/capability.py
════════════════════════════
Détection centralisée des capacités de l'environnement.

Chaque technique d'optimisation a des dépendances différentes et n'est pas
disponible partout. Ce module centralise la détection pour que le notebook
et les modules d'optimisation partagent une seule source de vérité.

Matrice de disponibilité (typique) :

  Technique              Windows local   Colab (Linux+GPU)   Dépend de
  ─────────────────────  ──────────────  ──────────────────  ──────────────────
  FP16 autocast          ✓               ✓                   CUDA (Tensor Cores)
  TorchScript            ✓               ✓                   torch (natif)
  torch.compile cudagraphs ✓             ✓                   CUDA
  torch.compile inductor  ✗ (pas Triton) ✓                   triton
  ONNX export            ✓               ✓                   onnx
  ONNX Runtime           ✓               ✓                   onnxruntime-gpu
  TensorRT FP16/INT8     ✗               ✓                   torch-tensorrt

Note : torch.compile a deux backends pertinents ici :
  - inductor  : génère des kernels Triton fusionnés (le plus rapide, mais Triton
                n'est pas packagé pour Windows par défaut)
  - cudagraphs: capture le graphe CUDA pour éliminer l'overhead de lancement
                des kernels (pas de codegen → aucune dépendance Triton)
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Dict

import torch


# ── Marques d'état tolérantes à l'encodage ────────────────────────────────────
# Jupyter est en UTF-8 (✓/✗ OK). Certains terminaux Windows (cp1252) ne savent
# pas encoder ces caractères → on détecte et on bascule sur des marques ASCII.

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or ""
    try:
        "✓✗".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError, TypeError):
        return False


_UNI = _supports_unicode()
_OK  = "✓" if _UNI else "[OK]"
_NO  = "✗" if _UNI else "[--]"


def has_triton() -> bool:
    """True si Triton est installé et utilisable (backend inductor de torch.compile)."""
    try:
        import triton  # noqa: F401
        # torch expose aussi un check qui vérifie la compatibilité GPU
        from torch.utils._triton import has_triton as _torch_has_triton
        return bool(_torch_has_triton())
    except Exception:
        try:
            import triton  # noqa: F401
            return True
        except Exception:
            return False


def has_torch_tensorrt() -> bool:
    """True si torch-tensorrt est importable ET un GPU CUDA est présent."""
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
    Backend torch.compile recommandé selon l'environnement.
      - "inductor"  si Triton disponible (fusion Triton complète)
      - "cudagraphs" sinon, si CUDA disponible (capture de graphe, sans Triton)
      - "eager"     en dernier recours (aucune accélération réelle)
    """
    if has_triton():
        return "inductor"
    if torch.cuda.is_available():
        return "cudagraphs"
    return "eager"


@dataclass
class Capabilities:
    """Instantané des capacités de l'environnement courant."""
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
        """Tableau lisible des techniques disponibles."""
        def mark(ok): return f"{_OK} disponible" if ok else f"{_NO} indisponible"
        inductor_label = "  - inductor (Triton)" if not _UNI else "  └ inductor (Triton)"
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
    """Construit l'instantané des capacités de l'environnement courant."""
    cuda = torch.cuda.is_available()
    caps = Capabilities(
        platform        = platform.system(),
        is_colab        = "google.colab" in sys.modules,
        cuda            = cuda,
        gpu_name        = torch.cuda.get_device_name(0) if cuda else "—",
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
    """Affiche un rapport complet et retourne l'objet Capabilities."""
    caps = detect()
    print(f"Plateforme   : {caps.platform} ({'Colab' if caps.is_colab else 'local'})")
    print(f"PyTorch      : {caps.torch_version}")
    print(f"CUDA         : {_OK + ' ' + caps.gpu_name if caps.cuda else _NO + ' (CPU)'}")
    print(f"Backend compile recommandé : {caps.compile_backend}")
    print()
    print("Techniques d'optimisation disponibles :")
    print(caps.matrix())
    return caps
