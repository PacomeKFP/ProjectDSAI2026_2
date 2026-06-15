"""
profiler/nsight_profiler.py
───────────────────────────
Profilage bas-niveau via NVIDIA Nsight Systems.

Principe de fonctionnement :
─────────────────────────────
Nsight Systems (nsys) est un profiler système externe — il s'accroche au
processus Python depuis l'extérieur et enregistre tous les événements CUDA,
cuDNN, cublas, NVTX, OS, etc.

Ce fichier joue deux rôles :
  1. Fonction callable  : `profile_with_nsight(model, data, ...)`
     Ajoute des annotations NVTX à plusieurs niveaux et délimite la fenêtre
     de capture avec cudaProfilerStart / cudaProfilerStop.
     → À utiliser quand le process Python est lancé sous nsys (voir commande).

  2. Script standalone  : `python -m profiler.nsight_profiler --model <nom> ...`
     Charge le modèle et les données, puis appelle profile_with_nsight.
     C'est ce script que nsys doit envelopper (voir print_nsys_command).

Mécanique de capture :
──────────────────────
  • cudaProfilerStart / cudaProfilerStop  →  API CUDA officielle pour délimiter
    la fenêtre de capture. Avec `--capture-range=cudaProfilerApi`, nsys
    n'enregistre que ce qui se passe entre ces deux appels (= phase active).
    La phase warmup est exécutée mais NON capturée → fichier .nsys-rep compact.

  • NVTX ranges  →  annotations hiérarchiques visibles dans la timeline Nsight :
      Niveau 0 : "WARMUP" / "ACTIVE"           (phases globales)
      Niveau 1 : "iter_N"                       (chaque itération)
      Niveau 2 : "preprocess" / "H2D" / "forward"  (étapes internes)

Données collectées (flags nsys recommandés) :
──────────────────────────────────────────────
  --trace=cuda,nvtx,cuDNN,cublas,cusparse
      cuda    : kernels GPU, copies H2D/D2H, synchronisations
      nvtx    : nos annotations + annotations internes PyTorch/cuDNN
      cuDNN   : appels cuDNN (conv, BN, pooling) avec formes et algorithmes
      cublas  : appels cuBLAS (matmul, gemm) avec formes
      cusparse: opérations sparse (si utilisées)

  --cuda-memory-usage=true
      Allocations / désallocations GPU avec pile d'appels

  --gpu-metrics-device=0
      Compteurs hardware : SM occupancy, L1/L2 hit rate,
      bande passante mémoire, IPC — données inaccessibles depuis PyTorch

Usage :
───────
  # 1. Générer la commande nsys complète
  python -m profiler.nsight_profiler --model retinanet_r50 --print-command

  # 2. Lancer le profiling
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

  # 3. Ouvrir le résultat dans Nsight Systems GUI
  #    File → Open → results/profiler/nsight/retinanet_r50.nsys-rep
"""

import gc
import sys
from pathlib import Path

import torch


# ── NVTX helpers ───────────────────────────────────────────────────────────────
# torch.cuda.nvtx est toujours disponible avec PyTorch CUDA.
# Le package 'nvtx' (pip install nvtx) ajoute la gestion des couleurs.

try:
    import nvtx as _nvtx_pkg
    def _push(label, color=None):
        _nvtx_pkg.push_range(label, color=color)
    def _pop():
        _nvtx_pkg.pop_range()
except ImportError:
    # Fallback : torch.cuda.nvtx (sans couleurs)
    def _push(label, color=None):
        torch.cuda.nvtx.range_push(label)
    def _pop():
        torch.cuda.nvtx.range_pop()


class _NvtxRange:
    """Context manager NVTX — fonctionne avec ou sans le package nvtx."""
    def __init__(self, label, color=None):
        self.label = label
        self.color = color
    def __enter__(self):
        _push(self.label, self.color)
        return self
    def __exit__(self, *_):
        _pop()


# ── Profiler principal ─────────────────────────────────────────────────────────

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
    Annote le forward pass avec NVTX et délimite la capture avec
    cudaProfilerStart / cudaProfilerStop.

    Ce code doit être lancé sous nsys (voir module docstring).
    En exécution normale (sans nsys), les annotations sont des no-ops
    et cudaProfilerStart/Stop sont sans effet.

    Parameters
    ----------
    model         : nn.Module en mode eval
    data          : LazySampleList (n_warmup + n_active éléments minimum)
    preprocess_fn : model.preprocess
    collate_fn    : model.collate
    n_warmup      : itérations hors fenêtre de capture (GPU warmup)
    n_active      : itérations dans la fenêtre de capture
    model_name    : label utilisé dans les annotations NVTX
    device        : 'cuda' ou 'cpu'
    """
    n_total = n_warmup + n_active
    if len(data) < n_total:
        raise ValueError(
            f"data contient {len(data)} samples, besoin de {n_total}."
        )

    model.eval()
    cudart = torch.cuda.cudart()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE WARMUP — hors fenêtre de capture nsys
    # ══════════════════════════════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE ACTIVE — fenêtre de capture nsys (cudaProfilerApi)
    # ══════════════════════════════════════════════════════════════════════════
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
                        torch.cuda.synchronize()   # H2D terminé avant forward

                    with _NvtxRange("forward", color="red"):
                        model(gpu)

                    torch.cuda.synchronize()       # forward terminé
                    del gpu

    cudart.cudaProfilerStop()

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


# ── Générateur de commande nsys ────────────────────────────────────────────────

def print_nsys_command(
    model_name,
    img_dir="datasets/coco/val2017",
    ann_file="datasets/coco/annotations/instances_val2017.json",
    n_warmup=50,
    n_active=1000,
    output_dir="results/profiler/nsight",
    device="cuda",
):
    """Affiche la commande nsys complète prête à copier-coller."""
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
    print("\n-- Commande Nsight Systems -----------------------------------")
    print(cmd)
    print("--------------------------------------------------------------\n")
    return cmd


# ── Script standalone (lancé par nsys) ────────────────────────────────────────

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

    # Ajouter la racine du projet au path
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
                        help="Afficher la commande nsys et quitter")
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

    # Chargement dynamique du module modèle
    mod = importlib.import_module(_MODEL_MAP[args.model])

    # Créer le répertoire de sortie si nécessaire (nsys ne le fait pas)
    Path("results/profiler/nsight").mkdir(parents=True, exist_ok=True)

    print(f"[nsight] Chargement modèle : {args.model}")
    model = mod.load_model(args.device)

    print(f"[nsight] Chargement données : {args.n_warmup + args.n_active} images")
    data = load_profiling_data(
        args.img_dir, args.ann_file,
        n=args.n_warmup + args.n_active,
    )

    print(f"[nsight] Démarrage — warmup={args.n_warmup}  active={args.n_active}")
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
    print("[nsight] Terminé. Ouvrir le .nsys-rep dans Nsight Systems GUI.")
