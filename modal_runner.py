"""
modal_runner.py
═══════════════
Orchestration Modal — 1 conteneur T4 (16 GB VRAM) par (modèle, variante).

────────────────────────────────────────────────────────────────────────────────
DESIGN
────────────────────────────────────────────────────────────────────────────────

  • Image : Debian slim + Python 3.13 + tout depuis PyPI (validé par
            modal_test_env.py). pip résout torch 2.8+cu128, torch-tensorrt 2.8,
            tensorrt 10.12, numpy 2.4, cv2 4.13 en cohérence — pas de conflit
            numpy 1.x/2.x. TOUTES les variantes (TRT incluses) sont actives.

  • Volume `dsai2026` (50 GB+), monté à /data :
        /data/coco/                      dataset COCO val2017 (téléchargé 1 fois)
        /data/cache/torch                TORCH_HOME partagé (R50/FCOS DL une fois)
        /data/cache/hf                   HuggingFace cache (timm pour effdet)
        /data/results/<run_id>/
            bench/<model>_<variant>.json
            eval/<model>_<variant>.json
            modules/<model>_<variant>.csv
            profiles/<model>_<variant>.csv
            errors/<model>_<variant>.txt   tracebacks en cas d'échec
            logs/<model>_<variant>.log    tee stdout+stderr du conteneur
            results.csv                    agrégé après le run

  • Granularité : 1 fonction Modal = 1 couple (model, variant). En parallèle
    via `.starmap()`. Avantages :
      - pollution d'état entre variantes = impossible (process différents)
      - timeouts heartbeat évités (chaque job est court et isolé)
      - relance ciblée d'un seul couple gratuite

  • Logs : Modal capture stdout/stderr automatiquement par conteneur, et on
    duplique aussi dans le Volume via Tee (fichier `.log` par job).

────────────────────────────────────────────────────────────────────────────────
PARAMÈTRES NORMAUX (cahier des charges)
────────────────────────────────────────────────────────────────────────────────
  N_PROFILE_DATA = 2000   images chargées (bench + profiling)
  N_WARMUP       = 50     itérations de chauffe
  N_MEASURE      = 1000   itérations actives benchmark
  N_PROFILE      = 150    itérations actives profiling
  N_EVAL         = 2000   images pour la MAP@640

────────────────────────────────────────────────────────────────────────────────
COMMANDES
────────────────────────────────────────────────────────────────────────────────
  modal run modal_runner.py                                # tout (parallèle)
  modal run modal_runner.py --models retinanet_r50         # un seul modèle
  modal run modal_runner.py --variants baseline,fp16,zone_trt_fp16
  modal run modal_runner.py --parallel 4                   # ≤4 conteneurs simultanés
  modal run modal_runner.py::ensure_coco                   # juste télécharger COCO
  modal run modal_runner.py::aggregate --run-id 20260616_xxxxxx   # ré-agréger

Récupérer un run :
  modal volume get dsai2026 results/<run_id> ./local_run/

Naviguer dans le Volume :
  modal volume ls dsai2026 results/
  modal shell --volume dsai2026:/data
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import modal


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

APP_NAME      = "dsai2026-opt"
VOLUME_NAME   = "dsai2026"
PROJECT_ROOT  = "/workspace/project"

# Données
COCO_DIR      = "/data/coco"
IMG_DIR       = f"{COCO_DIR}/val2017"
ANN_FILE      = f"{COCO_DIR}/annotations/instances_val2017.json"
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Caches partagés (téléchargements modèles)
TORCH_CACHE   = "/data/cache/torch"
HF_CACHE      = "/data/cache/hf"

# Paramètres expérience par défaut
N_PROFILE_DATA = 2000
N_WARMUP       = 50
N_MEASURE      = 1000
N_PROFILE      = 150
N_EVAL         = 2000


# ══════════════════════════════════════════════════════════════════════════════
# Image : COPIE EXACTE du run A100 qui a fonctionné
# ══════════════════════════════════════════════════════════════════════════════
#
# Historique honnête :
#
#   • Run A100 (succès partiel — c'est la BASE qu'on copie ici) :
#       NGC 24.10 + add_python="3.11" + pip install (effdet, timm,
#       pycocotools, opencv-python-headless, pillow, pandas, numpy, psutil,
#       tqdm, tensorboard, nvtx, onnx, onnxruntime-gpu, onnxsim).
#       Résultat : baseline, fp16, compile_fp16 → OK. trt_fp16 → SKIPPED
#       (TRT indisponible car add_python remplace Python natif NGC ; les
#       paquets torch_tensorrt/tensorrt préinstallés pour Python 3.10 ne
#       sont plus accessibles depuis Python 3.11).
#
#   • Run T4 #1 (no-add_python + --no-deps) : ÉCHEC install deps.
#
#   • Run T4 #2 (add_python + torch-tensorrt explicite) : ÉCHEC install
#       (conflits de version torch-tensorrt vs torch tiré par effdet).
#
#   • Présente version : ON COPIE LITTÉRALEMENT LE RUN A100. TRT restera
#     INDISPONIBLE (les variantes *trt* seront SKIPPED par le runner via
#     `requires="trt"`), mais tout le reste tourne :
#       baseline, fp16, torchscript, compile, cudagraphs, compile_fp16,
#       cudagraphs_fp16, torchscript_fp16, et toutes les zones non-TRT
#       (zone_torchscript, zone_compile, zone_cudagraphs).
#
#     → 11 variantes par modèle qui marchent, sur les 15 totales.
#     → Si on veut TRT plus tard, il faudra repartir d'une autre base
#       (containers NGC sans add_python + image custom).
#
# NE PAS TOUCHER À CETTE LISTE — elle correspond au run A100 mot pour mot.

image = (
    # Image VIERGE Debian + Python 3.13 + tout depuis PyPI.
    # Validé : torch 2.8+cu128, torch-tensorrt 2.8, tensorrt 10.12, numpy 2.4,
    # cv2 4.13, effdet, timm, onnx, onnxruntime-gpu — pas de conflit numpy.
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # PyTorch CUDA bundled (wheels PyPI)
        "torch", "torchvision",
        # TensorRT stack — pip résout en cohérence avec torch
        "tensorrt", "torch-tensorrt",
        # Détection
        "effdet", "timm",
        # Données / éval
        "pycocotools", "opencv-python-headless", "pillow",
        # Utilitaires
        "pandas", "numpy", "psutil", "tqdm",
        # Profilage
        "tensorboard", "nvtx",
        # ONNX
        "onnx", "onnxruntime-gpu", "onnxsim",
    )
    .add_local_dir(
        ".",
        remote_path=PROJECT_ROOT,
        ignore=[
            ".git", ".venv", "venv", "env", "__pycache__",
            "results", "outputs", "datasets",
            "*.ipynb_checkpoints", "*.nsys-rep",
            ".vscode", ".idea", "node_modules", "local_results", "local_run",
        ],
    )
)

app    = modal.App(APP_NAME)
# Volume créé à la demande, taille unique 1 TB inclus dans plan Starter
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ══════════════════════════════════════════════════════════════════════════════
# Cartographie : modèles & variantes
# ══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "retinanet_r50":   ("models.retinanet_r50",   "torchvision"),
    "fcos_r50":        ("models.fcos_r50",        "torchvision"),
    "efficientdet_d4": ("models.efficientdet_d4", "effdet"),
}


# Variantes incompatibles avec une famille (vivent ici, à côté du dispatch Modal).
# effdet : inductor explose sur le BiFPN (temps de compile monstrueux) →
# zone_trt_folded est l'alternative TRT pour le BiFPN.
_INCOMPATIBLE_VARIANTS = {
    "effdet": {"compile", "compile_fp16", "zone_compile"},
}


def variant_names_for(family: str) -> list[str]:
    """Noms des variantes pertinentes pour `family`.

    Source de vérité : FULL_VARIANTS + DEFAULT_VARIANTS du runner (pas de
    duplication d'inventaire ici). Filtrage des incompatibles via la table
    `_INCOMPATIBLE_VARIANTS`.
    """
    from optimizations.runner import FULL_VARIANTS, DEFAULT_VARIANTS
    seen, names = set(), []
    for v in (*FULL_VARIANTS, *DEFAULT_VARIANTS):
        if v.name not in seen:
            seen.add(v.name); names.append(v.name)
    drop = _INCOMPATIBLE_VARIANTS.get(family, set())
    return [n for n in names if n not in drop]


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaire : Tee stdout+stderr vers fichier du Volume
# ══════════════════════════════════════════════════════════════════════════════

class _Tee:
    """Multiplexeur d'écriture vers plusieurs flux. La file de log est
    line-buffered (`buffering=1`) → pas besoin de flusher à chaque write."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()
    def isatty(self):
        return False


from contextlib import contextmanager

@contextmanager
def _tee_to(log_path: Path):
    """Duplique sys.stdout/stderr vers log_path. Contexte → restauration auto."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w", encoding="utf-8", buffering=1)   # line-buffered
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, fh)
    sys.stderr = _Tee(orig_err, fh)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = orig_out, orig_err
        fh.close()


# ══════════════════════════════════════════════════════════════════════════════
# Setup COCO (CPU only, ~3 min, idempotent — court-circuite si déjà là)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image, volumes={"/data": volume},
    cpu=2.0, memory=4096, timeout=1800,
)
def ensure_coco():
    """Télécharge COCO val2017 + annotations dans le Volume si absent."""
    import urllib.request, zipfile
    from pathlib import Path

    Path(COCO_DIR).mkdir(parents=True, exist_ok=True)
    val_dir, ann_file = Path(IMG_DIR), Path(ANN_FILE)

    if val_dir.exists() and any(val_dir.iterdir()) and ann_file.exists():
        n = sum(1 for _ in val_dir.iterdir())
        print(f"[COCO] Déjà présent ({n} images, {ann_file}).")
        return

    coco_dir = Path(COCO_DIR)
    if not ann_file.exists():
        z = coco_dir / "annotations.zip"
        print("[COCO] Téléchargement annotations...")
        urllib.request.urlretrieve(COCO_ANN_URL, z)
        with zipfile.ZipFile(z) as zf: zf.extractall(coco_dir)
        z.unlink()

    if not (val_dir.exists() and any(val_dir.iterdir())):
        z = coco_dir / "val2017.zip"
        print("[COCO] Téléchargement val2017 (~1 GB)...")
        urllib.request.urlretrieve(COCO_IMAGES_URL, z)
        with zipfile.ZipFile(z) as zf: zf.extractall(coco_dir)
        z.unlink()

    volume.commit()
    print("[COCO] OK.")


# ══════════════════════════════════════════════════════════════════════════════
# Job principal : un (modèle, variante) — T4 16 GB
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    gpu="T4",                     # 16 GB VRAM
    cpu=8.0,                      # CPU généreux : éviter le goulot preprocess
    memory=16 * 1024,             # 32 GB RAM
    volumes={"/data": volume},
    timeout=3600,                 # 1 h max par job
)
def run_one(
    model_name: str,
    variant_name: str,
    run_id: str,
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    n_profile: int = N_PROFILE,
    n_profile_data: int = N_PROFILE_DATA,
    n_eval: int = N_EVAL,
):
    """Exécute un seul couple (modèle, variante) sur T4 16 GB.

    Sortie : dict {model, variant, status, mean_ms, fps, speedup, AP, ...}
             Sauvegardes complètes dans /data/results/<run_id>/.
    """
    import os, sys, importlib
    from pathlib import Path

    # Code projet + caches partagés (R50 téléchargé une fois pour tous les conteneurs)
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
    Path(TORCH_CACHE).mkdir(parents=True, exist_ok=True)
    Path(HF_CACHE).mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"]            = TORCH_CACHE
    os.environ["HF_HOME"]               = HF_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE

    log_path = Path(f"/data/results/{run_id}/logs/{model_name}_{variant_name}.log")

    with _tee_to(log_path):
        try:
            # Préfixe : toutes les sorties du runner vont dans le Volume
            from optimizations.paths import set_prefix, describe
            set_prefix("/data")
            print(describe())

            # ── Diagnostic environnement (versions TRT chargées paresseusement) ─
            import torch
            print("=" * 78)
            print(f"JOB       : {model_name} :: {variant_name}    run_id={run_id}")
            print(f"GPU       : {torch.cuda.get_device_name(0)}"
                  f"  (capability {torch.cuda.get_device_capability(0)})")
            print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
            print(f"PyTorch   : {torch.__version__}  |  CUDA : {torch.version.cuda}")
            # Importer torch_tensorrt/tensorrt seulement si la variante en a besoin
            # (économise ~3 s d'import par conteneur sur les ~10 variantes non-TRT).
            if "trt" in variant_name:
                for mod_name, label in [("torch_tensorrt", "torch_TRT"),
                                        ("tensorrt", "TensorRT ")]:
                    try:
                        m = importlib.import_module(mod_name)
                        print(f"{label} : {m.__version__}")
                    except Exception as e:
                        print(f"{label} : ABSENT ({e})")
            print("=" * 78)

            # ── Charger la VariantSpec demandée (dict-merge, dernière source gagne) ─
            from optimizations.runner import FULL_VARIANTS, DEFAULT_VARIANTS, RunConfig
            from optimizations import OptimizationRunner, ModelSpec, detect

            all_variants = {v.name: v for v in (*DEFAULT_VARIANTS, *FULL_VARIANTS)}
            chosen = all_variants.get(variant_name)
            if chosen is None:
                raise ValueError(f"Variante inconnue : {variant_name}")

            # ── Charger le modèle ──────────────────────────────────────────────
            if model_name not in MODELS:
                raise ValueError(f"Modèle inconnu : {model_name}")
            mod_path, family = MODELS[model_name]
            mod = importlib.import_module(mod_path)
            mspec = ModelSpec(model_name, mod, family, has_map=True)

            # ── Données (COCO chargé une seule fois → coco_gt réutilisé) ───────
            from pycocotools.coco import COCO
            from utils.data_loader import load_profiling_data, load_eval_data
            profile_data = load_profiling_data(IMG_DIR, ANN_FILE,
                                               n=max(n_profile_data, n_warmup + n_measure))
            eval_data, _ = load_eval_data(IMG_DIR, ANN_FILE, n=n_eval)
            coco_gt      = COCO(ANN_FILE)
            print(f"Données   : profil={len(profile_data)}  eval={len(eval_data)}")

            # ── Config + run (1 modèle × 1 variante) ───────────────────────────
            caps = detect()
            config = RunConfig(
                n_warmup=n_warmup, n_measure=n_measure, n_profile=n_profile,
                do_profile=True, device="cuda",
                compile_backend=caps.compile_backend,
                trt_available=caps.flags["tensorrt"], do_int8=False,
            )
            runner = OptimizationRunner(
                profile_data, eval_data, coco_gt,
                config=config, run_subdir=f"results/{run_id}",
            )
            runner.run_model(mspec, [chosen])

            # Retour pour l'agrégation
            if runner.results:
                r = runner.results[-1]
                return {
                    "model": model_name, "variant": variant_name,
                    "status":   r.get("status"),
                    "mean_ms":  r.get("mean_ms"),
                    "fps":      r.get("fps"),
                    "speedup":  r.get("speedup"),
                    "AP":       r.get("AP"),
                    "duration_s": r.get("duration_s"),
                    "log":      str(log_path),
                }
            return {"model": model_name, "variant": variant_name, "status": "NO_RESULT"}

        except Exception as e:
            # Traceback sauvegardé pour analyse hors-ligne
            err_dir = Path(f"/data/results/{run_id}/errors")
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / f"{model_name}_{variant_name}.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            return {
                "model": model_name, "variant": variant_name,
                "status": "CONTAINER_FAILED",
                "error":  f"{type(e).__name__}: {e}",
                "log":    str(log_path),
            }

        finally:
            # Un seul commit, qui couvre les deux chemins (success + except).
            try: volume.commit()
            except Exception: pass


# ══════════════════════════════════════════════════════════════════════════════
# Agrégation des résultats — produit results.csv après toutes les variantes
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    image=image, volumes={"/data": volume},
    cpu=1.0, memory=2048, timeout=600,
)
def aggregate(run_id: str):
    """Assemble results.csv à partir des bench/<model>_<variant>.json."""
    import csv, json
    from pathlib import Path

    run_dir = Path(f"/data/results/{run_id}")
    if not run_dir.exists():
        print(f"[AGGREGATE] {run_dir} introuvable.")
        return

    # Catalogue inverse {nom_de_fichier: (model, variant)} via MODELS pour
    # split fiable (certaines variantes contiennent un underscore).
    known_models = sorted(MODELS, key=len, reverse=True)   # plus long d'abord

    def split(name: str):
        for m in known_models:
            if name.startswith(m + "_"):
                return m, name[len(m) + 1:]
        return name, ""

    rows = []
    for bench_path in sorted((run_dir / "bench").glob("*.json")):
        name = bench_path.stem
        try:
            bench = json.loads(bench_path.read_text())
        except Exception:
            continue
        eval_path = run_dir / "eval" / f"{name}.json"
        ap = ap50 = ap75 = ""
        if eval_path.exists():
            try:
                ev = json.loads(eval_path.read_text())
                ap, ap50, ap75 = ev.get("AP", ""), ev.get("AP50", ""), ev.get("AP75", "")
            except Exception: pass
        model, variant = split(name)
        rows.append({
            "model": model, "variant": variant,
            "mean_ms": bench.get("mean_ms"),
            "std_ms":  bench.get("std_ms"),
            "fps":     bench.get("fps"),
            "AP": ap, "AP50": ap50, "AP75": ap75,
        })

    csv_path = run_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    volume.commit()
    print(f"[AGGREGATE] {len(rows)} lignes -> {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Local entrypoint : soumet toutes les combinaisons en parallèle
# ══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def main(
    models: str = "",                     # ex: "retinanet_r50,fcos_r50"
    variants: str = "",                   # ex: "baseline,fp16,zone_trt_fp16"
    parallel: int = 30,                   # ≤ conteneurs simultanés (défaut généreux ;
                                          # le coût total est le même quel que soit ce nombre,
                                          # seul le temps wall diminue)
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    n_profile: int = N_PROFILE,
    n_profile_data: int = N_PROFILE_DATA,
    n_eval: int = N_EVAL,
):
    """Lance tous les (modèle, variante) sélectionnés, en parallèle, sur T4."""
    from datetime import datetime

    sel_models = [m.strip() for m in models.split(",") if m.strip()] or list(MODELS)
    user_variants = {v.strip() for v in variants.split(",") if v.strip()}

    # ── Étape 1 : COCO ─────────────────────────────────────────────────────────
    print("[1/3] COCO sur le Volume...")
    ensure_coco.remote()

    # ── Étape 2 : soumettre les jobs ───────────────────────────────────────────
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    for model in sel_models:
        if model not in MODELS:
            print(f"  skip modèle inconnu : {model}"); continue
        family = MODELS[model][1]
        possible = variant_names_for(family)
        if user_variants:
            possible = [v for v in possible if v in user_variants]
        for v in possible:
            jobs.append((model, v, run_id,
                        n_warmup, n_measure, n_profile, n_profile_data, n_eval))

    if not jobs:
        print("Aucun job à lancer."); return

    print(f"[2/3] {len(jobs)} jobs (modèle, variante) sur T4, "
          f"max {parallel} en parallèle, run_id={run_id}")
    for model, variant, *_ in jobs:
        print(f"   - {model:18s} :: {variant}")

    # starmap : exécute les jobs en parallèle (chaque conteneur indépendant)
    statuses = []
    for result in run_one.starmap(jobs, return_exceptions=True):
        if isinstance(result, Exception):
            statuses.append({"status": "CONTAINER_EXCEPTION", "error": str(result)})
        else:
            statuses.append(result)

    # ── Étape 3 : agrégation des résultats (SYNCHRONE — on attend) ─────────────
    # `.remote()` (et non `.spawn()`) bloque jusqu'à ce que l'agrégation finisse.
    # CRUCIAL : avec `--detach`, le local entrypoint exit dès que ce print sort,
    # donc on doit s'assurer que tous les jobs ont vraiment fini ET que l'agrégat
    # a tourné. Si on faisait `.spawn()`, aggregate démarrerait en parallèle des
    # jobs et écrirait un results.csv vide (cas vécu run 20260616_044124).
    print(f"\n[3/3] Agrégation des résultats (attente)...")
    aggregate.remote(run_id)

    # ── Résumé console ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  RUN {run_id} — RÉSUMÉ")
    print("=" * 78)
    by_status = {}
    for s in statuses:
        by_status.setdefault(s.get("status", "?"), []).append(s)
    for st, items in sorted(by_status.items()):
        print(f"  {st:20s} : {len(items)}")
        for it in items[:30]:
            tag = f"{it.get('model','?'):18s} :: {it.get('variant','?')}"
            extra = ""
            if it.get("mean_ms") not in (None, ""):
                extra = f"  {it['mean_ms']} ms  x{it.get('speedup','')}"
            if it.get("error"):
                extra = f"  ! {it['error'][:60]}"
            print(f"    - {tag}{extra}")
    print("\nArtefacts : /data/results/" + run_id)
    print(f"Récupérer : modal volume get {VOLUME_NAME} results/{run_id} ./local_run/")


# Documentation des variantes : voir le docstring du module en tête de fichier
# (lignes 1-62) — la liste détaillée et le glossaire sont dans docs/runs_modal.md.
