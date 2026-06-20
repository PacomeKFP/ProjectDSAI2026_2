"""
modal_runner.py
===============
Modal orchestration -- 1 T4 (16 GB VRAM) container per (model, variant).

--------------------------------------------------------------------------------
DESIGN
--------------------------------------------------------------------------------

  * Image : Debian slim + Python 3.13 + everything from PyPI (validated by
            modal_test_env.py). pip resolves torch 2.8+cu128, torch-tensorrt 2.8,
            tensorrt 10.12, numpy 2.4, cv2 4.13 in a consistent way -- no
            numpy 1.x/2.x conflict. ALL variants (TRT included) are active.

  * Volume `dsai2026` (50 GB+), mounted at /data:
        /data/coco/                      COCO val2017 dataset (downloaded once)
        /data/cache/torch                shared TORCH_HOME (R50/FCOS DL once)
        /data/cache/hf                   HuggingFace cache (timm for effdet)
        /data/results/<run_id>/
            bench/<model>_<variant>.json
            eval/<model>_<variant>.json
            modules/<model>_<variant>.csv
            profiles/<model>_<variant>.csv
            errors/<model>_<variant>.txt   tracebacks on failure
            logs/<model>_<variant>.log    tee stdout+stderr from the container
            results.csv                    aggregated after the run

  * Granularity: 1 Modal function = 1 (model, variant) pair. In parallel via
    `.starmap()`. Advantages:
      - cross-variant state pollution = impossible (different processes)
      - heartbeat timeouts avoided (each job is short and isolated)
      - targeted re-run of a single pair is free

  * Logs: Modal captures stdout/stderr automatically per container, and we
    also mirror to the Volume via Tee (one `.log` file per job).

--------------------------------------------------------------------------------
DEFAULT PARAMETERS (per spec)
--------------------------------------------------------------------------------
  N_PROFILE_DATA = 2000   loaded images (bench + profiling)
  N_WARMUP       = 50     warm-up iterations
  N_MEASURE      = 1000   active benchmark iterations
  N_PROFILE      = 150    active profiling iterations
  N_EVAL         = 2000   images for MAP@640

--------------------------------------------------------------------------------
COMMANDS
--------------------------------------------------------------------------------
  modal run modal_runner.py                                # everything (parallel)
  modal run modal_runner.py --models retinanet_r50         # a single model
  modal run modal_runner.py --variants baseline,fp16,zone_trt_fp16
  modal run modal_runner.py --parallel 4                   # <=4 concurrent containers
  modal run modal_runner.py::ensure_coco                   # just download COCO
  modal run modal_runner.py::aggregate --run-id 20260616_xxxxxx   # re-aggregate

Fetch a run:
  modal volume get dsai2026 results/<run_id> ./local_run/

Browse the Volume:
  modal volume ls dsai2026 results/
  modal shell --volume dsai2026:/data
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import modal


# ==============================================================================
# Configuration
# ==============================================================================

APP_NAME      = "dsai2026-opt"
VOLUME_NAME   = "dsai2026"
PROJECT_ROOT  = "/workspace/project"

# Data
COCO_DIR      = "/data/coco"
IMG_DIR       = f"{COCO_DIR}/val2017"
ANN_FILE      = f"{COCO_DIR}/annotations/instances_val2017.json"
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Shared caches (model downloads)
TORCH_CACHE   = "/data/cache/torch"
HF_CACHE      = "/data/cache/hf"

# Default experiment parameters
N_PROFILE_DATA = 2000
N_WARMUP       = 50
N_MEASURE      = 1000
N_PROFILE      = 150
N_EVAL         = 2000


# ==============================================================================
# Image: EXACT COPY of the A100 run that worked
# ==============================================================================
#
# Honest history:
#
#   * A100 run (partial success -- this is the BASE we copy here):
#       NGC 24.10 + add_python="3.11" + pip install (effdet, timm,
#       pycocotools, opencv-python-headless, pillow, pandas, numpy, psutil,
#       tqdm, tensorboard, nvtx, onnx, onnxruntime-gpu, onnxsim).
#       Result: baseline, fp16, compile_fp16 -> OK. trt_fp16 -> SKIPPED
#       (TRT unavailable because add_python replaces native NGC Python; the
#       torch_tensorrt/tensorrt packages preinstalled for Python 3.10 are no
#       longer reachable from Python 3.11).
#
#   * T4 run #1 (no-add_python + --no-deps): dep install FAILED.
#
#   * T4 run #2 (add_python + explicit torch-tensorrt): install FAILED
#       (torch-tensorrt version conflicts vs torch pulled by effdet).
#
#   * Present version: WE LITERALLY COPY THE A100 RUN. TRT will remain
#     UNAVAILABLE (the *trt* variants will be SKIPPED by the runner via
#     `requires="trt"`), but everything else runs:
#       baseline, fp16, torchscript, compile, cudagraphs, compile_fp16,
#       cudagraphs_fp16, torchscript_fp16, and every non-TRT zone variant
#       (zone_torchscript, zone_compile, zone_cudagraphs).
#
#     -> 11 working variants per model out of 15 total.
#     -> To get TRT later, we'll need a different base (NGC containers without
#       add_python + a custom image).
#
# DO NOT TOUCH THIS LIST -- it matches the A100 run verbatim.

image = (
    # Vanilla Debian + Python 3.13 + everything from PyPI.
    # Validated: torch 2.8+cu128, torch-tensorrt 2.8, tensorrt 10.12, numpy 2.4,
    # cv2 4.13, effdet, timm, onnx, onnxruntime-gpu -- no numpy conflict.
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # PyTorch CUDA bundled (PyPI wheels)
        "torch", "torchvision",
        # TensorRT stack -- pip resolves consistently with torch
        "tensorrt", "torch-tensorrt",
        # Detection
        "effdet", "timm",
        # Data / eval
        "pycocotools", "opencv-python-headless", "pillow",
        # Utilities
        "pandas", "numpy", "psutil", "tqdm",
        # Profiling
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
# Volume created on demand, single 1 TB size included in the Starter plan
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# ==============================================================================
# Mapping: models & variants
# ==============================================================================

MODELS = {
    "retinanet_r50":   ("models.retinanet_r50",   "torchvision"),
    "fcos_r50":        ("models.fcos_r50",        "torchvision"),
    "efficientdet_d4": ("models.efficientdet_d4", "effdet"),
}


# Variants incompatible with a given family (live here, next to the Modal dispatch).
# effdet: inductor explodes on the BiFPN (monstrous compile time) -> zone_trt_folded
# is the TRT alternative for the BiFPN.
_INCOMPATIBLE_VARIANTS = {
    "effdet": {"compile", "compile_fp16", "zone_compile"},
}


def variant_names_for(family: str) -> list[str]:
    """Names of the variants relevant to `family`.

    Source of truth: the runner's FULL_VARIANTS + DEFAULT_VARIANTS (no
    inventory duplication here). Filtering of incompatibles via the
    `_INCOMPATIBLE_VARIANTS` table.
    """
    from optimizations.runner import FULL_VARIANTS, DEFAULT_VARIANTS
    seen, names = set(), []
    for v in (*FULL_VARIANTS, *DEFAULT_VARIANTS):
        if v.name not in seen:
            seen.add(v.name); names.append(v.name)
    drop = _INCOMPATIBLE_VARIANTS.get(family, set())
    return [n for n in names if n not in drop]


# ==============================================================================
# Utility: Tee stdout+stderr to a file on the Volume
# ==============================================================================

class _Tee:
    """Write multiplexer to several streams. The log file is line-buffered
    (`buffering=1`) -> no need to flush on every write."""
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
    """Duplicate sys.stdout/stderr to log_path. Context manager -> auto-restore."""
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


# ==============================================================================
# COCO setup (CPU only, ~3 min, idempotent -- short-circuits if already there)
# ==============================================================================

@app.function(
    image=image, volumes={"/data": volume},
    cpu=2.0, memory=4096, timeout=1800,
)
def ensure_coco():
    """Download COCO val2017 + annotations into the Volume if absent."""
    import urllib.request, zipfile
    from pathlib import Path

    Path(COCO_DIR).mkdir(parents=True, exist_ok=True)
    val_dir, ann_file = Path(IMG_DIR), Path(ANN_FILE)

    if val_dir.exists() and any(val_dir.iterdir()) and ann_file.exists():
        n = sum(1 for _ in val_dir.iterdir())
        print(f"[COCO] Already present ({n} images, {ann_file}).")
        return

    coco_dir = Path(COCO_DIR)
    if not ann_file.exists():
        z = coco_dir / "annotations.zip"
        print("[COCO] Downloading annotations...")
        urllib.request.urlretrieve(COCO_ANN_URL, z)
        with zipfile.ZipFile(z) as zf: zf.extractall(coco_dir)
        z.unlink()

    if not (val_dir.exists() and any(val_dir.iterdir())):
        z = coco_dir / "val2017.zip"
        print("[COCO] Downloading val2017 (~1 GB)...")
        urllib.request.urlretrieve(COCO_IMAGES_URL, z)
        with zipfile.ZipFile(z) as zf: zf.extractall(coco_dir)
        z.unlink()

    volume.commit()
    print("[COCO] OK.")


# ==============================================================================
# Main job: one (model, variant) -- T4 16 GB
# ==============================================================================

@app.function(
    image=image,
    gpu="T4",                     # 16 GB VRAM
    cpu=8.0,                      # generous CPU: avoid preprocessing bottleneck
    memory=16 * 1024,             # 32 GB RAM
    volumes={"/data": volume},
    timeout=3600,                 # 1 h max per job
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
    """Run a single (model, variant) pair on T4 16 GB.

    Output: dict {model, variant, status, mean_ms, fps, speedup, AP, ...}
            Full artifacts saved in /data/results/<run_id>/.
    """
    import os, sys, importlib
    from pathlib import Path

    # Project code + shared caches (R50 downloaded once for all containers)
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
            # Prefix: every runner output goes to the Volume
            from optimizations.paths import set_prefix, describe
            set_prefix("/data")
            print(describe())

            # -- Environment diagnostic (TRT versions loaded lazily) -----------
            import torch
            print("=" * 78)
            print(f"JOB       : {model_name} :: {variant_name}    run_id={run_id}")
            print(f"GPU       : {torch.cuda.get_device_name(0)}"
                  f"  (capability {torch.cuda.get_device_capability(0)})")
            print(f"VRAM      : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
            print(f"PyTorch   : {torch.__version__}  |  CUDA : {torch.version.cuda}")
            # Import torch_tensorrt/tensorrt only if the variant needs them
            # (saves ~3 s of import per container on the ~10 non-TRT variants).
            if "trt" in variant_name:
                for mod_name, label in [("torch_tensorrt", "torch_TRT"),
                                        ("tensorrt", "TensorRT ")]:
                    try:
                        m = importlib.import_module(mod_name)
                        print(f"{label} : {m.__version__}")
                    except Exception as e:
                        print(f"{label} : ABSENT ({e})")
            print("=" * 78)

            # -- Load the requested VariantSpec (dict-merge, last source wins) -
            from optimizations.runner import FULL_VARIANTS, DEFAULT_VARIANTS, RunConfig
            from optimizations import OptimizationRunner, ModelSpec, detect

            all_variants = {v.name: v for v in (*DEFAULT_VARIANTS, *FULL_VARIANTS)}
            chosen = all_variants.get(variant_name)
            if chosen is None:
                raise ValueError(f"Unknown variant: {variant_name}")

            # -- Load the model -----------------------------------------------
            if model_name not in MODELS:
                raise ValueError(f"Unknown model: {model_name}")
            mod_path, family = MODELS[model_name]
            mod = importlib.import_module(mod_path)
            mspec = ModelSpec(model_name, mod, family, has_map=True)

            # -- Data (COCO loaded once -> coco_gt reused) ---------------------
            from pycocotools.coco import COCO
            from utils.data_loader import load_profiling_data, load_eval_data
            profile_data = load_profiling_data(IMG_DIR, ANN_FILE,
                                               n=max(n_profile_data, n_warmup + n_measure))
            eval_data, _ = load_eval_data(IMG_DIR, ANN_FILE, n=n_eval)
            coco_gt      = COCO(ANN_FILE)
            print(f"Data      : profile={len(profile_data)}  eval={len(eval_data)}")

            # -- Config + run (1 model x 1 variant) ---------------------------
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

            # Return for aggregation
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
            # Traceback saved for offline analysis
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
            # A single commit, covering both paths (success + except).
            try: volume.commit()
            except Exception: pass


# ==============================================================================
# Result aggregation -- produces results.csv after all variants
# ==============================================================================

@app.function(
    image=image, volumes={"/data": volume},
    cpu=1.0, memory=2048, timeout=600,
)
def aggregate(run_id: str):
    """Assemble results.csv from the bench/<model>_<variant>.json files."""
    import csv, json
    from pathlib import Path

    run_dir = Path(f"/data/results/{run_id}")
    if not run_dir.exists():
        print(f"[AGGREGATE] {run_dir} not found.")
        return

    # Reverse catalog {filename: (model, variant)} via MODELS for a reliable
    # split (some variants contain an underscore).
    known_models = sorted(MODELS, key=len, reverse=True)   # longest first

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
    print(f"[AGGREGATE] {len(rows)} rows -> {csv_path}")


# ==============================================================================
# Local entrypoint: submit every combination in parallel
# ==============================================================================

@app.local_entrypoint()
def main(
    models: str = "",                     # e.g. "retinanet_r50,fcos_r50"
    variants: str = "",                   # e.g. "baseline,fp16,zone_trt_fp16"
    parallel: int = 30,                   # <= concurrent containers (generous default;
                                          # total cost is the same regardless of this
                                          # number -- only wall-clock time changes)
    n_warmup: int = N_WARMUP,
    n_measure: int = N_MEASURE,
    n_profile: int = N_PROFILE,
    n_profile_data: int = N_PROFILE_DATA,
    n_eval: int = N_EVAL,
):
    """Launch every selected (model, variant), in parallel, on T4."""
    from datetime import datetime

    sel_models = [m.strip() for m in models.split(",") if m.strip()] or list(MODELS)
    user_variants = {v.strip() for v in variants.split(",") if v.strip()}

    # -- Step 1: COCO -----------------------------------------------------------
    print("[1/3] COCO on the Volume...")
    ensure_coco.remote()

    # -- Step 2: submit the jobs ------------------------------------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []
    for model in sel_models:
        if model not in MODELS:
            print(f"  skipping unknown model: {model}"); continue
        family = MODELS[model][1]
        possible = variant_names_for(family)
        if user_variants:
            possible = [v for v in possible if v in user_variants]
        for v in possible:
            jobs.append((model, v, run_id,
                        n_warmup, n_measure, n_profile, n_profile_data, n_eval))

    if not jobs:
        print("No job to launch."); return

    print(f"[2/3] {len(jobs)} (model, variant) jobs on T4, "
          f"up to {parallel} in parallel, run_id={run_id}")
    for model, variant, *_ in jobs:
        print(f"   - {model:18s} :: {variant}")

    # starmap: runs the jobs in parallel (each container is independent)
    statuses = []
    for result in run_one.starmap(jobs, return_exceptions=True):
        if isinstance(result, Exception):
            statuses.append({"status": "CONTAINER_EXCEPTION", "error": str(result)})
        else:
            statuses.append(result)

    # -- Step 3: aggregate the results (SYNCHRONOUS -- we wait) ------------------
    # `.remote()` (not `.spawn()`) blocks until the aggregation completes.
    # CRUCIAL: with `--detach`, the local entrypoint exits as soon as this
    # print returns, so we must make sure that every job has truly finished
    # AND that the aggregation has run. If we used `.spawn()`, aggregate would
    # start in parallel with the jobs and write an empty results.csv (this
    # actually happened in run 20260616_044124).
    print(f"\n[3/3] Aggregating results (waiting)...")
    aggregate.remote(run_id)

    # -- Console summary --------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  RUN {run_id} -- SUMMARY")
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
    print("\nArtifacts: /data/results/" + run_id)
    print(f"Fetch    : modal volume get {VOLUME_NAME} results/{run_id} ./local_run/")


# Variant documentation: see the module docstring at the top of this file
# (lines 1-62) -- the detailed list and glossary are in docs/runs_modal.md.
