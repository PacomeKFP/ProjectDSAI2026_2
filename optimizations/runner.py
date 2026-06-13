"""
optimizations/runner.py
════════════════════════
Orchestrateur d'optimisation : applique en séquence toutes les optimisations
(et combinaisons raisonnables) sur une liste de modèles, avec :

  • Robustesse  : chaque (modèle, variante) est une unité try/except isolée.
                  Un échec est journalisé et N'INTERROMPT PAS le reste.
  • Logs        : tout est écrit dans run.log (console + fichier) et les
                  tracebacks d'erreur dans errors/<model>_<variant>.txt.
  • Sauvegarde  : results.csv réécrit après CHAQUE variante (survit à une
                  déconnexion Colab) ; CSV par module ; modèles optimisés.
  • Testable    : benchmark_impl / map_impl / module_benchmark_factory sont
                  injectables → orchestration testable sans torch.

Décision de conception — MAP à résolution fixe 640×640 :
  Les optimisations sont construites sur load_model() (entrée fixe 640×640).
  L'évaluation MAP réutilise CE MÊME modèle optimisé avec le pipeline benchmark
  (preprocess / collate / postprocess à 640), et NON load_model_eval() (natif).
  Pourquoi : TensorRT et cudagraphs exigent des shapes fixes ; évaluer à 640
  garde benchmark et MAP cohérents et compilables. La MAP@640 est plus basse
  que la MAP COCO native, mais c'est le DELTA baseline→optimisé qui mesure
  l'impact de précision (FP16/INT8), pas la valeur absolue.

Variantes par défaut (DEFAULT_VARIANTS) :
  baseline · fp16 · torchscript · compile · compile_fp16 · trt_fp16 · trt_int8
  (chacune filtrée selon les capacités de l'environnement)
"""

from __future__ import annotations

import csv
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Configuration & specs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    n_warmup:          int   = 50
    n_measure:         int   = 1000
    device:            str   = "cuda"
    compile_backend:   str   = "inductor"   # "inductor" | "cudagraphs" | "eager"
    trt_available:     bool  = False
    do_int8:           bool  = False
    int8_calib_images: int   = 300
    min_block_size:    int   = 5


@dataclass
class ModelSpec:
    name:    str          # "retinanet_r50"
    module:  object       # module importé (models.retinanet_r50)
    family:  str          # "torchvision" | "effdet"
    has_map: bool = True   # False pour r101 (tête aléatoire → MAP ~0)


@dataclass
class VariantSpec:
    name:         str
    build:        Callable          # (model, mspec, ctx) -> model optimisé
    do_map:       bool = False
    with_modules: bool = False
    requires:     Optional[str] = None   # "cuda" | "compile" | "trt" | "trt+int8"


# ══════════════════════════════════════════════════════════════════════════════
# Builders de variantes (imports paresseux → runner importable sans torch)
# ══════════════════════════════════════════════════════════════════════════════

def _example_input(mspec: ModelSpec, ctx):
    m = mspec.module
    return m.collate([m.preprocess(ctx.profile_data[0])], ctx.config.device)


def build_baseline(model, mspec, ctx):
    return model


def build_fp16(model, mspec, ctx):
    from optimizations import to_fp16_autocast
    return to_fp16_autocast(model)


def build_torchscript(model, mspec, ctx):
    from optimizations import optimize_with_torchscript
    method = "script" if mspec.family == "torchvision" else "trace"
    return optimize_with_torchscript(model, example_input=_example_input(mspec, ctx),
                                     prefer=method)


def build_compile(model, mspec, ctx):
    from optimizations import compile_model
    return compile_model(model, backend=ctx.config.compile_backend,
                         mode="default", dynamic=False)


def build_compile_fp16(model, mspec, ctx):
    from optimizations import to_fp16_autocast, compile_model
    return compile_model(to_fp16_autocast(model), backend=ctx.config.compile_backend,
                         mode="default", dynamic=False)


def build_trt_fp16(model, mspec, ctx):
    from optimizations import build_trt_fp16 as _b
    return _b(model, min_block_size=ctx.config.min_block_size)


def build_trt_int8(model, mspec, ctx):
    from optimizations.tensorrt_int8 import build_trt_int8 as _b, build_calibration_loader
    calib = build_calibration_loader(
        ctx.profile_data, mspec.module.preprocess,
        n_images=ctx.config.int8_calib_images, batch_size=4,
    )
    return _b(model, calib)


DEFAULT_VARIANTS: List[VariantSpec] = [
    VariantSpec("baseline",     build_baseline,     do_map=True,  with_modules=True,  requires=None),
    VariantSpec("fp16",         build_fp16,         do_map=True,  with_modules=True,  requires="cuda"),
    VariantSpec("torchscript",  build_torchscript,  do_map=False, with_modules=False, requires=None),
    VariantSpec("compile",      build_compile,      do_map=False, with_modules=False, requires="compile"),
    VariantSpec("compile_fp16", build_compile_fp16, do_map=False, with_modules=False, requires="compile"),
    VariantSpec("trt_fp16",     build_trt_fp16,     do_map=True,  with_modules=False, requires="trt"),
    VariantSpec("trt_int8",     build_trt_int8,     do_map=True,  with_modules=False, requires="trt+int8"),
]


# ══════════════════════════════════════════════════════════════════════════════
# Implémentations réelles (imports paresseux)
# ══════════════════════════════════════════════════════════════════════════════

def _real_benchmark_impl(model, data, preprocess, collate, n_warmup, n_measure,
                         device, module_benchmark):
    from utils.benchmark import benchmark_model
    return benchmark_model(model, data, preprocess, collate,
                           n_warmup=n_warmup, n_measure=n_measure,
                           device=device, module_benchmark=module_benchmark)


def _real_map_impl(model, data, coco_gt, preprocess, collate, postprocess, device):
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess, collate, postprocess, device)


def _real_mb_factory():
    from utils.benchmark import ModuleBenchmark
    return ModuleBenchmark()


def _free_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

_RESULT_FIELDS = [
    "model", "variant", "status", "mean_ms", "std_ms", "fps", "speedup",
    "AP", "AP50", "AP75", "n_modules", "duration_s", "error",
]


class OptimizationRunner:
    def __init__(
        self,
        profile_data,
        eval_data,
        coco_gt,
        config: RunConfig,
        run_dir: str,
        benchmark_impl: Optional[Callable] = None,
        map_impl: Optional[Callable] = None,
        module_benchmark_factory: Optional[Callable] = None,
    ):
        self.profile_data = profile_data
        self.eval_data    = eval_data
        self.coco_gt      = coco_gt
        self.config       = config

        self.benchmark_impl = benchmark_impl or _real_benchmark_impl
        self.map_impl       = map_impl or _real_map_impl
        self.mb_factory     = module_benchmark_factory or _real_mb_factory

        self.run_dir = Path(run_dir)
        (self.run_dir / "errors").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "modules").mkdir(parents=True, exist_ok=True)

        self.results: List[dict] = []
        self._baseline_ms: dict[str, float] = {}   # model -> baseline mean_ms

        self.logger = self._setup_logger()
        self.logger.info(f"Run dir : {self.run_dir}")
        self.logger.info(f"Config  : {json.dumps(asdict(config))}")

    # ── Logging ────────────────────────────────────────────────────────────────

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"optrunner.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        return logger

    # ── Capacités ───────────────────────────────────────────────────────────────

    def _capable(self, requires: Optional[str]) -> tuple[bool, str]:
        c = self.config
        if requires is None:
            return True, ""
        if requires == "cuda":
            return (c.device == "cuda"), "CUDA requis"
        if requires == "compile":
            return (c.compile_backend != "eager"), "backend compile indisponible"
        if requires == "trt":
            return c.trt_available, "TensorRT indisponible"
        if requires == "trt+int8":
            if not c.trt_available:
                return False, "TensorRT indisponible"
            if not c.do_int8:
                return False, "INT8 désactivé (config.do_int8=False)"
            return True, ""
        return True, ""

    # ── Persistance ──────────────────────────────────────────────────────────────

    def _record(self, **kw):
        row = {f: kw.get(f, "") for f in _RESULT_FIELDS}
        self.results.append(row)
        self._save_csv()

    def _save_csv(self):
        path = self.run_dir / "results.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            w.writeheader()
            w.writerows(self.results)

    def _save_modules(self, model_name, variant, bench):
        mods = bench.get("modules") if isinstance(bench, dict) else None
        if mods is None:
            return 0
        try:
            path = self.run_dir / "modules" / f"{model_name}_{variant}.csv"
            mods.to_csv(path, index=False)
            return len(mods)
        except Exception as e:
            self.logger.warning(f"  Sauvegarde modules échouée : {e}")
            return 0

    def _save_error(self, model_name, variant, exc: Exception):
        path = self.run_dir / "errors" / f"{model_name}_{variant}.txt"
        path.write_text("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                        encoding="utf-8")

    # ── Exécution d'une variante ────────────────────────────────────────────────

    def run_variant(self, mspec: ModelSpec, spec: VariantSpec):
        tag = f"{mspec.name} · {spec.name}"
        ok, reason = self._capable(spec.requires)
        if not ok:
            self.logger.info(f"[SKIP]   {tag}  ({reason})")
            self._record(model=mspec.name, variant=spec.name, status="SKIPPED", error=reason)
            return

        self.logger.info(f">> {tag}")
        t0 = time.time()
        model = optimized = None
        try:
            model = mspec.module.load_model(self.config.device)
            optimized = spec.build(model, mspec, self)

            mb = self.mb_factory() if spec.with_modules else None
            bench = self.benchmark_impl(
                optimized, self.profile_data,
                mspec.module.preprocess, mspec.module.collate,
                self.config.n_warmup, self.config.n_measure,
                self.config.device, mb,
            )
            mean_ms = float(bench["mean_ms"])
            fps     = float(bench["fps"])
            n_mods  = self._save_modules(mspec.name, spec.name, bench)

            if spec.name == "baseline":
                self._baseline_ms[mspec.name] = mean_ms
            base = self._baseline_ms.get(mspec.name)
            speedup = round(base / mean_ms, 3) if base else ""

            self.logger.info(f"   bench : {mean_ms:.2f} ms  |  {fps:.1f} FPS"
                             + (f"  |  x{speedup}" if speedup else ""))

            ap = ap50 = ap75 = ""
            if spec.do_map and mspec.has_map:
                self.logger.info(f"   MAP@640 ({len(self.eval_data)} images)...")
                maps = self.map_impl(
                    optimized, self.eval_data, self.coco_gt,
                    mspec.module.preprocess, mspec.module.collate,
                    mspec.module.postprocess, self.config.device,
                )
                ap, ap50, ap75 = (round(maps[k], 4) for k in ("AP", "AP50", "AP75"))
                self.logger.info(f"   AP={ap}  AP50={ap50}  AP75={ap75}")
            elif spec.do_map and not mspec.has_map:
                self.logger.info("   MAP ignoree (tete non entrainee)")

            dur = round(time.time() - t0, 1)
            self.logger.info(f"   OK  ({dur}s)")
            self._record(model=mspec.name, variant=spec.name, status="OK",
                         mean_ms=round(mean_ms, 3), std_ms=round(float(bench["std_ms"]), 3),
                         fps=round(fps, 2), speedup=speedup,
                         AP=ap, AP50=ap50, AP75=ap75, n_modules=n_mods, duration_s=dur)

        except Exception as e:
            dur = round(time.time() - t0, 1)
            self._save_error(mspec.name, spec.name, e)
            short = f"{type(e).__name__}: {e}"[:200]
            self.logger.error(f"   FAILED ({dur}s) : {short}")
            self._record(model=mspec.name, variant=spec.name, status="FAILED",
                         duration_s=dur, error=short)
        finally:
            del model, optimized
            _free_gpu()

    # ── Boucles ──────────────────────────────────────────────────────────────────

    def run_model(self, mspec: ModelSpec, variants: Optional[List[VariantSpec]] = None):
        variants = variants or DEFAULT_VARIANTS
        self.logger.info("=" * 70)
        self.logger.info(f"MODÈLE : {mspec.name}  (famille={mspec.family}, MAP={mspec.has_map})")
        self.logger.info("=" * 70)
        for spec in variants:
            self.run_variant(mspec, spec)

    def run_all(self, mspecs: List[ModelSpec], variants: Optional[List[VariantSpec]] = None):
        for mspec in mspecs:
            self.run_model(mspec, variants)
        self.logger.info("TERMINÉ — %d lignes de résultats.", len(self.results))
        return self.results

    # ── Agrégation ───────────────────────────────────────────────────────────────

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.results, columns=_RESULT_FIELDS)

    def speedup_table(self):
        """Tableau speedup robuste : ignore les modèles sans baseline OK."""
        import pandas as pd
        df = self.to_dataframe()
        df_ok = df[df["status"] == "OK"].copy()
        rows = []
        for model in df_ok["model"].unique():
            sub = df_ok[df_ok["model"] == model]
            base = sub[sub["variant"] == "baseline"]
            if base.empty or base["mean_ms"].values[0] in ("", None):
                continue                          # garde : pas de baseline → on saute
            base_ms = float(base["mean_ms"].values[0])
            for _, r in sub.iterrows():
                try:
                    cur = float(r["mean_ms"])
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "model": model, "variant": r["variant"],
                    "mean_ms": r["mean_ms"], "fps": r["fps"],
                    "speedup": round(base_ms / cur, 2) if cur else "",
                    "AP": r["AP"], "AP50": r["AP50"],
                })
        return pd.DataFrame(rows)
