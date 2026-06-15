"""
optimizations/runner.py
════════════════════════
Orchestrateur d'optimisation CONSCIENTE DE L'ARCHITECTURE.

Pour chaque modèle et chaque variante :
  1. construit le modèle optimisé (optimisation par ZONE — voir zones.py :
     seule la zone statique backbone/FPN est optimisée, le NMS reste en eager) ;
  2. BENCHMARK (vitesse GPU + détail par module pour le baseline) ;
  3. ÉVALUATION MAP@640 (uniquement variantes qui changent la précision) ;
  4. PROFILING PyTorch avant/après (kernels, mémoire, opérations) ;
  5. SAUVEGARDE de TOUT, sous le préfixe de sortie (Drive sur Colab).

Robustesse : chaque (modèle, variante) est isolé (try/except) ; un échec est
journalisé (errors/) et n'interrompt pas la suite. results.csv est réécrit
après chaque variante, et une sauvegarde par-modèle est faite à la fin de
chaque modèle → tout survit à une déconnexion Colab.

Sorties (sous <préfixe>/results/optimization/<run_id>/) :
  run.log                         journal complet
  results.csv / results_final.csv tableau récapitulatif (incrémental)
  bench/<model>_<variant>.json    métriques de vitesse brutes
  eval/<model>_<variant>.json     métriques MAP/AR COCO complètes
  modules/<model>_<variant>.csv   timing par module feuille (ModuleBenchmark)
  profiles/<model>_<variant>.csv  table d'opérations (kernels/mémoire) du profiler
  errors/<model>_<variant>.txt    traceback en cas d'échec

Décision MAP@640 : la MAP réutilise le modèle optimisé avec le pipeline 640×640
(preprocess/collate/postprocess), pour rester à shapes fixes et cohérent avec
le benchmark. C'est le DELTA baseline→optimisé qui mesure l'impact FP16/INT8.
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

from .paths import ensure_dir, ensure_parent, project_prefix
from .zones import (
    apply_zone_optimization, apply_subzone_plan, plan_mixed,
    opt_torchscript, opt_compile, opt_cudagraphs,
    opt_trt_fp16, opt_trt_fp16_folded, opt_trt_int8,
)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration & specs
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    n_warmup:          int   = 50
    n_measure:         int   = 1000
    n_profile:         int   = 150        # itérations actives du profiler
    do_profile:        bool  = True
    device:            str   = "cuda"
    compile_backend:   str   = "inductor"   # "inductor" (Colab) | "cudagraphs" (Windows)
    trt_available:     bool  = False
    do_int8:           bool  = False
    int8_calib_images: int   = 300
    min_block_size:    int   = 5
    size:              tuple = (640, 640)


@dataclass
class ModelSpec:
    name:    str
    module:  object
    family:  str          # "torchvision" | "effdet"
    has_map: bool = True


@dataclass
class VariantSpec:
    name:            str
    build:           Callable       # (model, mspec, runner) -> modèle optimisé
    do_map:          bool = False
    with_modules:    bool = False
    profile:         bool = True
    requires:        Optional[str] = None   # "cuda" | "compile" | "trt" | "trt+int8"
    eval_batch_size: Optional[int] = None   # 1 pour les modèles à shape figée (TRT)


# ══════════════════════════════════════════════════════════════════════════════
# Builders de variantes (imports paresseux → runner importable sans torch)
# ══════════════════════════════════════════════════════════════════════════════

# Backends graphe pour _full() — un par clé, lazy-imported pour ne pas tirer
# torch_tensorrt quand on ne fait que `none`/`compile`/etc.

def _graph_noop(model, mspec, ctx):
    return model


def _graph_compile(model, mspec, ctx):
    from optimizations import compile_model
    return compile_model(model, backend=ctx.config.compile_backend,
                         mode="default", dynamic=False)


def _graph_cudagraphs(model, mspec, ctx):
    from optimizations import compile_model
    return compile_model(model, backend="cudagraphs", dynamic=False)


def _graph_torchscript(model, mspec, ctx):
    from optimizations import optimize_with_torchscript
    m = mspec.module
    ex = m.collate([m.preprocess(ctx.profile_data[0])], ctx.config.device)
    method = "script" if mspec.family == "torchvision" else "trace"
    return optimize_with_torchscript(model, example_input=ex, prefer=method)


def _graph_trt(model, mspec, ctx):
    from optimizations import build_trt_fp16
    return build_trt_fp16(model, min_block_size=ctx.config.min_block_size)


_GRAPH_BACKENDS = {
    "none":        _graph_noop,
    "compile":     _graph_compile,
    "cudagraphs":  _graph_cudagraphs,
    "torchscript": _graph_torchscript,
    "trt":         _graph_trt,
}


def _full(graph: str, fp16: bool = False):
    """Builder MODÈLE COMPLET : autocast FP16 optionnel, puis une optim de graphe.

    graph ∈ _GRAPH_BACKENDS. Tout est appliqué au modèle entier (NMS inclus) —
    certaines combinaisons sous-performent ou échouent (le NMS casse
    compile/cudagraphs/TRT). compile_fp16 = _full("compile", fp16=True) était le
    meilleur sur T4 (×2.35). dynamic=False fige les shapes (640).
    """
    backend = _GRAPH_BACKENDS[graph]
    def build(model, mspec, ctx):
        if fp16:
            from optimizations import to_fp16_autocast
            model = to_fp16_autocast(model)
        return backend(model, mspec, ctx)
    return build


def _zone_ctx(ctx, calib=None) -> dict:
    return {
        "compile_backend": ctx.config.compile_backend,
        "min_block_size":  ctx.config.min_block_size,
        "calib_loader":    calib,
    }


def _zone_builder(optimizer):
    """Crée un builder qui applique `optimizer` à la zone statique du modèle."""
    def build(model, mspec, ctx):
        return apply_zone_optimization(
            model, mspec.family, optimizer, _zone_ctx(ctx),
            device=ctx.config.device, size=ctx.config.size,
        )
    return build


def build_zone_trt_int8(model, mspec, ctx):
    from optimizations.tensorrt_int8 import build_calibration_loader
    calib = build_calibration_loader(
        ctx.profile_data, mspec.module.preprocess,
        n_images=ctx.config.int8_calib_images, batch_size=4,
    )
    return apply_zone_optimization(
        model, mspec.family, opt_trt_int8, _zone_ctx(ctx, calib),
        device=ctx.config.device, size=ctx.config.size,
    )


def build_mixed_trt_cudagraphs(model, mspec, ctx):
    # Plan défini dans zones.plan_mixed() — décision d'architecture.
    return apply_subzone_plan(
        model, mspec.family, plan_mixed()[mspec.family], _zone_ctx(ctx),
        device=ctx.config.device, size=ctx.config.size,
    )


DEFAULT_VARIANTS: List[VariantSpec] = [
    # ── Configs VALIDÉES (speedup réel sur T4) — priorité ─────────────────────
    VariantSpec("baseline",     _full("none"),               do_map=True,  with_modules=True,  profile=True, requires=None),
    VariantSpec("fp16",         _full("none", fp16=True),    do_map=True,  with_modules=True,  profile=True, requires="cuda"),
    VariantSpec("compile_fp16", _full("compile", fp16=True), do_map=False, with_modules=False, profile=True, requires="compile"),  # ×2.35 sur T4
    # ── TensorRT (exigence de l'encadrant) — backbone, MAP à batch=1 ──────────
    VariantSpec("zone_trt_fp16",   _zone_builder(opt_trt_fp16),        do_map=True, with_modules=False, profile=True, requires="trt", eval_batch_size=1),
    VariantSpec("zone_trt_folded", _zone_builder(opt_trt_fp16_folded), do_map=True, with_modules=False, profile=True, requires="trt", eval_batch_size=1),
    # ── Autres leviers (gain modeste sur T4, comparaison) ─────────────────────
    VariantSpec("zone_cudagraphs",  _zone_builder(opt_cudagraphs),  do_map=False, with_modules=False, profile=True, requires="cuda"),
    VariantSpec("zone_torchscript", _zone_builder(opt_torchscript), do_map=False, with_modules=False, profile=True, requires=None),
    VariantSpec("zone_compile",     _zone_builder(opt_compile),     do_map=False, with_modules=False, profile=True, requires="compile"),
    # ── Expérimental (« anales ») — à activer ensuite ─────────────────────────
    VariantSpec("mixed_trt_bb__cudagraphs_rest", build_mixed_trt_cudagraphs, do_map=False, with_modules=False, profile=True, requires="trt"),
    VariantSpec("zone_trt_int8", build_zone_trt_int8, do_map=True, with_modules=False, profile=True, requires="trt+int8", eval_batch_size=1),
]


# ══════════════════════════════════════════════════════════════════════════════
# Variantes MODÈLE COMPLET (sans zones) — toutes les optims, individuelles et
# combinées (autocast FP16). Réutilise la factory _full ci-dessus.
# Modèles compilés/tracés/TRT : shape figée à batch=1 → MAP à batch=1.
# ══════════════════════════════════════════════════════════════════════════════

FULL_VARIANTS: List[VariantSpec] = [
    VariantSpec("baseline",         _full("none"),                 do_map=True,  with_modules=True,  requires=None),
    VariantSpec("fp16",             _full("none", fp16=True),      do_map=True,  with_modules=True,  requires="cuda"),
    VariantSpec("torchscript",      _full("torchscript"),          do_map=True,  requires=None,      eval_batch_size=1),
    VariantSpec("compile",          _full("compile"),              do_map=False, requires="compile"),
    VariantSpec("cudagraphs",       _full("cudagraphs"),           do_map=False, requires="cuda"),
    VariantSpec("trt_fp16",         _full("trt"),                  do_map=True,  requires="trt",     eval_batch_size=1),
    VariantSpec("compile_fp16",     _full("compile", fp16=True),   do_map=True,  requires="compile", eval_batch_size=1),
    VariantSpec("cudagraphs_fp16",  _full("cudagraphs", fp16=True),do_map=True,  requires="cuda",    eval_batch_size=1),
    VariantSpec("torchscript_fp16", _full("torchscript", fp16=True),do_map=True, requires=None,      eval_batch_size=1),
]


# ══════════════════════════════════════════════════════════════════════════════
# Implémentations réelles (imports paresseux)
# ══════════════════════════════════════════════════════════════════════════════

def _real_benchmark_impl(model, data, preprocess, collate, n_warmup, n_measure, device, module_benchmark):
    from utils.benchmark import benchmark_model
    return benchmark_model(model, data, preprocess, collate,
                           n_warmup=n_warmup, n_measure=n_measure,
                           device=device, module_benchmark=module_benchmark)


def _real_map_impl(model, data, coco_gt, preprocess, collate, postprocess, device, batch_size=None):
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess, collate, postprocess,
                              device, batch_size=batch_size)


def _real_profile_impl(model, data, preprocess, collate, n_warmup, n_active, device):
    from profiler.pytorch_profiler import run_profile, profile_tables
    prof = run_profile(model, data, preprocess, collate,
                       n_warmup=n_warmup, n_active=n_active, device=device)
    return profile_tables(prof)


def _real_mb_factory():
    from utils.benchmark import ModuleBenchmark
    return ModuleBenchmark()


def _free_gpu():
    """Nettoyage AGRESSIF entre variantes — INDISPENSABLE.

    torch.compile / cudagraphs / TRT laissent un état GLOBAL dans le processus
    (caches inductor, pools mémoire cudagraph, contextes TRT). Sans reset, cet
    état pollue les variantes suivantes : le speedup se dégrade avec l'ordre
    d'exécution et les cudagraphs finissent par se corrompre. torch.compiler.reset
    appelle déjà torch._dynamo.reset en interne — un seul reset suffit.
    """
    import gc
    try:
        import torch
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
            torch.compiler.reset()
        else:                                       # vieux torch sans torch.compiler
            import torch._dynamo as _dynamo
            _dynamo.reset()
    except Exception:
        pass
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
    "AP", "AP50", "AP75", "n_modules", "profiled", "duration_s", "error",
]


class OptimizationRunner:
    def __init__(
        self,
        profile_data,
        eval_data,
        coco_gt,
        config: RunConfig,
        run_subdir: str,                       # ex. "results/optimization/<id>"
        benchmark_impl: Optional[Callable] = None,
        map_impl: Optional[Callable] = None,
        profile_impl: Optional[Callable] = None,
        module_benchmark_factory: Optional[Callable] = None,
    ):
        self.profile_data = profile_data
        self.eval_data    = eval_data
        self.coco_gt      = coco_gt
        self.config       = config

        self.benchmark_impl = benchmark_impl or _real_benchmark_impl
        self.map_impl       = map_impl or _real_map_impl
        self.profile_impl   = profile_impl or _real_profile_impl
        self.mb_factory     = module_benchmark_factory or _real_mb_factory

        # Tous les sous-dossiers sont créés sous le préfixe (Drive sur Colab).
        self.run_dir = ensure_dir(run_subdir)
        for sub in ("errors", "modules", "profiles", "bench", "eval"):
            ensure_dir(run_subdir, sub)
        self.run_subdir = run_subdir

        self.results: List[dict] = []
        self._baseline_ms: dict[str, float] = {}

        self.logger = self._setup_logger()
        self.logger.info(f"Préfixe : {project_prefix() or '(local)'}")
        self.logger.info(f"Run dir : {self.run_dir}")
        self.logger.info(f"Config  : {json.dumps(asdict(config))}")

    # ── Logging ────────────────────────────────────────────────────────────────

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"optrunner.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        fh = logging.FileHandler(self.run_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt); logger.addHandler(fh)
        sh = logging.StreamHandler(); sh.setFormatter(fmt); logger.addHandler(sh)
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
        self.results.append({f: kw.get(f, "") for f in _RESULT_FIELDS})
        self._save_csv("results.csv")

    def _save_csv(self, name):
        with open(self.run_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_RESULT_FIELDS)
            w.writeheader(); w.writerows(self.results)

    def _save_json(self, sub, model, variant, payload):
        path = self.run_dir / sub / f"{model}_{variant}.json"
        clean = {k: v for k, v in payload.items() if k != "modules"}
        path.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    def _save_df(self, sub, model, variant, df):
        if df is None or getattr(df, "empty", True):
            return 0
        df.to_csv(self.run_dir / sub / f"{model}_{variant}.csv", index=False)
        return len(df)

    def _save_error(self, model, variant, exc):
        (self.run_dir / "errors" / f"{model}_{variant}.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
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

            # 1. Benchmark
            mb = self.mb_factory() if spec.with_modules else None
            bench = self.benchmark_impl(
                optimized, self.profile_data,
                mspec.module.preprocess, mspec.module.collate,
                self.config.n_warmup, self.config.n_measure,
                self.config.device, mb,
            )
            mean_ms, fps = float(bench["mean_ms"]), float(bench["fps"])
            self._save_json("bench", mspec.name, spec.name, bench)
            n_mods = self._save_df("modules", mspec.name, spec.name, bench.get("modules"))

            if spec.name == "baseline":
                self._baseline_ms[mspec.name] = mean_ms
            base = self._baseline_ms.get(mspec.name)
            speedup = round(base / mean_ms, 3) if base else ""
            self.logger.info(f"   bench : {mean_ms:.2f} ms | {fps:.1f} FPS"
                             + (f" | x{speedup}" if speedup else ""))

            # 2. Éval MAP@640
            ap = ap50 = ap75 = ""
            if spec.do_map and mspec.has_map:
                self.logger.info(f"   MAP@640 ({len(self.eval_data)} images)...")
                # postprocess_map = postprocess d'éval à 640 si le modèle en définit un
                # (effdet : corrige le label +1 du postprocess de profiling). Sinon
                # postprocess standard (torchvision : déjà correct pour l'éval).
                post = getattr(mspec.module, "postprocess_map", mspec.module.postprocess)
                maps = self.map_impl(
                    optimized, self.eval_data, self.coco_gt,
                    mspec.module.preprocess, mspec.module.collate,
                    post, self.config.device, spec.eval_batch_size,
                )
                self._save_json("eval", mspec.name, spec.name, maps)
                ap, ap50, ap75 = (round(maps[k], 4) for k in ("AP", "AP50", "AP75"))
                self.logger.info(f"   AP={ap}  AP50={ap50}  AP75={ap75}")
            elif spec.do_map and not mspec.has_map:
                self.logger.info("   MAP ignoree (tete non entrainee)")

            # 3. Profiling (avant/après — kernels, mémoire, opérations)
            profiled = ""
            if spec.profile and self.config.do_profile:
                self.logger.info(f"   profiling ({self.config.n_profile} iters)...")
                df_prof = self.profile_impl(
                    optimized, self.profile_data,
                    mspec.module.preprocess, mspec.module.collate,
                    20, self.config.n_profile, self.config.device,
                )
                n_ops = self._save_df("profiles", mspec.name, spec.name, df_prof)
                profiled = f"{n_ops} ops"

            dur = round(time.time() - t0, 1)
            self.logger.info(f"   OK  ({dur}s)")
            self._record(model=mspec.name, variant=spec.name, status="OK",
                         mean_ms=round(mean_ms, 3), std_ms=round(float(bench["std_ms"]), 3),
                         fps=round(fps, 2), speedup=speedup,
                         AP=ap, AP50=ap50, AP75=ap75, n_modules=n_mods,
                         profiled=profiled, duration_s=dur)

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
        # Sauvegarde par-modèle (tout est déjà sous le préfixe Drive ; on fige une copie)
        self._save_csv("results_final.csv")
        self.logger.info(f"[SAVE]   {mspec.name} terminé — résultats figés dans {self.run_dir}")

    def run_all(self, mspecs: List[ModelSpec], variants: Optional[List[VariantSpec]] = None):
        for mspec in mspecs:
            self.run_model(mspec, variants)
        self._save_csv("results_final.csv")
        self.logger.info("TERMINÉ — %d lignes de résultats.", len(self.results))
        return self.results

    # ── Agrégation ───────────────────────────────────────────────────────────────

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.results, columns=_RESULT_FIELDS)

    def speedup_table(self):
        import pandas as pd
        df = self.to_dataframe()
        df_ok = df[df["status"] == "OK"].copy()
        rows = []
        for model in df_ok["model"].unique():
            sub = df_ok[df_ok["model"] == model]
            base = sub[sub["variant"] == "baseline"]
            if base.empty or base["mean_ms"].values[0] in ("", None):
                continue
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
