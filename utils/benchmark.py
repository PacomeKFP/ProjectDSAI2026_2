"""
utils/benchmark.py
──────────────────
Outils de mesure de vitesse GPU pour les modèles de détection.

  benchmark_model(...)   — mesure le temps GPU end-to-end du forward
                           via CUDA Events (mean_ms, std_ms, fps)

  ModuleBenchmark        — mesure optionnelle par module feuille
                           (Conv2d, BatchNorm2d, ReLU, …) via hooks
                           forward_pre / forward, à passer en paramètre
                           de benchmark_model()

Ces outils mesurent la VITESSE — ils ne génèrent pas de trace ni de
décomposition des opérations ATen. Pour l'inspection et la visualisation,
voir profiler/pytorch_profiler.py (torch.profiler + export Chrome/Perfetto).

Pourquoi CUDA Events et non time.time() ?
  PyTorch dispatche les kernels de façon asynchrone : le CPU revient avant
  que le GPU ait terminé. time.time() mesurerait le dispatch CPU, pas le GPU.
  CUDA Events timestampent directement dans le stream GPU et donnent le temps
  d'exécution réel sur le silicium.

Caveat ModuleBenchmark : somme des modules ≠ temps total
  CUDA peut exécuter des kernels en parallèle (overlap entre BN et conv
  suivant, branches FPN, etc.). La somme donne le travail total théorique ;
  benchmark_model() donne le vrai temps mur GPU. L'écart mesure le
  parallélisme interne du modèle.
"""
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ── Utilitaire mémoire ────────────────────────────────────────────────────────

def estimate_batch_size(device="cuda", image_h=640, image_w=640,
                        safety=0.3, max_batch=200):
    bytes_per_img = image_h * image_w * 3 * 4 * 10
    if device != "cpu" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
    elif _HAS_PSUTIL:
        free_bytes = psutil.virtual_memory().available
    else:
        return 8
    return max(1, min(int(free_bytes * safety) // bytes_per_img, max_batch))


# ── ModuleBenchmark ───────────────────────────────────────────────────────────

class ModuleBenchmark:
    """
    Mesure le temps GPU de chaque module feuille d'un modèle via CUDA Events.

    Un module est une feuille s'il n'a aucun sous-module enfant
    (Conv2d, BatchNorm2d, ReLU, Linear, SiLU, MaxPool2d, …).

    Usage — passer une instance à benchmark_model() :

        mb = ModuleBenchmark()
        result = benchmark_model(model, data, preprocess, collate,
                                 module_benchmark=mb)
        result["modules"]   # DataFrame trié par mean_ms décroissant

    Usage standalone :

        mb = ModuleBenchmark()
        mb.attach(model)
        for s in data:
            model(preprocess(s))
            torch.cuda.synchronize()
            mb.collect()
        mb.detach()
        df = mb.summary()
    """

    def __init__(self):
        self._hooks   = []
        self._events  = {}                    # label → (start_event, end_event)
        self._records = defaultdict(list)     # label → [ms, ms, ...]

    def attach(self, model: nn.Module):
        """Attache des CUDA Events sur tous les modules feuilles du modèle."""
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue

            label = name if name else type(module).__name__
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            self._events[label] = (start, end)

            def _pre(lbl):
                def hook(mod, inp):
                    self._events[lbl][0].record()
                return hook

            def _post(lbl):
                def hook(mod, inp, out):
                    self._events[lbl][1].record()
                return hook

            self._hooks.append(module.register_forward_pre_hook(_pre(label)))
            self._hooks.append(module.register_forward_hook(_post(label)))

    def collect(self):
        """
        Lit le temps GPU de chaque module pour l'itération courante.
        Doit être appelé APRÈS torch.cuda.synchronize().
        """
        for label, (start, end) in self._events.items():
            try:
                t = start.elapsed_time(end)
                if t > 0:
                    self._records[label].append(t)
            except (RuntimeError, ValueError):
                pass

    def detach(self):
        """Supprime tous les hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> pd.DataFrame:
        """
        Retourne un DataFrame avec une ligne par module feuille :

          module         — chemin hiérarchique complet
          type           — classe PyTorch (Conv2d, BatchNorm2d, …)
          root_component — premier segment du chemin (backbone, fpn, head, …)
          mean_ms        — moyenne GPU sur toutes les itérations mesurées
          std_ms         — écart-type
          min_ms / max_ms
          pct_sum        — % du temps cumulé de tous les modules
          n_samples      — nombre d'itérations collectées

        Trié par mean_ms décroissant.
        """
        rows = []
        for label, times in self._records.items():
            if not times:
                continue
            t = np.array(times)
            rows.append({
                "module":         label,
                "type":           label.split(".")[-1] if "." in label else label,
                "root_component": label.split(".")[0]  if "." in label else label,
                "mean_ms":        float(t.mean()),
                "std_ms":         float(t.std()) if len(t) > 1 else 0.0,
                "min_ms":         float(t.min()),
                "max_ms":         float(t.max()),
                "n_samples":      len(t),
            })

        if not rows:
            return pd.DataFrame()

        df = (pd.DataFrame(rows)
                .sort_values("mean_ms", ascending=False)
                .reset_index(drop=True))
        total = df["mean_ms"].sum()
        df["pct_sum"] = (df["mean_ms"] / total * 100).round(2) if total > 0 else 0.0
        return df


# ── benchmark_model ───────────────────────────────────────────────────────────

def benchmark_model(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=50,
    n_measure=1000,
    device="cuda",
    module_benchmark=None,
):
    """
    Mesure le temps GPU end-to-end du forward (batch_size=1).

    Protocole :
      - n_warmup itérations non mesurées (chauffe GPU + caches)
      - n_measure itérations mesurées avec CUDA Events
      - H2D exclu : synchronize() avant starter.record()

    Parameters
    ----------
    module_benchmark : ModuleBenchmark | None
        Si fourni, mesure également le temps de chaque module feuille.
        Les hooks sont actifs dès le warmup (pour chauffer les caches)
        mais collect() n'est appelé que pendant la phase de mesure.

    Retourne
    --------
    dict :
      mean_ms, std_ms, min_ms, max_ms, fps
      + "modules" (DataFrame ModuleBenchmark.summary()) si module_benchmark fourni
    """
    if len(data) < n_warmup + n_measure:
        raise ValueError(
            f"Need {n_warmup + n_measure} samples, got {len(data)}."
        )

    model.eval()

    if module_benchmark is not None:
        module_benchmark.attach(model)

    with torch.no_grad():
        for s in data[:n_warmup]:
            inp = preprocess_fn(s)
            gpu = collate_fn([inp], device)
            model(gpu)
            del inp, gpu
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times   = []

    with torch.no_grad():
        for s in data[n_warmup : n_warmup + n_measure]:
            inp = preprocess_fn(s)
            gpu = collate_fn([inp], device)
            del inp
            torch.cuda.synchronize()
            starter.record()
            model(gpu)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
            if module_benchmark is not None:
                module_benchmark.collect()
            del gpu

    if module_benchmark is not None:
        module_benchmark.detach()

    gc.collect()
    torch.cuda.empty_cache()

    t = np.array(times)
    result = {
        "mean_ms": float(t.mean()),
        "std_ms":  float(t.std()),
        "min_ms":  float(t.min()),
        "max_ms":  float(t.max()),
        "fps":     float(1000.0 / t.mean()),
    }
    if module_benchmark is not None:
        result["modules"] = module_benchmark.summary()
    return result
