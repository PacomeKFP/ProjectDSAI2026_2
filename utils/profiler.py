"""
profiler.py
───────────
Deux niveaux de profilage GPU, même interface.

  profile_model(...)          — end-to-end (macro)
                                CUDA Events autour du forward complet.

  profile_model_detailed(...) — feuille par feuille (méso)
                                CUDA Events sur chaque module atomique.
                                Retourne les stats globales + un DataFrame
                                par feuille trié par temps GPU décroissant.

Principe des hooks feuilles
───────────────────────────
Un module est une « feuille » s'il n'a aucun sous-module enfant
(Conv2d, BatchNorm2d, ReLU, Linear, SiLU, MaxPool2d, LayerNorm…).

Pour chaque feuille on attache :
  forward_pre_hook  →  start.record()   (timestamp injecté dans le stream CUDA)
  forward_hook      →  end.record()     (timestamp injecté dans le stream CUDA)

Après chaque forward :
  synchronize()     →  le CPU attend que le GPU ait tout exécuté
  elapsed_time()    →  lit l'écart entre les deux timestamps GPU → ms réelles

Pourquoi pas time.time() ?
  PyTorch dispatche les kernels de façon asynchrone : le CPU revient avant
  que le GPU ait terminé. time.time() mesure le dispatch CPU, pas le GPU.
  CUDA Events timestampent directement dans le stream GPU.

Caveat : somme des feuilles ≠ temps total
  CUDA peut exécuter des kernels en parallèle (ex : branches FPN sur le même
  stream, ou overlap entre BN et le conv suivant). La somme donne le « travail
  total » théorique, le temps total mesuré par profile_model() donne le vrai
  mur d'horloge GPU. L'écart révèle le taux de parallélisme interne.
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


# ── Memory helper ──────────────────────────────────────────────────────────────

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


# ── LeafProfiler ───────────────────────────────────────────────────────────────

class LeafProfiler:
    """
    Attache des CUDA Events sur toutes les feuilles d'un modèle.

    Usage :
        lp = LeafProfiler()
        lp.attach(model)

        for s in data:
            model(preprocess(s))
            torch.cuda.synchronize()
            lp.collect()            # lit les temps de cette itération

        lp.detach()
        df = lp.summary()           # DataFrame trié par mean_ms décroissant
    """

    def __init__(self):
        self._hooks   = []
        self._events  = {}            # label → (start_event, end_event)
        self._records = defaultdict(list)   # label → [ms, ms, ...]

    # ── Attache ────────────────────────────────────────────────────────────────

    def attach(self, model: nn.Module):
        """
        Parcourt le graphe de modules et enregistre des hooks sur les feuilles.
        Un module est une feuille si `list(module.children())` est vide.
        """
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue                              # nœud intermédiaire, skip

            # Étiquette unique = chemin hiérarchique complet
            # Ex : "backbone.body.layer1.0.conv1"
            label = name if name else type(module).__name__

            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            self._events[label] = (start, end)

            # Closures — capturent label par valeur
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

    # ── Collecte (après chaque forward + synchronize) ──────────────────────────

    def collect(self):
        """
        Lit le temps GPU de chaque feuille pour l'itération qui vient de
        se terminer. Doit être appelé APRÈS torch.cuda.synchronize().
        """
        for label, (start, end) in self._events.items():
            try:
                t = start.elapsed_time(end)   # ms GPU entre les deux events
                if t > 0:
                    self._records[label].append(t)
            except RuntimeError:
                # elapsed_time lève RuntimeError si les events n'ont pas encore
                # été enregistrés (module non activé sur cette image, ex : dropout)
                pass

    # ── Détache ────────────────────────────────────────────────────────────────

    def detach(self):
        """Supprime tous les hooks — appeler impérativement en fin de profiling."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Résumé ─────────────────────────────────────────────────────────────────

    def summary(self) -> pd.DataFrame:
        """
        Retourne un DataFrame avec une ligne par feuille :

          module      — chemin hiérarchique complet
          type        — classe PyTorch  (Conv2d, BatchNorm2d, …)
          component   — premier segment du chemin  (backbone, fpn, head, …)
          mean_ms     — moyenne sur toutes les itérations mesurées
          std_ms      — écart-type
          min_ms / max_ms
          pct_sum     — % du temps total cumulé (sum of all leaves)
          n           — nombre d'itérations collectées

        Trié par mean_ms décroissant.
        """
        rows = []
        for label, times in self._records.items():
            if not times:
                continue
            t = np.array(times)
            rows.append({
                "module":    label,
                "type":      label.split(".")[-1] if "." in label else label,
                "component": label.split(".")[0]  if "." in label else label,
                "mean_ms":   float(t.mean()),
                "std_ms":    float(t.std()) if len(t) > 1 else 0.0,
                "min_ms":    float(t.min()),
                "max_ms":    float(t.max()),
                "n":         len(t),
            })

        if not rows:
            return pd.DataFrame()

        df = (pd.DataFrame(rows)
                .sort_values("mean_ms", ascending=False)
                .reset_index(drop=True))

        total = df["mean_ms"].sum()
        df["pct_sum"] = (df["mean_ms"] / total * 100).round(2) if total > 0 else 0.0
        return df


# ── profile_model — end-to-end (inchangé) ─────────────────────────────────────

def profile_model(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=50,
    n_measure=1000,
    device="cuda",
):
    """
    Mesure le temps GPU end-to-end du forward (batch_size=1).
    H2D exclu via synchronize() avant starter.record().
    Retourne dict : mean_ms, std_ms, min_ms, max_ms, fps.
    """
    if len(data) < n_warmup + n_measure:
        raise ValueError(
            f"Need {n_warmup + n_measure} samples, got {len(data)}."
        )

    model.eval()

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
            del gpu

    gc.collect()
    torch.cuda.empty_cache()
    t = np.array(times)
    return {
        "mean_ms": float(t.mean()),
        "std_ms":  float(t.std()),
        "min_ms":  float(t.min()),
        "max_ms":  float(t.max()),
        "fps":     float(1000.0 / t.mean()),
    }


# ── profile_model_detailed — feuille par feuille (méso) ───────────────────────

def profile_model_detailed(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=50,
    n_measure=1000,
    device="cuda",
):
    """
    Mesure le temps GPU de chaque feuille du modèle + le temps global.

    Même protocole que profile_model() :
      - n_warmup itérations non mesurées (GPU warmup, hooks actifs mais non collectés)
      - n_measure itérations mesurées

    Retourne
    --------
    dict :
      "global"  — même dict que profile_model() (mean_ms, std_ms, fps, …)
      "leaves"  — DataFrame LeafProfiler.summary() trié par mean_ms décroissant
    """
    if len(data) < n_warmup + n_measure:
        raise ValueError(
            f"Need {n_warmup + n_measure} samples, got {len(data)}."
        )

    model.eval()

    # Attache les hooks sur toutes les feuilles
    lp = LeafProfiler()
    lp.attach(model)

    # ── Warmup (hooks actifs, non collectés) ───────────────────────────────────
    with torch.no_grad():
        for s in data[:n_warmup]:
            inp = preprocess_fn(s)
            gpu = collate_fn([inp], device)
            model(gpu)
            del inp, gpu
    torch.cuda.synchronize()

    # ── Boucle mesurée ─────────────────────────────────────────────────────────
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times   = []

    with torch.no_grad():
        for s in data[n_warmup : n_warmup + n_measure]:
            inp = preprocess_fn(s)
            gpu = collate_fn([inp], device)
            del inp

            torch.cuda.synchronize()      # H2D terminé
            starter.record()
            model(gpu)                    # hooks enregistrent les events pendant ce call
            ender.record()
            torch.cuda.synchronize()      # tous les kernels GPU terminés

            times.append(starter.elapsed_time(ender))
            lp.collect()                  # lit elapsed_time de chaque feuille
            del gpu

    lp.detach()
    gc.collect()
    torch.cuda.empty_cache()

    t = np.array(times)
    global_stats = {
        "mean_ms": float(t.mean()),
        "std_ms":  float(t.std()),
        "min_ms":  float(t.min()),
        "max_ms":  float(t.max()),
        "fps":     float(1000.0 / t.mean()),
    }

    return {
        "global": global_stats,
        "leaves": lp.summary(),
    }
