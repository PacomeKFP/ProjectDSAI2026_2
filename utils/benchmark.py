"""
utils/benchmark.py
------------------
GPU speed measurement tools for detection models.

  benchmark_model(...)   -- measures end-to-end GPU time of the forward pass
                           via CUDA Events (mean_ms, std_ms, fps)

  ModuleBenchmark        -- optional per-leaf-module measurement
                           (Conv2d, BatchNorm2d, ReLU, ...) via forward_pre /
                           forward hooks, passed as a parameter to
                           benchmark_model()

These tools measure SPEED -- they do not generate a trace nor a breakdown of
ATen operations. For inspection and visualization, see
profiler/pytorch_profiler.py (torch.profiler + Chrome/Perfetto export).

Why CUDA Events and not time.time()?
  PyTorch dispatches kernels asynchronously: the CPU returns before the GPU
  has finished. time.time() would measure the CPU dispatch, not the GPU.
  CUDA Events timestamp directly inside the GPU stream and yield the real
  on-silicon execution time.

ModuleBenchmark caveat: sum of modules != total time
  CUDA can execute kernels in parallel (overlap between BN and the following
  conv, FPN branches, etc.). The sum gives the theoretical total work;
  benchmark_model() gives the actual GPU wall-clock time. The gap measures
  the model's internal parallelism.
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

from utils.tqdm_compat import tqdm   # progress bar (no-op if absent)


# -- Memory helper -------------------------------------------------------------

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


# -- ModuleBenchmark -----------------------------------------------------------

class ModuleBenchmark:
    """
    Measures the GPU time of every leaf module of a model via CUDA Events.

    A module is a leaf if it has no child sub-module
    (Conv2d, BatchNorm2d, ReLU, Linear, SiLU, MaxPool2d, ...).

    Usage -- pass an instance to benchmark_model():

        mb = ModuleBenchmark()
        result = benchmark_model(model, data, preprocess, collate,
                                 module_benchmark=mb)
        result["modules"]   # DataFrame sorted by mean_ms descending

    Standalone usage:

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
        self._events  = {}                    # label -> (start_event, end_event)
        self._records = defaultdict(list)     # label -> [ms, ms, ...]

    def attach(self, model: nn.Module):
        """Attach CUDA Events on every leaf module of the model."""
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
        Read the GPU time of each module for the current iteration.
        Must be called AFTER torch.cuda.synchronize().
        """
        for label, (start, end) in self._events.items():
            try:
                t = start.elapsed_time(end)
                if t > 0:
                    self._records[label].append(t)
            except (RuntimeError, ValueError):
                pass

    def detach(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per leaf module:

          module         -- full hierarchical path
          type           -- PyTorch class (Conv2d, BatchNorm2d, ...)
          root_component -- first path segment (backbone, fpn, head, ...)
          mean_ms        -- GPU mean across all measured iterations
          std_ms         -- standard deviation
          min_ms / max_ms
          pct_sum        -- % of the cumulative time of all modules
          n_samples      -- number of collected iterations

        Sorted by mean_ms descending.
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


# -- benchmark_model -----------------------------------------------------------

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
    Measure end-to-end GPU forward time (batch_size=1).

    Protocol:
      - n_warmup unmeasured iterations (warm up GPU + caches)
      - n_measure iterations measured with CUDA Events
      - H2D excluded: synchronize() before starter.record()

    Parameters
    ----------
    module_benchmark : ModuleBenchmark | None
        If provided, also measures the time of every leaf module. Hooks are
        active from the warmup (to warm caches up) but collect() is only
        called during the measurement phase.

    Returns
    -------
    dict :
      mean_ms, std_ms, min_ms, max_ms, fps
      + "modules" (DataFrame from ModuleBenchmark.summary()) if
        module_benchmark is provided
    """
    if len(data) < n_warmup + n_measure:
        raise ValueError(
            f"Need {n_warmup + n_measure} samples, got {len(data)}."
        )

    model.eval()

    if module_benchmark is not None:
        module_benchmark.attach(model)

    with torch.no_grad():
        for s in tqdm(data[:n_warmup], desc="  warmup", leave=False):
            inp = preprocess_fn(s)
            gpu = collate_fn([inp], device)
            model(gpu)
            del inp, gpu
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times   = []

    with torch.no_grad():
        for s in tqdm(data[n_warmup : n_warmup + n_measure], desc="  benchmark", leave=False):
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
