"""
profiler/pytorch_profiler.py
----------------------------
Meso-scopic profiling via torch.profiler.

Mechanics:
  * torch.profiler.schedule manages the warmup / active phases natively:
      wait=0       : no iterations skipped
      warmup=N_W   : GPU warms up, trace is collected but not committed
      active=N_A   : trace is committed and exported
      repeat=1     : a single cycle
  * prof.step() advances the internal state machine on every iteration.
  * on_trace_ready exports automatically at the end of the active phase.

Note on exports:
  tensorboard_trace_handler and export_chrome_trace both call
  kineto_results.save() internally -> only one export possible per run.
  We use tensorboard_trace_handler: the resulting .pt.trace.json file is a
  standard Chrome JSON trace, readable in:
    - TensorBoard  (PyTorch Profiler tab)
    - chrome://tracing
    - ui.perfetto.dev  (recommended, faster than Chrome)

Data collected (maximum):
  CPU activities       -- Python calls, ATen, BLAS, cuDNN dispatch
  CUDA activities      -- GPU kernels, memory copies, synchronizations
  record_shapes=True   -- input tensor shapes per operation
  profile_memory=True  -- allocations / deallocations / peak memory per op
  with_stack=True      -- full Python->C++ call stack
  with_flops=True      -- FLOPs estimation (conv2d, matmul, bmm)
  with_modules=True    -- attribution at the nn.Module level (>= PyTorch 1.12)

Run naming convention:
  <model_name>--<tagCamelCase>--<YYYYMMDD_HHMMSS>
  Ex: retinanet_r50--baseline--20250609_143022
      retinanet_r50--tensorRt--20250610_091500

Outputs:
  results/profiler/pytorch/<run_name>/
    tensorboard/        <- TensorBoard  +  Chrome / Perfetto (.pt.trace.json)
    summary.txt         <- table sorted by cuda_time_total
    summary_by_shape.txt
    summary_by_stack.txt
"""

import gc
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
)

from utils.tqdm_compat import tqdm


# -- Utilities ------------------------------------------------------------------

def _to_camel_case(tag: str) -> str:
    """
    Convert a free-form text tag to camelCase.
    Examples:
      "baseline"    -> "baseline"
      "base line"   -> "baseLine"
      "tensor rt"   -> "tensorRt"
      "my new tag"  -> "myNewTag"
    """
    words = tag.strip().split()
    if not words:
        return "baseline"
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _run_name(model_name: str, tag: str) -> str:
    tag_cc = _to_camel_case(tag)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}__{tag_cc}__{ts}"


def _supports_with_modules() -> bool:
    try:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        return (major, minor) >= (1, 12)
    except Exception:
        return False


# -- Main profiler --------------------------------------------------------------

def profile_with_pytorch(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=50,
    n_active=1000,
    output_dir="results/profiler/pytorch",
    model_name="model",
    tag="baseline",
    device="cuda",
):
    """
    Profile the forward pass with torch.profiler (native warmup/active phases).

    Parameters
    ----------
    model         : nn.Module in eval mode -- from load_model()
    data          : LazySampleList -- from load_profiling_data()
                    Must contain at least n_warmup + n_active items.
    preprocess_fn : model.preprocess
    collate_fn    : model.collate
    n_warmup      : warm-up iterations (trace not exported)
    n_active      : measured iterations (trace exported)
    output_dir    : root output directory
    model_name    : model name (run prefix)
    tag           : run tag, converted to camelCase
                    Ex: "base line" -> "baseLine"
    device        : 'cuda' or 'cpu'

    Returns
    -------
    dict :
        run_name     -- full run identifier (str)
        tb_dir       -- TensorBoard / traces directory (str)
        summary_path -- main text-table path (str)
        key_averages -- raw EventList for post-processing
    """
    n_total = n_warmup + n_active
    if len(data) < n_total:
        raise ValueError(
            f"data contains {len(data)} samples, need {n_total} "
            f"(n_warmup={n_warmup} + n_active={n_active})."
        )

    # -- Output directories -----------------------------------------------------
    run  = _run_name(model_name, tag)
    out_dir = Path(output_dir) / run
    tb_dir  = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # -- Build profiler kwargs -------------------------------------------------
    profiler_kwargs = dict(
        activities=[ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=n_warmup,
            active=n_active,
            repeat=1,
        ),
        # on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name=tb_dir, worker_name=model_name),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )
    if _supports_with_modules():
        profiler_kwargs["with_modules"] = True

    # -- Profiled loop ----------------------------------------------------------
    model.eval()
    with profile(**profiler_kwargs) as prof:
        for s in data[:n_total]:
            with torch.no_grad():
                inp = preprocess_fn(s)
                gpu = collate_fn([inp], device)
                del inp

                with record_function("model_forward"):
                    model(gpu)

                del gpu

            prof.step()


    # Explicit export of the Chrome JSON trace (.pt.trace.json)
    # Readable in: chrome://tracing, ui.perfetto.dev, TensorBoard (profiler plugin)
    trace_path = tb_dir / f"{model_name}.pt.trace.json"
    # prof.export_chrome_trace(f"{model_name}.pt.trace.json")

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- Text tables ------------------------------------------------------------
    def _write(path, content):
        Path(path).write_text(content, encoding="utf-8")

    summary_path = out_dir / "summary.txt"
    _write(summary_path,
           prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    _write(out_dir / "summary_by_shape.txt",
           prof.key_averages(group_by_input_shape=True)
               .table(sort_by="cuda_time_total", row_limit=40))

    _write(out_dir / "summary_by_stack.txt",
           prof.key_averages(group_by_stack_n=5)
               .table(sort_by="cuda_time_total", row_limit=40))

    # -- Summary display --------------------------------------------------------
    print(f"\n{'='*62}")
    print(f"  PyTorch Profiler -- {run}")
    print(f"{'='*62}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\n  Run         : {run}")
    # print(f"  Trace       : {trace_path}")
    print(f"  Perfetto    : drag the file onto ui.perfetto.dev")
    print(f"  Chrome      : open chrome://tracing -> Load -> select the file")
    print(f"  TensorBoard : tensorboard --logdir {tb_dir}")
    print(f"  Summaries   : {out_dir}/summary*.txt")

    return {
        "run_name":     run,
        "tb_dir":       str(tb_dir),
        "summary_path": str(summary_path),
        "key_averages": prof.key_averages(),
        'profiler': prof,
    }


# ==============================================================================
# Lightweight profiling for the orchestrator -- kernels / memory / operations
# ==============================================================================
# Designed based on user feedback:
#   * No trace export (.json): too large for 1000 images, and the internal
#     export was not reliably found on disk.
#   * The function RETURNS the prof object; saving (tables) is done by the
#     caller via profile_tables() -- exactly the requested pattern.
#   * Activates CPU + CUDA (the old code only enabled CPU -> no GPU kernels).

def run_profile(
    model,
    data,
    preprocess_fn,
    collate_fn,
    n_warmup=20,
    n_active=200,
    device="cuda",
):
    """
    Profile the forward (CPU+CUDA, memory, modules) over n_active iterations.
    Returns the `prof` object. The caller extracts/saves via profile_tables().

    We write NO file here and we do not export a trace: only aggregated
    statistics (key_averages) matter -> lightweight and searchable.
    """
    if len(data) < n_warmup + n_active:
        n_active = max(1, len(data) - n_warmup)

    activities = [ProfilerActivity.CPU]
    if device == "cuda" and torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    with torch.no_grad():
        for s in data[:n_warmup]:
            model(collate_fn([preprocess_fn(s)], device))
    if device == "cuda":
        torch.cuda.synchronize()

    kwargs = dict(activities=activities, record_shapes=False,
                  profile_memory=True, with_flops=True)
    if _supports_with_modules():
        kwargs["with_modules"] = True

    with profile(**kwargs) as prof:
        with torch.no_grad():
            for s in tqdm(data[n_warmup : n_warmup + n_active],
                          desc="  profile", leave=False):
                with record_function("forward"):
                    model(collate_fn([preprocess_fn(s)], device))
        if device == "cuda":
            torch.cuda.synchronize()

    return prof


def profile_tables(prof, top=None):
    """
    Convert prof.key_averages() to a searchable DataFrame, sorted by GPU time.

    Columns: op, count, CPU/CUDA time (total + self, us), CPU/CUDA memory
    (total + self, bytes), flops. Handles the torch 2.x rename
    (cuda_time_total -> device_time_total, etc.).
    """
    import pandas as pd

    def g(e, *names):
        for n in names:
            v = getattr(e, n, None)
            if v is not None:
                return v
        return 0

    rows = []
    for e in prof.key_averages():
        rows.append({
            "op":            e.key,
            "count":         e.count,
            "cpu_us_total":  g(e, "cpu_time_total"),
            "self_cpu_us":   g(e, "self_cpu_time_total"),
            "cuda_us_total": g(e, "device_time_total", "cuda_time_total"),
            "self_cuda_us":  g(e, "self_device_time_total", "self_cuda_time_total"),
            "cpu_mem":       g(e, "cpu_memory_usage"),
            "self_cpu_mem":  g(e, "self_cpu_memory_usage"),
            "cuda_mem":      g(e, "device_memory_usage", "cuda_memory_usage"),
            "self_cuda_mem": g(e, "self_device_memory_usage", "self_cuda_memory_usage"),
            "flops":         g(e, "flops"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("cuda_us_total", ascending=False).reset_index(drop=True)
    if top:
        df = df.head(top)
    return df
