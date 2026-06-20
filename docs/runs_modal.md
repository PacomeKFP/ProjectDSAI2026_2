# Modal runs -- history, results, limits

> Living document. Updated after every run and every fix.
> Reference for the report and for deciding the next actions.

---

## Variant glossary

### Full model (dynamic NMS included)
| Variant | Description |
|---|---|
| `baseline` | raw FP32, reference |
| `fp16` | autocast (Tensor Cores), zero compilation |
| `torchscript` | `jit.script` + `freeze` + `optimize_for_inference` |
| `compile` | `torch.compile` backend `inductor`, `dynamic=False` |
| `cudagraphs` | `torch.compile` backend `cudagraphs` (CUDA graph capture) |
| `compile_fp16` | autocast + inductor |
| `cudagraphs_fp16` | autocast + cudagraphs |
| `torchscript_fp16` | autocast + torchscript |
| `trt_fp16` | TensorRT FP16 via `torch_tensorrt`, full model |

### Zone-based (backbone+FPN+heads optimized, NMS stays eager)
| Variant | Description |
|---|---|
| `zone_torchscript` | TorchScript on the static zone |
| `zone_compile` | inductor on the static zone |
| `zone_cudagraphs` | CUDA graphs on the static zone |
| `zone_trt_fp16` | TensorRT FP16 on the backbone (clean TRT piece -- advisor's requirement) |
| `zone_trt_folded` | constant-fold (`jit.freeze`) **then** TRT -- for the BiFPN |
| `mixed_trt_bb__cudagraphs_rest` | TRT(backbone) + cudagraphs(FPN+heads) |

---

## Run history

### Run #1 -- 2026-06-15 21:00 (Colab T4)

**Context**: R50 on Colab T4, console output only (no Volume).

**Results** (R50, full-model only, 500 eval images):

| Variant | ms | FPS | speedup | MAP | Status |
|---|---|---|---|---|---|
| baseline | 58.55 | 17.1 | x1.00 | 0.401 | OK |
| fp16 | 62.65 | 16.0 | **x0.94** | 0.401 | OK (regresses) |
| torchscript | 50.94 | 19.6 | x1.15 | 0.401 | OK |
| compile | 61.02 | 16.4 | **x0.96** | -- | OK (regresses) |
| cudagraphs | 64.92 | 15.4 | **x0.90** | -- | OK (regresses) |
| **compile_fp16** | **32.47** | **30.8** | **x1.80** | 0.401 | **OK *** |
| cudagraphs_fp16 | 43.39 | 23.0 | x1.35 | 0.401 | OK |
| torchscript_fp16 | 44.75 | 22.3 | ~~x1.31~~ | 0.401 | **OK (FALSE)** |
| `trt_fp16` | -- | -- | -- | -- | **FAILED** |

### Run #2 -- 2026-06-15 23:42 (Modal A100 80 GB)

**Context**: first Modal test, A100 80 GB, NGC 24.10 image **with** `add_python="3.11"`.

**Results** (R50, 500 eval images, only 4 variants):

| Variant | ms | speedup | MAP | Status |
|---|---|---|---|---|
| baseline | 30.99 | x1.00 | 0.401 | OK |
| fp16 | 42.76 | **x0.725** | 0.401 | OK (regresses) |
| `trt_fp16` | -- | -- | -- | **SKIPPED** (TRT missing) |
| compile_fp16 | 19.85 | x1.56 | 0.401 | OK |

**Major anomaly**: `torch_tensorrt` and `tensorrt` were absent from the
container despite the NGC base. Cause: **`add_python="3.11"` reinstalls
Python on top of NGC** and breaks the preinstalled native packages.
-> **Fix applied in Run #3**.

### Run #3 -- upcoming (Modal T4 16 GB, 3 models x ~15 variants)

**Configuration**:
- Image: NGC 24.10 **without** `add_python` (native Python 3.10 preserved)
- GPU: T4 16 GB
- CPU/RAM: 8 vCPUs / 32 GB per container
- Volume `dsai2026` (1 TB included in the Starter plan)
- Shared caches: `TORCH_HOME=/data/cache/torch`, `HF_HOME=/data/cache/hf`
- Parameters: N_WARMUP=50, N_MEASURE=1000, N_PROFILE=150, N_PROFILE_DATA=2000, N_EVAL=2000
- 1 container per (model, variant), parallelism = 6
- 3 models x 12-15 variants = ~42 jobs

**Table to be filled in after the run**.

---

## Limits observed per approach

### baseline
- OK everywhere. Reference metric.

### fp16 (pure autocast)
- **Regresses** on A100 (x0.725) and on T4 (x0.94) **when the baseline is
  fast**: the FP32<->FP16 cast overhead exceeds the Tensor Cores gain.
- -> FP16 alone is never the winner; it must be **combined with a
  compilation** (compile, torchscript).

### torchscript (full)
- T4: x1.15 (modest gain -- Conv+BN fusion).
- With the FP16 wrapper, the trace fails (wrong input format) -> **must
  raise an exception**, not return the original model. **Fix applied.**

### compile (full)
- The dynamic NMS causes **recompile loops** on `decode_single`,
  `batched_nms`, `clip_boxes_to_image`.
- Hits `recompile_limit (8)` -> falls back to eager for the unseen shapes.
- T4: x0.96 (regresses). Only useful **combined with FP16**.

### cudagraphs (full)
- **`skipping cudagraphs due to cpu device`** on
  `anchor_generator.set_cell_anchors` and `_batched_nms_coordinate_trick`.
- **`CUDA Graph is empty`** -- cudagraphs captures nothing useful.
- T4: x0.90 (regresses). Unusable on the full model.

### compile_fp16 (full)
- **The full-model winner**: T4 x1.80, A100 x1.56.
- Also suffers NMS recompiles but the backbone gain dominates.
- To be highlighted in the report.

### trt_fp16 (full)
- **Systematic failure**: `out of bounds slice ... input dimensions = [0,1]`
  then `Error while setting the input shape`.
- Root cause: the NMS sometimes produces zero detections -> TRT tries to
  compile with input batch=0.
- TRT explicitly asks in the log: *"consider constant fold the model first"*
  and *"set upper bound on dynamic shapes"*.
- -> **Unusable on the full model.** The solution is `zone_trt_fp16`.

### zone_trt_fp16 -- the clean TRT path
- Optimizes **only** the backbone (fixed 640x640 shapes), leaves the NMS in
  eager.
- No dynamic shapes -> TRT compiles cleanly.
- **This is what we must present in the report to satisfy the advisor's
  requirement.**
- To be measured in Run #3.

### zone_trt_folded -- for the BiFPN (EfficientDet)
- `jit.freeze` propagates the constants of the frozen weights -> the
  **weighted fusion** of the BiFPN becomes a standard weighted addition ->
  TRT can fuse it.
- TRT explicitly asked for it (*"consider constant fold the model first"*).
- To be measured in Run #3.

### zone_cudagraphs
- x1.71 on R50 measured locally (RTX 5060) -> **removing the kernel launch
  overhead** is the lever on fast GPUs.
- To be confirmed on T4 (gain probably more modest because the T4 is
  slower, so the overhead weighs less in relative terms).

### mixed_trt_bb__cudagraphs_rest
- Hypothesis: TRT(backbone) + cudagraphs(FPN+heads) stacks the gains.
- To be measured on the 3 models. Risk: the transitions between optimized
  regions can be costly (buffer copies).

---

## Applied fixes

### Fix #1 -- `torchscript` fail-loud (committed)
Before: if `script` and `trace` failed, `optimize_with_torchscript` silently
returned the original model. The runner then measured the eager model and
reported a **fake speedup** (cf. Run #1, fake x1.31 for `torchscript_fp16`).
After: raises `RuntimeError`. The runner's try/except marks it `FAILED`
cleanly.

### Fix #2 -- Modal without `add_python` (committed)
Before: `add_python="3.11"` reinstalled Python on top of NGC, degrading the
native packages (torch_tensorrt, tensorrt). -> TRT unavailable.
After: we keep the native NGC Python. `effdet` and `timm` installed with
`--no-deps` to avoid touching the NGC `torch`. TRT preserved.

### Fix #3 -- shared caches on the Volume (committed)
Before: each container re-downloaded
`retinanet_resnet50_fpn_v2_coco-5905b1c5.pth` (146 MB).
After: `TORCH_HOME=/data/cache/torch` -> downloaded once, shared across all
Modal containers.

### Fix #4 -- granularity 1 job = 1 (model, variant) (committed)
Before: all variants of a model in a single container -> heartbeat timeouts
(blocking MAP eval), potential state pollution (resolved by reset but fragile).
After: one container per pair. More isolation, no heartbeat timeouts,
targeted re-run for free.

---

## TODO (later)

List of identified work items, to be coded once Run #3 is analyzed:

- [ ] **Bounding dynamic shapes for TRT**: `torch._dynamo.mark_dynamic` or
  `torch.export.Dim` to set an upper bound on the maximum number of boxes
  after NMS -> would enable full-model `trt_fp16` (low priority because
  `zone_trt_fp16` is the real way).
- [ ] **Constant folding via onnxsim** for path C2 from the design doc
  (alternative to `jit.freeze` for `zone_trt_folded`).
- [ ] **Possible rewrite of `FpnCombine.forward` (BiFPN)** to use coefficients
  pre-computed at inference (`relu(w)/sumw` -> constants in eval) -> makes the
  BiFPN TRT-friendly without depending on `freeze`.
- [ ] Profiler analysis: extract the most accelerable operations
  (specification point 2) from the CSVs in `/data/results/<run_id>/profiles/`.

---

## How to interpret a run

Fetch the results:
```bash
modal volume get dsai2026 results/<run_id> ./local_run/
```

Typical tree:
```
local_run/
  results.csv                    <- summary table (open first)
  bench/<model>_<variant>.json   <- raw speed metrics
  eval/<model>_<variant>.json    <- full COCO MAP/AR (per variant)
  modules/<model>_<variant>.csv  <- per-leaf-module timing (baseline/fp16)
  profiles/<model>_<variant>.csv <- operation table (kernels/memory)
  logs/<model>_<variant>.log     <- full container stdout/stderr
  errors/<model>_<variant>.txt   <- Python traceback if CONTAINER_FAILED
```

To understand why a variant failed:
1. Look at `results.csv` -> status
2. If `FAILED` or `CONTAINER_FAILED` -> open `errors/<model>_<variant>.txt`
   and `logs/<model>_<variant>.log`
3. Search the log for framework-specific warnings (TRT, dynamo, cudagraphs)
