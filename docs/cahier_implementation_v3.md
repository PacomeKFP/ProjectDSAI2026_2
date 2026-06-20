# Implementation design doc -- Zone and sub-zone optimization (v3)

> Status: **to be approved**. Once approved, implement in the order of Sec.12.
> Guiding priority: **maximum gain for minimum effort**.

---

## 0. Purpose

Extend the current optimization pipeline (`optimizations/`) to:

1. **Also optimize the heads** (currently left eager on the torchvision side).
2. Introduce **sub-zone optimization**: apply to each sub-module the tool
   suited to *its* architecture (the troublesome module gets a specific
   treatment).
3. Add for the BiFPN the **two decided paths**:
   - **cudagraphs** (bypasses fusion -- already in place),
   - **constant-fold + TensorRT** (restores fusion -- TRT-recommended).

All of this without breaking the existing setup: the current variants remain
valid.

---

## 1. Established findings (do not redo)

| Established fact | Evidence |
|---|---|
| **Static-zone** optimization (backbone) unlocks the gain | cudagraphs R50 x1.71 vs x0.96 on the full model |
| The **NMS** must stay eager (dynamic shapes) | recompilations / `size mismatch` on the full model |
| **FP16** = **full-model** optimization (autocast), not zone | FP16 leak -> FP32 head -> dtype mismatch |
| **FP16 hurts effdet** (depthwise, no Tensor Cores) | x0.79 measured on D4 |
| **cudagraphs** is the right lever for effdet (small kernels) | x4.82 signal (noisy) on D4 |
| The **resample** hurts `inductor` (sympy), **not** TRT | sympy logs; TRT supports `Resize` |
| The **TRT brake on the BiFPN** = the **weighted fusion** (non-standard op) | `FpnCombine` + "consider constant fold" warnings |
| The profiler must **return** `prof`, the caller saves | trace not found otherwise; done in the runner |

**Model structure (verified):**

```
torchvision (R50, FCOS)          effdet (D4/D5/D6)
model.backbone : BackboneWithFPN  model.model : EfficientDet
  .body  (ResNet)                   .backbone (EfficientNet)
  .fpn   (FPN)                      .fpn      (BiFPN)
model.head                          .class_net
  .classification_head              .box_net
  .regression_head                model.anchors
model.transform, .anchor_generator
-> decoding + NMS = dynamic         -> decoding + NMS = dynamic
```

---

## 2. Design principles (the rules)

1. **Static/dynamic boundary**: everything before decoding/NMS is static at
   640x640 -> optimizable. Decoding + NMS **always** stays eager. The static
   zone = `backbone + FPN + heads` (heads included).
2. **Input frozen at 640x640**: every optimization assumes this fixed size.
   Outside that size -> breakage (hard for cudagraphs/TRT, recompilation for
   compile).
3. **Per sub-zone**: we can independently replace `backbone`, `fpn`, `head`.
   Each one gets the tool best suited to its architecture (cf. Sec.5.2).
4. **No nested `torch.compile`**: optimized sub-modules must be **disjoint**
   (never TRT *inside* a cudagraph). Eager glue between them.
5. **Robustness**: each optimization is attempted in try/except inside the
   runner. A sub-zone failure -> log + eager fallback for that sub-zone,
   everything else continues.
6. **Family symmetry**: after this work, torchvision and effdet both
   optimize `backbone + fpn + heads` (today only effdet includes the heads).

---

## 3. Work -- overview

| Task | Title | Gain | Effort | Depends on |
|---|---|---|---|---|
| **A** | Sub-zone access + zone broadened to the heads | medium-high | low | -- |
| **B** | Mixed variant `mixed_trt_bb__cudagraphs_rest` | TBD | medium | A |
| **C** | `zone_trt_folded` (constant-fold + TRT) | high (effdet) | medium | -- |
| **D** | `ort_full` (dynamic NMS in ONNX Runtime) | targeted test | medium | -- (optional) |

Chosen order: **A -> C -> B -> D** (A unlocks B; C is the independent "anales"
piece).

---

## 4. Task A -- Sub-zone access + zone broadened to the heads

### 4.1 Sub-zone mapping (`zones.py`)

```python
SUBZONES = {
    "torchvision": ["backbone", "fpn", "head"],
    "effdet":      ["backbone", "fpn", "class_net", "box_net"],
}
```

### 4.2 API to add in `zones.py`

```python
def get_subzone(model, family, name) -> tuple[nn.Module, Callable]:
    """Return (sub_module, setter) for a named sub-zone.
    torchvision: backbone->model.backbone.body, fpn->model.backbone.fpn, head->model.head
    effdet:      backbone->model.model.backbone, fpn->model.model.fpn,
                 class_net->model.model.class_net, box_net->model.model.box_net
    """

def capture_subzone_inputs(model, family, device, size=(640,640)) -> dict[str, tuple]:
    """Pass ONE 640x640 dummy through the full model and capture, via
    forward_pre_hooks, the real input of each sub-zone (for tracing/TRT
    which require an example). Return {name: (args...)}.
    - torchvision dummy: [torch.zeros(3,H,W)]   (List[Tensor])
    - effdet dummy:      torch.zeros(1,3,H,W)
    Hooks are removed at the end of the call.
    """

def apply_subzone_plan(model, family, plan: dict[str, Callable|None],
                       ctx: dict, device="cuda", size=(640,640)) -> nn.Module:
    """Apply a {sub_zone: optimizer or None} plan.
    None = leave the sub-zone in eager.
    - Capture intermediate inputs ONLY if at least one optimizer needs them
      (torchscript/TRT). cudagraphs/compile need no example.
    - Replace each sub-module in-place via its setter.
    Return the model (same API).
    """
```

### 4.3 Example need per optimizer

| Optimizer | Example required? | Example source |
|---|---|---|
| `opt_cudagraphs` | no (captured on the 1st call) | -- |
| `opt_compile` | no | -- |
| `opt_torchscript` (trace) | **yes** | `capture_subzone_inputs` |
| `opt_trt_fp16` | **yes** | `capture_subzone_inputs` |
| `opt_trt_int8` | yes + calib | same + calib loader |

-> `apply_subzone_plan` only calls `capture_subzone_inputs` if at least one
optimizer in the plan is in the "example required" list.

### 4.4 Heads-broadened zone -- integration

`apply_zone_optimization` (existing) becomes a special case of
`apply_subzone_plan`: **the same optimizer** is applied to every static
sub-zone, heads included.

```python
def apply_zone_optimization(model, family, optimizer, ctx, device, size,
                            include_heads=True):
    """Apply `optimizer` to backbone + fpn (+ heads if include_heads).
    Implemented via apply_subzone_plan with the same optimizer everywhere.
    include_heads=True by default -> torchvision/effdet symmetry."""
    names = list(SUBZONES[family])
    if not include_heads:
        names = [n for n in names if n not in ("head","class_net","box_net")]
    plan = {n: optimizer for n in names}
    return apply_subzone_plan(model, family, plan, ctx, device, size)
```

> Consequence: the existing `zone_*` variants now also optimize **the
> heads** on the torchvision side. That's the "broadened zone" improvement.

### 4.5 Acceptance criteria (task A)

- [ ] `get_subzone` returns the right module for the 6 sub-zones (3 tv + 4 effdet).
- [ ] `capture_subzone_inputs` returns a non-empty input for every sub-zone.
- [ ] `apply_zone_optimization(..., include_heads=True)` on R50 produces a
      correct `List[Dict]` output (test: forward on 1 dummy image).
- [ ] `zone_cudagraphs` on R50 with heads included: no crash, speedup >=
      backbone-only version.
- [ ] Local tests: cudagraphs + torchscript; **known risk**: tracing a head
      with `dict` input (RetinaNetHead). On failure -> eager head fallback
      (runner try/except).

---

## 5. Task B -- Mixed variant `mixed_trt_bb__cudagraphs_rest`

### 5.1 Design

```
input 640 -> [TRT FP16] backbone -> [cudagraphs] fpn + heads -> [eager] decoding+NMS
```

**Key trick**: TRT is applied only to the **backbone** (known input =
`[1,3,640,640]`, no capture needed); the rest (fpn + heads) goes to
**cudagraphs** (no example required). -> the mixed variant **does not need**
`capture_subzone_inputs`. Direct implementation.

### 5.2 Per-family optimization plan (the "right tool per module")

```python
PLAN_MIXED = {
  "torchvision": {"backbone": opt_trt_fp16, "fpn": opt_cudagraphs, "head": opt_cudagraphs},
  "effdet":      {"backbone": opt_trt_fp16, "fpn": opt_cudagraphs,
                  "class_net": opt_cudagraphs, "box_net": opt_cudagraphs},
}
```

> For torchvision, `backbone` = `model.backbone.body` (the ResNet alone);
> the FPN goes to cudagraphs together with the heads.

### 5.3 Runner builder

```python
def build_mixed_trt_cudagraphs(model, mspec, ctx):
    from optimizations.zones import apply_subzone_plan, opt_trt_fp16, opt_cudagraphs
    plan = PLAN_MIXED[mspec.family]
    return apply_subzone_plan(model, mspec.family, plan, _zone_ctx(ctx),
                              ctx.config.device, ctx.config.size)
```

Variant: `VariantSpec("mixed_trt_bb__cudagraphs_rest", build_mixed_trt_cudagraphs,
do_map=True, with_modules=False, profile=True, requires="trt")`.

### 5.4 Acceptance criteria (task B)

- [ ] The variant runs on Colab (TRT) without crashing, correct output.
- [ ] We compare in the numbers: `zone_cudagraphs` (all cudagraphs) vs
      `zone_trt_fp16` (all TRT) vs `mixed` -> per-family decision table.
- [ ] Locally (without TRT): the variant is cleanly **SKIPPED**
      (`requires="trt"`).

---

## 6. Task C -- `zone_trt_folded` (constant-fold + TensorRT)

### 6.1 The mechanism

At inference, the BiFPN weights are frozen -> `relu(w_i)/(eps+sumrelu(w_j))` are
**constants**. Folding them turns the weighted fusion into a **weighted
addition by constants**, which TRT can fuse. This is what the TRT warnings
were asking for ("consider constant fold the model first").

### 6.2 Path C1 -- `freeze` + TRT TorchScript frontend (primary, low effort)

`torch.jit.freeze` performs **constant propagation**: it treats the frozen
parameters as constants and folds `relu(param)`, the sum, and the division.

```python
def opt_trt_fp16_folded(zone, ex, ctx):
    import torch_tensorrt
    zone.eval()
    scripted = torch.jit.trace(zone, ex, strict=False)
    frozen   = torch.jit.freeze(scripted)         # <- constant-fold
    return torch_tensorrt.compile(
        frozen, ir="torchscript", inputs=[ex],
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
    )
```

- **Example required** -> via `capture_subzone_inputs` (effdet zone =
  `model.model`, input = `[1,3,640,640]`, so known; no capture needed if
  we apply it to the whole `model.model` zone).
- Drop-in: stays in the torch world.

### 6.3 Path C2 -- ONNX + `onnxsim` + TRT (production, optional)

```
export ONNX (model.model, fixed 640) -> onnxsim (simplify + constant fold)
  -> TensorRT engine (via torch_tensorrt ONNX frontend OR ORT-TRT EP)
```

- `onnxsim` does a more aggressive constant-fold than `freeze`.
- The drop-in integration is heavier (ORT/engine wrapper). **Only to do if
  C1 is not enough** (measure fusion via the profiler before/after).
- Reference: NVIDIA's official EfficientDet sample (graph-surgeon +
  EfficientNMS).

### 6.4 Acceptance criteria (task C)

- [ ] `zone_trt_folded` (C1) runs on Colab, correct output, MAP@640 ~=
      `zone_trt_fp16`.
- [ ] **Key measurement**: via the profiler before/after, the **number of
      distinct operations decreases** and the GPU time of the BiFPN drops
      vs non-folded `zone_trt_fp16`. -> this proves that constant-fold
      restored fusion.
- [ ] effdet comparison: `zone_cudagraphs` vs `zone_trt_fp16` vs
      `zone_trt_folded`.

---

## 7. Task D -- `ort_full` (optional)

Export the **full model** (NMS included) to ONNX -> ONNX Runtime CUDA EP.
Tests whether a **dynamic-shape** runtime breaks the NMS wall where TRT
stumbles.

```python
def build_ort_full(model, mspec, ctx):
    # 1. export_full_detection(model, onnx_path, image_size=(640,640))
    # 2. ORTModelFull(onnx_path): wrapper that exposes the same API (List[Dict] / Tensor)
    ...
```

- **Risk**: the ONNX export of a full torchvision detector (with NMS) is
  tricky; same for effdet via `DetBenchPredict`. -> variant marked
  *experimental*, `requires="ort"`, clean fallback if the export fails.
- Bench only (MAP comparison would need a dedicated postprocess) ->
  `do_map=False`.

---

## 8. Final variant set

| Variant | backbone | fpn | heads | NMS | MAP | Profile | requires | Families |
|---|---|---|---|---|:---:|:---:|---|---|
| `baseline` | -- | -- | -- | eager | [OK] | [OK] | -- | all |
| `fp16` | full-model autocast | [OK] | [OK] | cuda | all |
| `zone_torchscript` | TS | TS | TS | eager | [X] | [OK] | -- | all |
| `zone_compile` | compile | compile | compile | eager | [X] | [OK] | compile | tv (effdet opt-in) |
| `zone_cudagraphs` | cg | cg | cg | eager | [X] | [OK] | cuda | all * |
| `zone_trt_fp16` | TRT | TRT | TRT | eager | [OK] | [OK] | trt | all |
| `zone_trt_int8` | TRT-int8 | ... | ... | eager | [OK] | [OK] | trt+int8 | all (opt) |
| **`mixed_trt_bb__cudagraphs_rest`** | TRT | cg | cg | eager | [OK] | [OK] | trt | all * |
| **`zone_trt_folded`** | TRT(folded) | TRT(folded) | TRT(folded) | eager | [OK] | [OK] | trt | effdet * |
| `ort_full` *(opt)* | ONNX Runtime full model | -- | [X] | [OK] | ort | all |

(cg = cudagraphs; * = pieces of high interest for the report)

**Assignment per family (notebook):**

```python
VARIANTS_TV     = [baseline, fp16, zone_torchscript, zone_compile, zone_cudagraphs,
                   zone_trt_fp16, mixed_trt_bb__cudagraphs_rest, zone_trt_int8(opt)]
VARIANTS_EFFDET = [baseline, fp16, zone_torchscript, zone_cudagraphs,
                   zone_trt_fp16, zone_trt_folded, mixed_trt_bb__cudagraphs_rest,
                   zone_trt_int8(opt)]   # zone_compile opt-in
```

---

## 9. Per-file changes

| File | Change |
|---|---|
| `optimizations/zones.py` | + `SUBZONES`, `get_subzone`, `capture_subzone_inputs`, `apply_subzone_plan`, `opt_trt_fp16_folded`; `apply_zone_optimization` rewritten via plan + `include_heads` |
| `optimizations/runner.py` | + builders `build_mixed_trt_cudagraphs`, `build_zone_trt_folded`, (opt) `build_ort_full`; + variants in `DEFAULT_VARIANTS`; + `PLAN_MIXED` |
| `optimizations/ort_inference.py` | (task D) + `export_full_detection` usage + `ORTModelFull` wrapper |
| `optimizations/__init__.py` | export the new functions |
| `optimization_full.ipynb` | updated `VARIANTS_TV`/`VARIANTS_EFFDET` lists; comparative analysis cell `cudagraphs vs trt vs folded vs mixed` |
| `docs/cahier_implementation_v3.md` | this document |

---

## 10. Test plan

**Local (Windows, `base` conda, CUDA, NO TRT):**
- [ ] `get_subzone`: 6 sub-zones resolved correctly.
- [ ] `capture_subzone_inputs`: inputs captured (R50 + D4).
- [ ] `apply_zone_optimization(include_heads=True)` cudagraphs on R50/FCOS/D4:
      correct forward + measured speedup.
- [ ] tracing the head (`opt_torchscript` on `head`): OK or clean fallback.
- [ ] `mixed_*` cudagraphs part (backbone left eager locally for lack of
      TRT): forward OK.

**Colab (Linux, TRT):**
- [ ] `mixed_trt_bb__cudagraphs_rest`: forward + MAP@640 + profile.
- [ ] `zone_trt_folded` (C1): forward + MAP@640 + **before/after profile**
      (proof of fusion).
- [ ] (opt) `ort_full`.

For each test: a small `python -u` script via `conda` base, reduced n_iter,
filtered output.

---

## 11. Risks & open decisions

| Risk | Mitigation |
|---|---|
| Tracing/TRT of a head with **dict** input (RetinaNetHead) fails | eager head fallback (runner try/except); cudagraphs/compile on the head do not have this issue |
| `freeze` does not fold **all** of the BiFPN | measure via profiler; if insufficient -> path C2 (onnxsim) |
| **Boundary cost** in `mixed` cancels the gain | measure; if negative -> conclude "cudagraphs-everywhere wins" (valid result) |
| Accidental `torch.compile` nesting | ensure **disjoint** sub-modules in the plan |
| Wrong captured input if the dummy does not follow the right path | use the real pipeline input (model's collate) for the dummy |

**Open decisions to discuss with you:**
1. `ort_full` (task D): include it now or keep it for later?
2. For `zone_trt_folded`: limit ourselves to C1 (freeze) first, C2 (onnxsim)
   only if C1 is insufficient -- OK?
3. Heads-broadened zone: do we enable it **by default** on every `zone_*`
   variant (symmetry), or do we keep a `zone_*_no_head` variant to compare
   the heads' contribution? (I lean toward: heads included by default + 1
   one-off backbone-only measurement on R50 to quantify the heads'
   contribution.)

---

## 12. Implementation order & milestones

1. **Milestone 1 -- Task A**: sub-zones + heads-broadened zone.
   Local cudagraphs/TS test. -> freezes the sub-zone API.
2. **Milestone 2 -- Task C (C1)**: `zone_trt_folded` (freeze + TRT TS).
   Colab test + before/after profile (proof of fusion).
3. **Milestone 3 -- Task B**: mixed variant. Colab test + comparison table.
4. **Milestone 4 -- Task D (if kept)**: `ort_full`.
5. **Milestone 5**: notebook update + final comparative analysis cell.

After each milestone: isolated test (local or Colab depending on
availability), then runner integration.
