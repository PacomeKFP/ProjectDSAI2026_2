"""
optimizations/zones.py
=======================
ARCHITECTURE-AWARE optimization, by zone and SUB-ZONE.

Static / dynamic boundary of a detector:

  +----------- STATIC ZONE (optimizable, fixed 640 shapes) ------------+  +- DYNAMIC -+
  backbone + FPN/BiFPN + heads                                            decoding + NMS
        ^ TRT / compile / cudagraphs / TorchScript                          ^ eager (forced)

We REPLACE sub-modules with their optimized versions in-place. The full model
keeps its API; benchmark / MAP / profiling run on it.

Two granularities:
  * COARSE ZONE (whole-zone) -- to apply ONE single tool to the whole static
    part, minimizing boundaries:
        torchvision: model.backbone (body+fpn)  +  model.head
        effdet     : model.model (backbone+fpn+class_net+box_net, heads included)
  * SUB-ZONE (per-module)    -- for the MIXED variant: each sub-module receives
    the tool best suited to ITS architecture:
        torchvision: body | fpn | head
        effdet     : backbone | fpn | class_net | box_net

Backbone compatibility:
  * ResNet (R50, FCOS) -- DENSE convolutions -> FP16 Tensor Cores + TRT fusion
    are very effective.
  * EfficientNet (effdet) -- DEPTHWISE convolutions -> FP16 less useful
    (memory-bound); the BiFPN (weighted fusion) fuses poorly under TRT
    -> use cudagraphs, or constant-fold+TRT.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


# -- Sub-zone mapping (per-module, for the mixed variant) ----------------------

SUBZONES = {
    "torchvision": ["backbone", "fpn", "head"],
    "effdet":      ["backbone", "fpn", "class_net", "box_net"],
}


def plan_mixed():
    """Per-sub-zone plan for the mixed variant: one tool per module.

    Backbone -> TensorRT (known input [1,3,640,640], no capture needed).
    Everything else -> cudagraphs (no example required, removes the launch
    overhead of the many small FPN/head kernels).

    Lazily imported (opt_trt_fp16 pulls torch_tensorrt when the plan is built).
    Lives here, next to SUBZONES, because it is an architecture decision,
    not an orchestration one.
    """
    return {
        family: {
            zone: (opt_trt_fp16 if zone == "backbone" else opt_cudagraphs)
            for zone in SUBZONES[family]
        }
        for family in SUBZONES
    }


def get_subzone(model: nn.Module, family: str, name: str) -> Tuple[nn.Module, Callable]:
    """Return (submodule, setter) for a named sub-zone (fine granularity)."""
    if family == "torchvision":
        bb = model.backbone                       # BackboneWithFPN (body + fpn)
        if name == "backbone":
            return bb.body, lambda m: setattr(bb, "body", m)
        if name == "fpn":
            return bb.fpn, lambda m: setattr(bb, "fpn", m)
        if name == "head":
            return model.head, lambda m: setattr(model, "head", m)
    elif family == "effdet":
        inner = model.model                       # EfficientDet
        if name == "backbone":
            return inner.backbone, lambda m: setattr(inner, "backbone", m)
        if name == "fpn":
            return inner.fpn, lambda m: setattr(inner, "fpn", m)
        if name == "class_net":
            return inner.class_net, lambda m: setattr(inner, "class_net", m)
        if name == "box_net":
            return inner.box_net, lambda m: setattr(inner, "box_net", m)
    raise ValueError(f"unknown sub-zone: {family}/{name}")


# -- Coarse zones (whole-zone, minimal boundaries) -----------------------------

def get_coarse_zones(model: nn.Module, family: str,
                     include_heads: bool = True) -> List[Tuple[str, nn.Module, Callable]]:
    """
    Large contiguous static units (to apply ONE tool to the whole zone).
    Returns a list of (name, module, setter).
    """
    zones: List[Tuple[str, nn.Module, Callable]] = []
    if family == "torchvision":
        zones.append(("backbone", model.backbone, lambda m: setattr(model, "backbone", m)))
        if include_heads:
            zones.append(("head", model.head, lambda m: setattr(model, "head", m)))
    elif family == "effdet":
        # model.model already wraps backbone + fpn + heads -> a single unit.
        zones.append(("model", model.model, lambda m: setattr(model, "model", m)))
    else:
        raise ValueError(f"unknown family: {family!r}")
    return zones


def get_static_zone(model: nn.Module, family: str) -> Tuple[nn.Module, Callable]:
    """Main static zone (compat): backbone (tv) or model.model (effdet)."""
    name, mod, setter = get_coarse_zones(model, family, include_heads=False)[0]
    return mod, setter


def zone_example_input(device: str, size=(640, 640)) -> torch.Tensor:
    """Image input for the zone: tensor [1,3,H,W]."""
    h, w = size
    return torch.zeros(1, 3, h, w, device=device)


# -- Capturing intermediate inputs (for trace/TRT-folded) ----------------------

def _full_dummy(family: str, device: str, size=(640, 640)):
    """Input for the FULL model (format expected by the forward)."""
    h, w = size
    if family == "torchvision":
        return [torch.zeros(3, h, w, device=device)]      # List[Tensor]
    if family == "effdet":
        return torch.zeros(1, 3, h, w, device=device)      # Tensor[B,C,H,W]
    raise ValueError(family)


def _capture_inputs(model: nn.Module, family: str, device: str, size,
                    named_modules: List[Tuple[str, nn.Module]]) -> dict:
    """
    Run ONE dummy through the full model and capture, via forward_pre_hooks,
    each listed module's input argument tuple. Returns {name: (args...)}.
    """
    captured: dict = {}
    handles = []
    for name, mod in named_modules:
        def make(nm):
            def hook(m, inp):
                captured.setdefault(nm, inp)   # first call only
            return hook
        handles.append(mod.register_forward_pre_hook(make(name)))

    model.eval()
    try:
        with torch.no_grad():
            model(_full_dummy(family, device, size))
    finally:
        for h in handles:
            h.remove()
    return captured


def capture_subzone_inputs(model: nn.Module, family: str, device: str, size=(640, 640)) -> dict:
    """Real inputs for each fine-grained sub-zone (body/fpn/head...)."""
    named = [(n, get_subzone(model, family, n)[0]) for n in SUBZONES[family]]
    return _capture_inputs(model, family, device, size, named)


# -- Optimizers (module, ex, ctx) -> optimized module --------------------------
# ex : example input (captured tuple), only needed for trace/TRT-folded.
# ctx: {compile_backend, min_block_size, calib_loader}
#
# No opt_fp16: autocast must wrap the WHOLE forward (otherwise FP16 leaks into
# the FP32 head) -> FP16 is a full-model optimization (to_fp16_autocast).

def opt_torchscript(zone, ex, ctx):
    zone.eval()
    ts = torch.jit.trace(zone, ex, strict=False)
    try:
        ts = torch.jit.freeze(ts)
    except Exception:
        pass
    try:
        ts = torch.jit.optimize_for_inference(ts)
    except Exception:
        pass
    return ts


def opt_compile(zone, ex, ctx):
    return torch.compile(zone, backend=ctx.get("compile_backend", "inductor"), dynamic=False)


def opt_cudagraphs(zone, ex, ctx):
    return torch.compile(zone, backend="cudagraphs", dynamic=False)


def opt_trt_fp16(zone, ex, ctx):
    import torch_tensorrt  # noqa: F401
    return torch.compile(
        zone, backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.float16},
            "min_block_size": ctx.get("min_block_size", 5),
            "truncate_long_and_double": True,
        },
    )


def opt_trt_fp16_folded(zone, ex, ctx):
    """
    TensorRT FP16 AFTER constant-folding (path C1 from the design doc).

    torch.jit.freeze propagates constants: the frozen BiFPN weights turn
    relu(w)/(eps+sumrelu(w)) into a CONSTANT -> the weighted fusion reduces to a
    standard weighted addition that TRT knows how to fuse. This is what the
    "consider constant fold the model first" warnings were asking for.
    """
    import torch_tensorrt
    zone.eval()
    scripted = torch.jit.trace(zone, ex, strict=False)
    frozen   = torch.jit.freeze(scripted)                      # <- constant-fold
    inputs   = list(ex) if isinstance(ex, (tuple, list)) else [ex]
    return torch_tensorrt.compile(
        frozen, ir="torchscript", inputs=inputs,
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
    )


def opt_trt_int8(zone, ex, ctx):
    import torch_tensorrt
    calib = ctx.get("calib_loader")
    if calib is None:
        raise ValueError("opt_trt_int8 requires ctx['calib_loader']")
    from torch_tensorrt.ptq import DataLoaderCalibrator, CalibrationAlgo
    calibrator = DataLoaderCalibrator(
        calib, use_cache=False,
        algo_type=CalibrationAlgo.ENTROPY_CALIBRATION_2,
        device=torch.device("cuda:0"),
    )
    return torch.compile(
        zone, backend="torch_tensorrt",
        options={
            "enabled_precisions": {torch.int8},
            "calibrator": calibrator,
            "min_block_size": ctx.get("min_block_size", 5),
            "truncate_long_and_double": True,
        },
    )


# Optimizers that require an example input (trace/freeze).
_NEEDS_EXAMPLE = {"opt_torchscript", "opt_trt_fp16_folded"}


def _needs_example(optimizer: Callable) -> bool:
    return getattr(optimizer, "__name__", "") in _NEEDS_EXAMPLE


# -- Application: coarse zone (one tool over the whole static zone) ------------

def apply_zone_optimization(
    model: nn.Module,
    family: str,
    optimizer: Callable,
    ctx: dict,
    device: str = "cuda",
    size=(640, 640),
    include_heads: bool = False,
) -> nn.Module:
    """
    Apply `optimizer` to the coarse static zone. By default, a SINGLE
    contiguous region (backbone on torchvision; model.model on effdet, which
    already wraps the heads).

    [!] include_heads=False by default -- MEASURED finding: adding the head as a
    SEPARATE compiled region (torchvision) heavily degrades cudagraphs (x0.48
    vs x1.71 backbone-only), because FPN features must be copied between the
    two regions' static buffers on every iteration. "One big region + rest
    eager" beats "many small regions". The head is explored in the mixed
    variant (per-sub-zone) for comparison.

    Each unit is optimized in try/except (isolated eager fallback).
    """
    model.eval()
    zones = get_coarse_zones(model, family, include_heads)
    examples = {}
    if _needs_example(optimizer):
        examples = _capture_inputs(model, family, device, size,
                                   [(n, m) for n, m, _ in zones])

    for name, mod, setter in zones:
        mod.eval()
        try:
            ex = examples.get(name) if _needs_example(optimizer) else None
            setter(optimizer(mod, ex, ctx))
        except Exception as e:
            print(f"[zone] '{name}' not optimized ({type(e).__name__}: {str(e)[:80]}) -> eager")
    return model


# -- Application: per-sub-zone (mixed variant, one tool per module) ------------

def apply_subzone_plan(
    model: nn.Module,
    family: str,
    plan: dict,                # {subzone_name: optimizer or None}
    ctx: dict,
    device: str = "cuda",
    size=(640, 640),
) -> nn.Module:
    """
    Apply a DIFFERENT tool per sub-zone (None = leave it as eager).
    Only capture intermediate inputs if an optimizer needs them.
    Each sub-zone is optimized in try/except (isolated eager fallback).
    """
    model.eval()
    need_capture = any(opt is not None and _needs_example(opt) for opt in plan.values())
    examples = capture_subzone_inputs(model, family, device, size) if need_capture else {}

    for name, optimizer in plan.items():
        if optimizer is None:
            continue
        mod, setter = get_subzone(model, family, name)
        mod.eval()
        try:
            ex = examples.get(name) if _needs_example(optimizer) else None
            setter(optimizer(mod, ex, ctx))
        except Exception as e:
            print(f"[subzone] '{name}' not optimized ({type(e).__name__}: {str(e)[:80]}) -> eager")
    return model
