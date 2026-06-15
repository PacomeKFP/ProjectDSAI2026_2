"""
optimizations/zones.py
═══════════════════════
Optimisation CONSCIENTE DE L'ARCHITECTURE, par zone et SOUS-ZONE.

Frontière statique / dynamique d'un détecteur :

  ┌─────────── ZONE STATIQUE (optimisable, shapes fixes 640) ──────────┐  ┌─ DYNAMIQUE ─┐
  backbone + FPN/BiFPN + têtes                                            décodage + NMS
        ▲ TRT / compile / cudagraphs / TorchScript                          ▲ eager (obligé)

On REMPLACE des sous-modules par leur version optimisée, in-place. Le modèle
complet garde son API ; benchmark / MAP / profiling tournent dessus.

Deux granularités :
  • ZONE GROSSIÈRE (whole-zone)  — pour appliquer UN même outil à toute la partie
    statique, en minimisant les frontières :
        torchvision : model.backbone (body+fpn)  +  model.head
        effdet      : model.model (backbone+fpn+class_net+box_net, têtes incluses)
  • SOUS-ZONE (per-module)       — pour la variante MIXTE : chaque sous-module reçoit
    l'outil adapté à SON architecture :
        torchvision : body | fpn | head
        effdet      : backbone | fpn | class_net | box_net

Compatibilité par architecture de backbone :
  • ResNet (R50, FCOS) — convolutions DENSES → FP16 Tensor Cores + fusion TRT efficaces.
  • EfficientNet (effdet) — convolutions DEPTHWISE → FP16 peu utile (memory-bound) ;
    le BiFPN (fusion pondérée) fusionne mal sous TRT → cudagraphs, ou constant-fold+TRT.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn


# ── Cartographie des sous-zones (per-module, pour la variante mixte) ───────────

SUBZONES = {
    "torchvision": ["backbone", "fpn", "head"],
    "effdet":      ["backbone", "fpn", "class_net", "box_net"],
}


def plan_mixed():
    """Plan per-sous-zone de la variante mixte : un outil par module.

    Backbone → TensorRT (entrée connue [1,3,640,640], pas de capture nécessaire).
    Tout le reste → cudagraphs (pas d'exemple requis, supprime l'overhead de
    lancement des nombreux petits kernels FPN/têtes).

    Importé paresseusement (opt_trt_fp16 tire torch_tensorrt à la construction
    du plan). Vit ici, à côté de SUBZONES, parce que c'est une décision
    d'architecture, pas d'orchestration.
    """
    return {
        family: {
            zone: (opt_trt_fp16 if zone == "backbone" else opt_cudagraphs)
            for zone in SUBZONES[family]
        }
        for family in SUBZONES
    }


def get_subzone(model: nn.Module, family: str, name: str) -> Tuple[nn.Module, Callable]:
    """Retourne (sous_module, setter) pour une sous-zone nommée (granularité fine)."""
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
    raise ValueError(f"sous-zone inconnue : {family}/{name}")


# ── Zones grossières (whole-zone, frontières minimales) ────────────────────────

def get_coarse_zones(model: nn.Module, family: str,
                     include_heads: bool = True) -> List[Tuple[str, nn.Module, Callable]]:
    """
    Grandes unités statiques contiguës (pour appliquer UN outil à toute la zone).
    Retourne une liste de (nom, module, setter).
    """
    zones: List[Tuple[str, nn.Module, Callable]] = []
    if family == "torchvision":
        zones.append(("backbone", model.backbone, lambda m: setattr(model, "backbone", m)))
        if include_heads:
            zones.append(("head", model.head, lambda m: setattr(model, "head", m)))
    elif family == "effdet":
        # model.model englobe déjà backbone + fpn + têtes → une seule unité.
        zones.append(("model", model.model, lambda m: setattr(model, "model", m)))
    else:
        raise ValueError(f"famille inconnue : {family!r}")
    return zones


def get_static_zone(model: nn.Module, family: str) -> Tuple[nn.Module, Callable]:
    """Zone statique principale (compat) : backbone (tv) ou model.model (effdet)."""
    name, mod, setter = get_coarse_zones(model, family, include_heads=False)[0]
    return mod, setter


def zone_example_input(device: str, size=(640, 640)) -> torch.Tensor:
    """Entrée image de la zone : tenseur [1,3,H,W]."""
    h, w = size
    return torch.zeros(1, 3, h, w, device=device)


# ── Capture des entrées intermédiaires (pour trace/TRT-foldé) ──────────────────

def _full_dummy(family: str, device: str, size=(640, 640)):
    """Entrée du MODÈLE complet (format attendu par le forward)."""
    h, w = size
    if family == "torchvision":
        return [torch.zeros(3, h, w, device=device)]      # List[Tensor]
    if family == "effdet":
        return torch.zeros(1, 3, h, w, device=device)      # Tensor[B,C,H,W]
    raise ValueError(family)


def _capture_inputs(model: nn.Module, family: str, device: str, size,
                    named_modules: List[Tuple[str, nn.Module]]) -> dict:
    """
    Fait passer UN dummy dans le modèle complet et capture, via forward_pre_hooks,
    le tuple d'arguments d'entrée de chaque module listé. Retourne {nom: (args...)}.
    """
    captured: dict = {}
    handles = []
    for name, mod in named_modules:
        def make(nm):
            def hook(m, inp):
                captured.setdefault(nm, inp)   # 1er appel seulement
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
    """Entrées réelles de chaque sous-zone fine (body/fpn/head…)."""
    named = [(n, get_subzone(model, family, n)[0]) for n in SUBZONES[family]]
    return _capture_inputs(model, family, device, size, named)


# ── Optimiseurs (module, ex, ctx) -> module optimisé ───────────────────────────
# ex : exemple d'entrée (tuple capturé), nécessaire seulement pour trace/TRT-foldé.
# ctx : {compile_backend, min_block_size, calib_loader}
#
# Pas d'opt_fp16 : l'autocast doit envelopper TOUT le forward (sinon fuite FP16
# vers la tête FP32) → le FP16 est une optim du modèle complet (to_fp16_autocast).

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
    TensorRT FP16 APRÈS constant-folding (voie C1 du cahier).

    torch.jit.freeze propage les constantes : les poids gelés du BiFPN font que
    relu(w)/(ε+Σrelu(w)) devient une CONSTANTE → la fusion pondérée se réduit à
    une addition pondérée standard, que TRT sait fusionner. C'est ce que réclamaient
    les warnings « consider constant fold the model first ».
    """
    import torch_tensorrt
    zone.eval()
    scripted = torch.jit.trace(zone, ex, strict=False)
    frozen   = torch.jit.freeze(scripted)                      # ← constant-fold
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
        raise ValueError("opt_trt_int8 nécessite ctx['calib_loader']")
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


# Optimiseurs qui ont besoin d'un exemple d'entrée (trace/freeze).
_NEEDS_EXAMPLE = {"opt_torchscript", "opt_trt_fp16_folded"}


def _needs_example(optimizer: Callable) -> bool:
    return getattr(optimizer, "__name__", "") in _NEEDS_EXAMPLE


# ── Application : zone grossière (un même outil sur toute la zone statique) ─────

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
    Applique `optimizer` à la zone grossière statique. Par défaut, UNE seule
    région contiguë (backbone côté torchvision ; model.model côté effdet, qui
    englobe déjà les têtes).

    ⚠ include_heads=False par défaut — résultat MESURÉ : ajouter la tête comme
    région compilée SÉPARÉE (torchvision) dégrade fortement cudagraphs (×0.48 vs
    ×1.71 backbone seul), car les features FPN doivent être recopiées entre les
    buffers statiques des deux régions à chaque itération. « Une grande région +
    reste eager » bat « plusieurs petites régions ». La tête est explorée dans la
    variante mixte (per-sous-zone) à titre de comparaison.

    Chaque unité est optimisée en try/except (fallback eager isolé).
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
            print(f"[zone] '{name}' non optimise ({type(e).__name__}: {str(e)[:80]}) -> eager")
    return model


# ── Application : per-sous-zone (variante mixte, un outil par module) ───────────

def apply_subzone_plan(
    model: nn.Module,
    family: str,
    plan: dict,                # {nom_sous_zone: optimiseur ou None}
    ctx: dict,
    device: str = "cuda",
    size=(640, 640),
) -> nn.Module:
    """
    Applique un outil DIFFÉRENT par sous-zone (None = laisser en eager).
    Ne capture les entrées intermédiaires que si un optimiseur en a besoin.
    Chaque sous-zone est optimisée en try/except (fallback eager isolé).
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
            print(f"[subzone] '{name}' non optimise ({type(e).__name__}: {str(e)[:80]}) -> eager")
    return model
