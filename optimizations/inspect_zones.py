"""
optimizations/inspect_zones.py
═══════════════════════════════
Inspection PRÉCISE de l'arbre des modules — pour découper les zones sans deviner.

Charge chaque modèle, imprime son arbre de modules (type + nombre de paramètres
+ % du total) jusqu'à une profondeur donnée, puis VÉRIFIE que get_subzone() pointe
exactement sur les bons sous-modules. Sauvegarde tout dans un fichier texte.

Usage :
    python -m optimizations.inspect_zones                 # tous, → docs/module_trees.txt
    python -m optimizations.inspect_zones --models retinanet_r50 efficientdet_d4
    python -m optimizations.inspect_zones --depth 4
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


_MODELS = {
    "retinanet_r50":   ("models.retinanet_r50",   "torchvision"),
    "fcos_r50":        ("models.fcos_r50",        "torchvision"),
    "efficientdet_d4": ("models.efficientdet_d4", "effdet"),
    "efficientdet_d5": ("models.efficientdet_d5", "effdet"),
    "efficientdet_d6": ("models.efficientdet_d6", "effdet"),
}


def _nparams(module) -> int:
    return sum(p.numel() for p in module.parameters())


def dump_tree(module, name, total, depth, max_depth, lines, prefix=""):
    """Imprime récursivement type + #params + %total de chaque sous-module."""
    n = _nparams(module)
    pct = (100.0 * n / total) if total else 0.0
    n_children = len(list(module.named_children()))
    leaf = " (feuille)" if n_children == 0 else ""
    lines.append(f"{prefix}{name}: {type(module).__name__}  "
                 f"[{n/1e6:.3f}M, {pct:4.1f}%]{leaf}")
    if depth < max_depth:
        for cname, child in module.named_children():
            dump_tree(child, cname, total, depth + 1, max_depth, lines, prefix + "   ")


def verify_subzones(model, family, lines):
    """Confirme que get_subzone pointe sur des modules réels et cohérents."""
    from optimizations.zones import SUBZONES, get_subzone, get_coarse_zones

    total = _nparams(model)
    lines.append("")
    lines.append(f"  VÉRIFICATION get_subzone (famille={family}) :")
    for sz in SUBZONES[family]:
        try:
            mod, setter = get_subzone(model, family, sz)
            n = _nparams(mod)
            lines.append(f"    sous-zone '{sz:10s}' -> {type(mod).__name__:24s} "
                         f"[{n/1e6:.3f}M, {100*n/total:4.1f}%]")
        except Exception as e:
            lines.append(f"    sous-zone '{sz:10s}' -> ERREUR : {e}")

    lines.append(f"  ZONES GROSSIÈRES (whole-zone) :")
    for cname, cmod, _ in get_coarse_zones(model, family, include_heads=True):
        n = _nparams(cmod)
        lines.append(f"    '{cname:10s}' -> {type(cmod).__name__:24s} "
                     f"[{n/1e6:.3f}M, {100*n/total:4.1f}%]")

    # Part statique optimisable vs reste (décodage/NMS = code du forward, pas un module)
    static = sum(_nparams(m) for _, m, _ in get_coarse_zones(model, family, include_heads=True))
    lines.append(f"  → Zone statique (params) : {static/1e6:.2f}M / {total/1e6:.2f}M "
                 f"({100*static/total:.1f}%)")
    lines.append(f"  → Décodage + NMS : pas un sous-module (logique du forward) → reste eager")


def inspect_one(key, depth, lines):
    mod_path, family = _MODELS[key]
    lines.append("=" * 78)
    lines.append(f"MODÈLE : {key}   (module={mod_path}, famille={family})")
    lines.append("=" * 78)
    m = importlib.import_module(mod_path).load_model("cpu")  # CPU suffit pour l'arbre
    m.eval()
    total = _nparams(m)
    lines.append(f"Total paramètres : {total/1e6:.2f}M  —  racine : {type(m).__name__}")
    lines.append("")
    dump_tree(m, "(racine)", total, 0, depth, lines)
    verify_subzones(m, family, lines)
    lines.append("")
    del m


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(_MODELS))
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--out", default="docs/module_trees.txt")
    args = ap.parse_args()

    lines = []
    for key in args.models:
        if key not in _MODELS:
            print(f"modèle inconnu : {key}"); continue
        try:
            inspect_one(key, args.depth, lines)
        except Exception as e:
            lines.append(f"[{key}] ÉCHEC du chargement : {type(e).__name__}: {e}\n")

    text = "\n".join(lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:                       # terminal cp1252 (Windows)
        print(text.encode("ascii", "replace").decode())
    print(f"\n-> Arbre sauvegarde dans {out}")


if __name__ == "__main__":
    main()
