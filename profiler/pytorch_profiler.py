"""
profiler/pytorch_profiler.py
────────────────────────────
Profilage méso-scopique via torch.profiler.

Mécanique :
  • torch.profiler.schedule gère les phases warmup / active nativement :
      wait=0       : aucune itération sautée
      warmup=N_W   : GPU chauffe, trace collectée mais non commitée
      active=N_A   : trace commitée et exportée
      repeat=1     : un seul cycle
  • prof.step() avance la machine d'état interne à chaque itération.
  • on_trace_ready exporte automatiquement en fin de phase active.

Note sur les exports :
  tensorboard_trace_handler et export_chrome_trace appellent tous deux
  kineto_results.save() en interne → un seul export possible par run.
  On utilise tensorboard_trace_handler : le fichier .pt.trace.json produit
  est un Chrome JSON trace standard, lisible dans :
    - TensorBoard  (onglet PyTorch Profiler)
    - chrome://tracing
    - ui.perfetto.dev  (recommandé, plus rapide que Chrome)

Données collectées (maximum) :
  CPU activities       — appels Python, ATen, BLAS, cuDNN dispatch
  CUDA activities      — kernels GPU, copies mémoire, synchronisations
  record_shapes=True   — forme des tenseurs d'entrée par opération
  profile_memory=True  — allocations / désallocations / pic mémoire par op
  with_stack=True      — pile d'appels Python→C++ complète
  with_flops=True      — estimation FLOPs (conv2d, matmul, bmm)
  with_modules=True    — attribution au niveau module nn.Module (≥ PyTorch 1.12)

Convention de nommage des runs :
  <model_name>--<tagCamelCase>--<YYYYMMDD_HHMMSS>
  Ex : retinanet_r50--baseline--20250609_143022
       retinanet_r50--tensorRt--20250610_091500

Sorties :
  results/profiler/pytorch/<run_name>/
    tensorboard/        ← TensorBoard  +  Chrome / Perfetto (.pt.trace.json)
    summary.txt         ← tableau trié par cuda_time_total
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
    tensorboard_trace_handler,
)


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _to_camel_case(tag: str) -> str:
    """
    Convertit un tag texte libre en camelCase.
    Exemples :
      "baseline"    → "baseline"
      "base line"   → "baseLine"
      "tensor rt"   → "tensorRt"
      "my new tag"  → "myNewTag"
    """
    words = tag.strip().split()
    if not words:
        return "baseline"
    return words[0].lower() + "".join(w.capitalize() for w in words[1:])


def _run_name(model_name: str, tag: str) -> str:
    tag_cc = _to_camel_case(tag)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{model_name}—{tag_cc}—{ts}"


def _supports_with_modules() -> bool:
    try:
        major, minor = (int(x) for x in torch.__version__.split(".")[:2])
        return (major, minor) >= (1, 12)
    except Exception:
        return False


# ── Profiler principal ─────────────────────────────────────────────────────────

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
    Profile le forward pass avec torch.profiler (phases warmup/active natives).

    Parameters
    ----------
    model         : nn.Module en mode eval — issu de load_model()
    data          : LazySampleList — issu de load_profiling_data()
                    Doit contenir au moins n_warmup + n_active éléments.
    preprocess_fn : model.preprocess
    collate_fn    : model.collate
    n_warmup      : itérations de chauffe (trace non exportée)
    n_active      : itérations mesurées (trace exportée)
    output_dir    : répertoire racine des sorties
    model_name    : nom du modèle (préfixe du run)
    tag           : tag du run, converti en camelCase
                    Ex : "base line" → "baseLine"
    device        : 'cuda' ou 'cpu'

    Returns
    -------
    dict :
        run_name     — identifiant complet du run (str)
        tb_dir       — répertoire TensorBoard / traces (str)
        summary_path — chemin tableau texte principal (str)
        key_averages — EventList brut pour post-traitement
    """
    n_total = n_warmup + n_active
    if len(data) < n_total:
        raise ValueError(
            f"data contient {len(data)} samples, besoin de {n_total} "
            f"(n_warmup={n_warmup} + n_active={n_active})."
        )

    # ── Répertoires de sortie ──────────────────────────────────────────────────
    run  = _run_name(model_name, tag)
    out_dir = Path(output_dir) / run
    tb_dir  = out_dir / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    # ── Construction des kwargs profiler ──────────────────────────────────────
    profiler_kwargs = dict(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=n_warmup,
            active=n_active,
            repeat=1,
        ),
        on_trace_ready=tensorboard_trace_handler(str(tb_dir)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    )
    if _supports_with_modules():
        profiler_kwargs["with_modules"] = True

    # ── Boucle profilée ────────────────────────────────────────────────────────
    # Note sur le schedule PyTorch Profiler :
    #   on_trace_ready est déclenché au step() qui SUIT la fin de la phase active,
    #   pas pendant le dernier step actif. Avec schedule(wait=0, warmup=W, active=A),
    #   il faut W + A + 1 appels à prof.step() pour que le callback se déclenche.
    #   On ajoute donc un step "vide" (sans forward) à la fin — inoffensif car le
    #   schedule passe en état "closed" après repeat=1 et ignore tout step suivant.
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

            prof.step()   # avance wait → warmup → active

        prof.step()       # step +1 : déclenche on_trace_ready (transition active → closed)

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── Tableaux texte ─────────────────────────────────────────────────────────
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

    # ── Affichage résumé ───────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  PyTorch Profiler — {run}")
    print(f"{'='*62}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\n  Run         : {run}")
    print(f"  TensorBoard : tensorboard --logdir {out_dir}")
    print(f"  Perfetto    : ouvrir le .pt.trace.json de {tb_dir}")
    print(f"  Résumés     : {out_dir}/summary*.txt")

    return {
        "run_name":     run,
        "tb_dir":       str(tb_dir),
        "summary_path": str(summary_path),
        "key_averages": prof.key_averages(),
    }
