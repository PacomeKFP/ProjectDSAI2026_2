"""utils/tqdm_compat.py — un seul point d'import pour tqdm.auto.

Dégrade en identité no-op si tqdm absent. Centralisé pour éviter la duplication
du fallback dans benchmark.py / map_eval.py / pytorch_profiler.py.
"""

try:
    from tqdm.auto import tqdm
except ImportError:                               # pragma: no cover
    def tqdm(it=None, **_):
        return it

__all__ = ["tqdm"]
