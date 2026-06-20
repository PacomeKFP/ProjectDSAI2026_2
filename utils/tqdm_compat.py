"""utils/tqdm_compat.py -- single import point for tqdm.auto.

Degrades to an identity no-op if tqdm is missing. Centralized to avoid
duplicating the fallback in benchmark.py / map_eval.py / pytorch_profiler.py.
"""

try:
    from tqdm.auto import tqdm
except ImportError:                               # pragma: no cover
    def tqdm(it=None, **_):
        return it

__all__ = ["tqdm"]
