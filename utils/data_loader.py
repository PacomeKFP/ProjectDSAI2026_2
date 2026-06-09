"""
data_loader.py
──────────────
Chargement paresseux (lazy) des données COCO.

Principe :
  • Seule la liste des image_ids est conservée en mémoire au moment du sampling.
  • Les chemins, tailles originales et pixels ne sont résolus / lus
    qu'au moment où un élément est effectivement accédé.
  • LazySampleList se comporte comme une liste Python ordinaire (len, [], iter, slice)
    → aucun changement d'interface dans les modèles ou le notebook.

Empreinte mémoire après load_*_data :
  • Avant  : 2000 dicts pré-construits avec path + orig_size + éventuellement l'image
  • Après  : 2000 int  (≈ 16 Ko total)
"""
import random
from pathlib import Path

import cv2
from pycocotools.coco import COCO

_N    = 2000
_SEED = 42


# ── Sampling ────────────────────────────────────────────────────────────────────

def _sample(ann_file, n, seed):
    coco = COCO(ann_file)
    ids  = sorted(coco.getImgIds())          # tri déterministe
    random.seed(seed)
    return coco, random.sample(ids, min(n, len(ids)))


# ── Helper image ────────────────────────────────────────────────────────────────

def read_rgb(sample):
    """
    Retourne l'image sous forme de ndarray uint8 RGB.
    Lit depuis le disque via sample['path'] (chemin résolu à la demande).
    Accepte aussi sample['image'] si l'image est déjà chargée (rétro-compatibilité).
    """
    if "image" in sample:
        return sample["image"]
    img = cv2.imread(sample["path"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Conteneur lazy ──────────────────────────────────────────────────────────────

class LazySampleList:
    """
    Liste d'image_ids COCO. Un sample dict n'est construit que quand
    on accède à l'élément ([], iter, slice).

    Chaque sample dict contient :
      - 'image_id'   : int
      - 'path'       : str  (résolu à la volée depuis l'index COCO)
      - 'orig_size'  : (H, W)
    Les pixels ne sont jamais stockés ici.
    """

    def __init__(self, ids, coco, img_dir):
        self._ids     = ids
        self._coco    = coco
        self._img_dir = str(img_dir)

    # ── Interface list ──────────────────────────────────────────────────────────

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return (self._build(img_id) for img_id in self._ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._build(i) for i in self._ids[idx]]
        return self._build(self._ids[idx])

    # ── Construction d'un sample dict (lookup O(1) dans l'index COCO en RAM) ───

    def _build(self, img_id):
        info = self._coco.loadImgs(img_id)[0]
        return {
            "image_id":  img_id,
            "path":      str(Path(self._img_dir) / info["file_name"]),
            "orig_size": (info["height"], info["width"]),
        }

    def __repr__(self):
        return f"LazySampleList(n={len(self)}, img_dir={self._img_dir!r})"


# ── API publique ────────────────────────────────────────────────────────────────

def load_profiling_data(img_dir, ann_file, n=_N, seed=_SEED):
    """
    Retourne un LazySampleList d'n images (même seed → même sélection que load_eval_data).
    Seuls les image_ids sont en mémoire ; chemins et tailles sont résolus à la demande.
    """
    coco, ids = _sample(ann_file, n, seed)
    return LazySampleList(ids, coco, img_dir)


def load_eval_data(img_dir, ann_file, n=_N, seed=_SEED):
    """
    Retourne (LazySampleList, coco_gt).
    Même remarque : images non chargées, chemins non pré-calculés.
    coco_gt est l'objet COCO complet (annotations) pour COCOeval.
    """
    coco, ids = _sample(ann_file, n, seed)
    return LazySampleList(ids, coco, img_dir), coco
