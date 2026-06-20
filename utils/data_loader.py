"""
data_loader.py
--------------
Lazy loading of COCO data.

Principle:
  * Only the list of image_ids is kept in memory at sampling time.
  * Paths, original sizes and pixels are resolved/read only when an item is
    actually accessed.
  * LazySampleList behaves like an ordinary Python list (len, [], iter, slice)
    -> no interface change in the models or the notebook.

Memory footprint after load_*_data:
  * Before : 2000 pre-built dicts with path + orig_size + possibly the image
  * After  : 2000 ints (~ 16 KB total)
"""
import random
from pathlib import Path

import cv2
from pycocotools.coco import COCO

_N    = 2000
_SEED = 42


# -- Sampling --------------------------------------------------------------------

def _sample(ann_file, n, seed):
    coco = COCO(ann_file)
    ids  = sorted(coco.getImgIds())          # deterministic sort
    random.seed(seed)
    return coco, random.sample(ids, min(n, len(ids)))


# -- Image helper ----------------------------------------------------------------

def read_rgb(sample):
    """
    Return the image as an RGB uint8 ndarray.
    Reads from disk via sample['path'] (path resolved on demand).
    Also accepts sample['image'] if the image is already loaded (back-compat).
    """
    if "image" in sample:
        return sample["image"]
    img = cv2.imread(sample["path"])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -- Lazy container --------------------------------------------------------------

class LazySampleList:
    """
    List of COCO image_ids. A sample dict is only built when the corresponding
    element is accessed ([], iter, slice).

    Each sample dict contains:
      - 'image_id'   : int
      - 'path'       : str  (resolved on the fly from the COCO index)
      - 'orig_size'  : (H, W)
    Pixels are never stored here.
    """

    def __init__(self, ids, coco, img_dir):
        self._ids     = ids
        self._coco    = coco
        self._img_dir = str(img_dir)

    # -- List interface ----------------------------------------------------------

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return (self._build(img_id) for img_id in self._ids)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._build(i) for i in self._ids[idx]]
        return self._build(self._ids[idx])

    # -- Build a sample dict (O(1) lookup in the in-RAM COCO index) -------------

    def _build(self, img_id):
        info = self._coco.loadImgs(img_id)[0]
        return {
            "image_id":  img_id,
            "path":      str(Path(self._img_dir) / info["file_name"]),
            "orig_size": (info["height"], info["width"]),
        }

    def __repr__(self):
        return f"LazySampleList(n={len(self)}, img_dir={self._img_dir!r})"


# -- Public API ------------------------------------------------------------------

def load_profiling_data(img_dir, ann_file, n=_N, seed=_SEED):
    """
    Return a LazySampleList of n images (same seed -> same selection as load_eval_data).
    Only image_ids live in memory; paths and sizes are resolved on demand.
    """
    coco, ids = _sample(ann_file, n, seed)
    return LazySampleList(ids, coco, img_dir)


def load_eval_data(img_dir, ann_file, n=_N, seed=_SEED):
    """
    Return (LazySampleList, coco_gt).
    Same remark: images are not loaded, paths are not pre-computed.
    coco_gt is the full COCO object (annotations) for COCOeval.
    """
    coco, ids = _sample(ann_file, n, seed)
    return LazySampleList(ids, coco, img_dir), coco
