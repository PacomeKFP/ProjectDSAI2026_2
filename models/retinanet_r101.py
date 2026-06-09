import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from utils.data_loader import read_rgb

# Detectron2 is used only to load COCO-pretrained R101 weights.
# The returned object is a plain nn.Module; no Detectron2 inference pipeline is used.
# TODO: replace with a torchvision-based loader once R101 COCO weights are converted.

# D2 pred_classes are 0-79 contiguous. COCO category IDs are non-consecutive (1–90, 10 gaps).
# Simple +1 is wrong from index 11 onwards (e.g. contiguous 11 → cat_id 13, not 12).
_COCO_CAT_IDS = torch.tensor([
    1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
   21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
   43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
   63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86,
   87, 88, 89, 90,
], dtype=torch.long)

_SCALE = 640.0


def _build(cfg, device):
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
    cfg.MODEL.DEVICE = device
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    return model.eval()


# ── Profiling ──────────────────────────────────────────────────────────────────

def load_model(device="cuda"):
    """640×640 — standardised speed benchmark."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 640
    return _build(cfg, device)


def preprocess(sample):
    """Load + resize to 640×640 → D2 input dict with BGR float32 [0,255], height=640, width=640.
    D2 applies pixel_mean/std internally."""
    img = read_rgb(sample)
    img = cv2.resize(img, (640, 640))
    img_bgr = img[:, :, ::-1].astype(np.float32)
    tensor  = torch.from_numpy(img_bgr.copy()).permute(2, 0, 1)
    return {"image": tensor, "height": 640, "width": 640}


def collate(inputs, device):
    """List[dict_cpu] → List[dict_gpu]."""
    return [{**d, "image": d["image"].to(device)} for d in inputs]


def postprocess(raw_item, orig_size):
    """Boxes already in orig_size space (D2 rescales internally via height/width in input dict)."""
    inst = raw_item["instances"]
    return {
        "boxes":  inst.pred_boxes.tensor.cpu(),
        "labels": _COCO_CAT_IDS[inst.pred_classes.cpu()],
        "scores": inst.scores.cpu(),
    }


def run_inference(model, sample, device="cuda"):
    inp   = preprocess(sample)
    batch = collate([inp], device)
    with torch.no_grad():
        raw = model(batch)
    result = postprocess(raw[0], sample.get("orig_size", (640, 640)))
    result["image_id"] = sample["image_id"]
    return result


# ── COCO-standard MAP evaluation ───────────────────────────────────────────────
# D2 resizes internally (default 800/1333) and outputs boxes in the coordinate
# space of height/width passed in the input dict → no extra rescaling needed.

def load_model_eval(device="cuda"):
    """Natural resolution (D2 default 800/1333) — COCO-standard evaluation."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    # No MIN/MAX size override → D2 default (800 short side, 1333 long side)
    return _build(cfg, device)


def preprocess_eval(sample):
    """Load at original resolution → D2 input dict with orig height/width.
    D2 handles resize internally; boxes output in orig_size space."""
    orig_h, orig_w = sample.get("orig_size", (640, 640))
    img = read_rgb(sample)
    img_bgr = img[:, :, ::-1].astype(np.float32)
    tensor  = torch.from_numpy(img_bgr.copy()).permute(2, 0, 1)
    return {"image": tensor, "height": orig_h, "width": orig_w}


def postprocess_eval(raw_item, orig_size):
    """Identical to postprocess — boxes already in orig_size space."""
    return postprocess(raw_item, orig_size)


def evaluate_map(model, data, coco_gt, device="cuda"):
    """COCO-standard MAP evaluation. Pass model from load_model_eval()."""
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess_eval, collate, postprocess_eval, device)
