import cv2
import numpy as np
import torch
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights
from utils.data_loader import read_rgb

# ── Profiling ──────────────────────────────────────────────────────────────────

_SCALE = 640.0


def load_model(device="cuda"):
    """640×640 — standardised speed benchmark."""
    model = fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.COCO_V1,
        min_size=640, max_size=640,
    )
    return model.eval().to(device)


def preprocess(sample):
    """Load image → resize to 640×640 → Tensor[3,H,W] float32 [0,1] CPU."""
    img = read_rgb(sample)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def collate(inputs, device):
    """List[Tensor[3,H,W]] CPU → List[Tensor[3,H,W]] on device."""
    return [t.to(device) for t in inputs]


def postprocess(raw_item, orig_size):
    """Boxes in 640-space → rescale to original image coordinates."""
    orig_h, orig_w = orig_size
    sx, sy = orig_w / _SCALE, orig_h / _SCALE
    boxes = raw_item["boxes"].clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return {"boxes": boxes, "labels": raw_item["labels"], "scores": raw_item["scores"]}


def run_inference(model, sample, device="cuda"):
    inp   = preprocess(sample)
    batch = collate([inp], device)
    with torch.no_grad():
        raw = model(batch)
    result = postprocess(raw[0], sample.get("orig_size", (640, 640)))
    result["image_id"] = sample["image_id"]
    return result


# ── COCO-standard MAP evaluation ───────────────────────────────────────────────

def load_model_eval(device="cuda"):
    """Natural resolution (800/1333), score_thresh=0.01 — COCO-standard evaluation."""
    model = fcos_resnet50_fpn(
        weights=FCOS_ResNet50_FPN_Weights.COCO_V1,
        score_thresh=0.01,
    )
    return model.eval().to(device)


def preprocess_eval(sample):
    """Load image at original resolution → Tensor[3,H,W] float32 [0,1] CPU.
    torchvision handles internal resizing."""
    img = read_rgb(sample)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def postprocess_eval(raw_item, orig_size):
    """Boxes already in input-image coordinates (torchvision rescales internally)."""
    return {
        "boxes":  raw_item["boxes"],
        "labels": raw_item["labels"],
        "scores": raw_item["scores"],
    }


def evaluate_map(model, data, coco_gt, device="cuda"):
    """COCO-standard MAP evaluation. Pass model from load_model_eval()."""
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess_eval, collate, postprocess_eval, device)
