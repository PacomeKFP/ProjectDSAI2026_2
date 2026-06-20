import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from effdet import create_model
from utils.data_loader import read_rgb

_MODEL_NAME  = "tf_efficientdet_d6"
_NATIVE_SIZE = (1280, 1280)
_SCALE_PROF  = 640.0

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_model(device="cuda"):
    model = create_model(
        _MODEL_NAME, pretrained=True, num_classes=90,
        image_size=(640, 640), bench_task="predict",
    )
    return model.eval().to(device)


def preprocess(sample):
    img = read_rgb(sample)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.copy()).permute(2, 0, 1).unsqueeze(0)


def collate(inputs, device):
    return torch.cat(inputs, dim=0).to(device)


def postprocess(raw_item, orig_size):
    orig_h, orig_w = orig_size
    sx = orig_w / _SCALE_PROF
    sy = orig_h / _SCALE_PROF
    det   = raw_item[raw_item[:, 4] > 0]
    boxes = det[:, :4].clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return {"boxes": boxes, "labels": det[:, 5].long() + 1, "scores": det[:, 4]}


def postprocess_map(raw_item, orig_size):
    """Postprocess for MAP@640 EVAL -- without the "+1" on the label (which
    breaks COCO category matching), /640 scale, 0.05 score threshold."""
    orig_h, orig_w = orig_size
    sx = orig_w / _SCALE_PROF
    sy = orig_h / _SCALE_PROF
    det   = raw_item[raw_item[:, 4] > 0.05]
    boxes = det[:, :4].clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return {"boxes": boxes, "labels": det[:, 5].long(), "scores": det[:, 4]}


def run_inference(model, sample, device="cuda"):
    inp   = preprocess(sample)
    batch = collate([inp], device)
    with torch.no_grad():
        raw = model(batch)
    result = postprocess(raw[0], sample.get("orig_size", (640, 640)))
    result["image_id"] = sample["image_id"]
    return result


def load_model_eval(device="cuda"):
    model = create_model(_MODEL_NAME, pretrained=True, bench_task="predict")
    return model.eval().to(device)


def preprocess_eval(sample):
    img = Image.open(sample["path"]).convert("RGB")
    img = img.resize((_NATIVE_SIZE[1], _NATIVE_SIZE[0]), Image.BILINEAR)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    return t.unsqueeze(0)


def postprocess_eval(raw_item, orig_size):
    orig_h, orig_w = orig_size
    sx = orig_w / _NATIVE_SIZE[1]
    sy = orig_h / _NATIVE_SIZE[0]
    det   = raw_item[raw_item[:, 4] > 0.05]
    boxes = det[:, :4].clone()
    boxes[:, [0, 2]] *= sx
    boxes[:, [1, 3]] *= sy
    return {"boxes": boxes, "labels": det[:, 5].long(), "scores": det[:, 4]}


def evaluate_map(model, data, coco_gt, device="cuda"):
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess_eval, collate, postprocess_eval, device)
