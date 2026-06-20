import cv2
import numpy as np
import torch
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops
from utils.data_loader import read_rgb

# [!] NO COCO-pretrained RetinaNet-R101 exists in either torchvision or Detectron2.
#   (D2 only ships retinanet_R_50; torchvision only retinanet_resnet50_fpn[_v2].)
#
# Here we build a RetinaNet with an ImageNet-pretrained ResNet-101 backbone +
# FPN, mirroring exactly torchvision's construction of retinanet_resnet50_fpn,
# but with resnet101 instead of resnet50. The detection HEAD is initialized
# RANDOMLY.
#
# Consequences:
#   * SPEED BENCHMARK valid: the compute cost (backbone + FPN + conv heads)
#     does not depend on the weight values. Furthermore, torchvision
#     initializes the classification-head bias with prior_probability=0.01 ->
#     almost no detection passes the threshold -> light NMS -> the measured
#     forward correctly reflects the R101 architecture.
#   * MAP NOT MEANINGFUL: since the head is untrained, evaluate_map() will
#     return ~0. To be used for speed and optimizations only
#     (compile/FP16/TS/TRT).
#
# Interface identical to models/retinanet_r50.py (input List[Tensor[3,H,W]])
# -> drop-in for benchmark_model(), run_map_evaluation() and the
# optimizations/ pipeline.

_SCALE = 640.0


def _build_retinanet_r101(num_classes=91, min_size=640, max_size=640,
                          device="cuda", **kwargs):
    """Build RetinaNet + ResNet-101-FPN (ImageNet backbone, random head)."""
    backbone = resnet101(
        weights=ResNet101_Weights.IMAGENET1K_V2,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,   # detection convention (frozen BN)
    )
    # _resnet_fpn_extractor: same config as torchvision retinanet_resnet50_fpn
    #   returned_layers=[2,3,4] -> C3,C4,C5; extra_blocks=P6,P7 computed from P5
    backbone = _resnet_fpn_extractor(
        backbone, trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = RetinaNet(backbone, num_classes=num_classes,
                      min_size=min_size, max_size=max_size, **kwargs)
    return model.eval().to(device)


# -- Profiling ------------------------------------------------------------------

def load_model(device="cuda"):
    """640x640 -- standardized speed benchmark. ImageNet backbone, random head."""
    return _build_retinanet_r101(min_size=640, max_size=640, device=device)


def preprocess(sample):
    """Load image -> resize to 640x640 -> Tensor[3,H,W] float32 [0,1] CPU."""
    img = read_rgb(sample)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def collate(inputs, device):
    """List[Tensor[3,H,W]] CPU -> List[Tensor[3,H,W]] on device."""
    return [t.to(device) for t in inputs]


def postprocess(raw_item, orig_size):
    """Boxes in 640-space -> rescale to original image coordinates."""
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


# -- COCO-standard MAP evaluation -----------------------------------------------
# [!] Untrained head -> MAP ~0. Kept for interface consistency with r50/fcos.

def load_model_eval(device="cuda"):
    """Native resolution (800/1333), score_thresh=0.01.
    [!] Random head -> MAP not meaningful (speed only)."""
    print("[r101] [!] Detection head untrained -- MAP will be ~0 (speed only).")
    return _build_retinanet_r101(min_size=800, max_size=1333,
                                 score_thresh=0.01, device=device)


def preprocess_eval(sample):
    """Load at original resolution -> Tensor[3,H,W] float32 [0,1] CPU.
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
    """COCO-standard MAP evaluation. [!] Returns ~0 (untrained head)."""
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess_eval, collate, postprocess_eval, device)
