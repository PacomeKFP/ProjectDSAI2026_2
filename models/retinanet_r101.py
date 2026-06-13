import cv2
import numpy as np
import torch
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.detection import RetinaNet
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops
from utils.data_loader import read_rgb

# ⚠ AUCUN RetinaNet-R101 préentraîné COCO n'existe dans torchvision ni Detectron2.
#   (D2 ne fournit que retinanet_R_50 ; torchvision que retinanet_resnet50_fpn[_v2].)
#
# On construit ici un RetinaNet avec backbone ResNet-101 préentraîné ImageNet + FPN,
# en miroir exact de la construction torchvision de retinanet_resnet50_fpn, mais avec
# resnet101 à la place de resnet50. La TÊTE de détection est initialisée ALÉATOIREMENT.
#
# Conséquences :
#   • SPEED BENCHMARK valide : le coût de calcul (backbone + FPN + têtes conv) ne dépend
#     pas des valeurs des poids. De plus, torchvision initialise le biais de la tête de
#     classification avec prior_probability=0.01 → presque aucune détection ne passe le
#     seuil → NMS léger → le forward mesuré reflète bien l'architecture R101.
#   • MAP NON SIGNIFICATIVE : la tête n'étant pas entraînée, evaluate_map() renverra ~0.
#     À utiliser uniquement pour la vitesse et les optimisations (compile/FP16/TS/TRT).
#
# Interface identique à models/retinanet_r50.py (entrée List[Tensor[3,H,W]]) → drop-in
# pour benchmark_model(), run_map_evaluation() et le pipeline optimizations/.

_SCALE = 640.0


def _build_retinanet_r101(num_classes=91, min_size=640, max_size=640,
                          device="cuda", **kwargs):
    """Construit RetinaNet + ResNet-101-FPN (backbone ImageNet, tête aléatoire)."""
    backbone = resnet101(
        weights=ResNet101_Weights.IMAGENET1K_V2,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d,   # convention détection (BN figée)
    )
    # _resnet_fpn_extractor : même config que torchvision retinanet_resnet50_fpn
    #   returned_layers=[2,3,4] → C3,C4,C5 ; extra_blocks=P6,P7 calculés depuis P5
    backbone = _resnet_fpn_extractor(
        backbone, trainable_layers=3,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(256, 256),
    )
    model = RetinaNet(backbone, num_classes=num_classes,
                      min_size=min_size, max_size=max_size, **kwargs)
    return model.eval().to(device)


# ── Profiling ──────────────────────────────────────────────────────────────────

def load_model(device="cuda"):
    """640×640 — benchmark de vitesse standardisé. Backbone ImageNet, tête aléatoire."""
    return _build_retinanet_r101(min_size=640, max_size=640, device=device)


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
# ⚠ Tête non entraînée → MAP ~0. Conservé pour cohérence d'interface avec r50/fcos.

def load_model_eval(device="cuda"):
    """Résolution native (800/1333), score_thresh=0.01.
    ⚠ Tête aléatoire → MAP non significative (vitesse uniquement)."""
    print("[r101] ⚠ Tête de détection non entraînée — la MAP sera ~0 (vitesse uniquement).")
    return _build_retinanet_r101(min_size=800, max_size=1333,
                                 score_thresh=0.01, device=device)


def preprocess_eval(sample):
    """Load at original resolution → Tensor[3,H,W] float32 [0,1] CPU.
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
    """COCO-standard MAP evaluation. ⚠ Renvoie ~0 (tête non entraînée)."""
    from eval.map_eval import run_map_evaluation
    return run_map_evaluation(model, data, coco_gt, preprocess_eval, collate, postprocess_eval, device)
