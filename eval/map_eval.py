"""
map_eval.py
-----------
COCO-standard MAP evaluation loop.

Memory strategy:
  * Images are NOT preloaded. preprocess_fn() reads from disk per chunk.
  * Batch size defaults to auto-estimated from free VRAM/RAM (conservative 30%).
  * Each chunk is freed immediately after postprocessing (gc.collect + empty_cache).
  * Works on machines with 1.6 GB RAM + 8 GB VRAM; adapts automatically to more.
"""
import gc

import torch
from pycocotools.cocoeval import COCOeval

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

from utils.tqdm_compat import tqdm


# -- Memory helper --------------------------------------------------------------

def _estimate_batch_size(device="cuda", image_h=640, image_w=640,
                         safety=0.3, max_batch=32):
    """
    Auto batch_size based on available VRAM (GPU) or RAM (CPU).

    Heuristic: 50x the input tensor per image.
      - The 10x factor heavily underestimates detection models with FPN:
        conv1 alone produces H/2 x W/2 x 64 activations, and the P3-P7
        pyramids accumulate feature maps that all coexist in memory during
        the forward pass.
      - At native COCO resolution (~800-1333 px), activations are 3-5x larger
        than at 640x640. The 50x factor gives a conservative estimate that
        covers backbone + FPN + detection heads.
      - max_batch=32: hard cap to avoid OOM on high-resolution images.
    """
    bytes_per_img = image_h * image_w * 3 * 4 * 50

    if device != "cpu" and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
    elif _HAS_PSUTIL:
        free_bytes = psutil.virtual_memory().available
    else:
        return 4  # safe fallback

    usable = int(free_bytes * safety)
    return max(1, min(usable // bytes_per_img, max_batch))


# -- Main evaluation loop -------------------------------------------------------

def run_map_evaluation(
    model,
    data,
    coco_gt,
    preprocess_fn,
    collate_fn,
    postprocess_fn,
    device="cuda",
    batch_size=None,
):
    """
    Shared COCO MAP evaluation.

    Parameters
    ----------
    model          : nn.Module -- from load_model_eval()
    data           : list of {'path', 'image_id', 'orig_size'} -- load_eval_data()
    coco_gt        : pycocotools COCO object (full annotations)
    preprocess_fn  : reads image from disk + prepares CPU tensor
    collate_fn     : moves list of CPU tensors to device
    postprocess_fn : (raw_item, orig_size) -> {'boxes', 'labels', 'scores'}
    device         : 'cuda' or 'cpu'
    batch_size     : None = auto-estimate from free memory

    Returns
    -------
    dict with AP, AP50, AP75, APs, APm, APl  (-1 if no predictions)
    """
    if batch_size is None:
        batch_size = _estimate_batch_size(device)

    predictions = []
    model.eval()

    n_batches = (len(data) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(data), batch_size), total=n_batches,
                  desc="  MAP eval", leave=False):
        chunk = data[i : i + batch_size]

        # Load + preprocess this chunk on CPU (one image at a time from disk)
        cpu_inputs = [preprocess_fn(s) for s in chunk]

        with torch.no_grad():
            batch = collate_fn(cpu_inputs, device)
            raw   = model(batch)
            # TorchScript torchvision detection model: forward returns
            # (losses, detections) instead of detections alone. Keep detections.
            if (isinstance(raw, tuple) and len(raw) == 2
                    and isinstance(raw[0], dict) and isinstance(raw[1], list)):
                raw = raw[1]
            # Move raw predictions to CPU immediately to free VRAM
            if isinstance(raw, (list, tuple)):
                raw_cpu = [
                    {k: v.cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in r.items()}
                    if isinstance(r, dict) else r.cpu()
                    for r in raw
                ]
            else:
                raw_cpu = raw.cpu()
            del batch

        # Postprocess + accumulate
        for j, s in enumerate(chunk):
            pred   = postprocess_fn(raw_cpu[j], s["orig_size"])
            boxes  = pred["boxes"].cpu()
            labels = pred["labels"].cpu()
            scores = pred["scores"].cpu()

            # xyxy -> xywh (COCO annotation format)
            xywh = boxes.clone()
            xywh[:, 2] -= xywh[:, 0]
            xywh[:, 3] -= xywh[:, 1]

            for box, label, score in zip(xywh, labels, scores):
                predictions.append({
                    "image_id":    s["image_id"],
                    "category_id": int(label),
                    "bbox":        box.tolist(),
                    "score":       float(score),
                })

        # Free this chunk's memory
        del cpu_inputs, raw_cpu
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if not predictions:
        return {k: -1.0 for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")}

    img_ids   = [s["image_id"] for s in data]
    coco_dt   = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = img_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    s = evaluator.stats
    return {
        "AP":   float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "APs":  float(s[3]),
        "APm":  float(s[4]),
        "APl":  float(s[5]),
    }
