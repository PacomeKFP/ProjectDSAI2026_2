"""
modal_test_env.py
═════════════════
Test d'environnement Modal : on part d'une image VIERGE (Debian slim), on
installe Python 3.13 + tout le stack DL depuis PyPI (comme sur Colab), puis
on vérifie que TRT, CUDA, torch.compile et tous les imports projet marchent.

Lancer : modal run modal_test_env.py
"""

import modal

app = modal.App("dsai2026-env-test")

# Image vierge — on construit tout. Pip résout les versions ensemble (pas de
# conflit numpy comme dans NGC). Numpy 2.x est laissé à pip, comme sur Colab.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # PyTorch CUDA bundled (les wheels PyPI torch incluent libcudart etc.)
        "torch", "torchvision",
        # TensorRT stack — pip résout en cohérence avec torch
        "tensorrt", "torch-tensorrt",
        # Détection
        "effdet", "timm",
        # Données / éval
        "pycocotools", "opencv-python-headless", "pillow",
        # Utilitaires
        "pandas", "numpy", "psutil", "tqdm",
        # Profilage
        "tensorboard", "nvtx",
        # ONNX
        "onnx", "onnxruntime-gpu", "onnxsim",
    )
)


@app.function(image=image, gpu="T4", timeout=600, cpu=2.0, memory=8192)
def test_env():
    import sys, platform
    print("=" * 70)
    print(f"Python      : {sys.version.split()[0]}  ({platform.platform()})")
    print("=" * 70)

    # Versions des paquets critiques
    pkgs = ["numpy", "torch", "torchvision", "tensorrt", "torch_tensorrt",
            "cv2", "pycocotools", "PIL", "pandas", "tqdm",
            "effdet", "timm", "onnx", "onnxruntime", "onnxsim"]
    print("\n── Versions des paquets ─────────────────────────────────────────")
    for name in pkgs:
        try:
            m = __import__(name)
            v = getattr(m, "__version__", "?")
            print(f"  {name:18s} {v}")
        except Exception as e:
            print(f"  {name:18s} FAIL  ({type(e).__name__}: {str(e)[:80]})")

    # CUDA et GPU
    print("\n── GPU / CUDA ───────────────────────────────────────────────────")
    import torch
    print(f"  CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU               : {torch.cuda.get_device_name(0)}")
        print(f"  Compute capability: {torch.cuda.get_device_capability(0)}")
        print(f"  VRAM              : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
        print(f"  torch CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ Pas de CUDA — test interrompu.")
        return {"status": "NO_CUDA"}

    # Test matmul GPU
    print("\n── Test matmul GPU ──────────────────────────────────────────────")
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    c = a @ b
    torch.cuda.synchronize()
    print(f"  Résultat  : shape={tuple(c.shape)}, device={c.device}, "
          f"mean={c.abs().mean().item():.3f}")

    # Test torch.compile (inductor)
    print("\n── Test torch.compile (inductor) ────────────────────────────────")
    try:
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding=1),
        ).cuda().eval()
        compiled = torch.compile(model, backend="inductor", mode="default")
        x = torch.randn(2, 3, 224, 224, device="cuda")
        with torch.no_grad():
            for _ in range(3):
                y = compiled(x)
            torch.cuda.synchronize()
        print(f"  torch.compile OK : output shape {tuple(y.shape)}")
    except Exception as e:
        print(f"  torch.compile FAIL : {type(e).__name__}: {str(e)[:200]}")

    # Test TensorRT FP16
    print("\n── Test TensorRT FP16 (torch_tensorrt) ──────────────────────────")
    try:
        import torch_tensorrt
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 3, 3, padding=1),
        ).cuda().eval()
        x = torch.randn(1, 3, 224, 224, device="cuda")
        trt_model = torch_tensorrt.compile(
            model,
            inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
            enabled_precisions={torch.float16},
        )
        with torch.no_grad():
            y = trt_model(x)
        torch.cuda.synchronize()
        print(f"  TRT compile OK   : output shape {tuple(y.shape)}, "
              f"dtype={y.dtype}")
    except Exception as e:
        print(f"  TRT compile FAIL : {type(e).__name__}: {str(e)[:300]}")

    # Test torchvision detection (R50)
    print("\n── Test torchvision detection ───────────────────────────────────")
    try:
        from torchvision.models.detection import retinanet_resnet50_fpn_v2
        m = retinanet_resnet50_fpn_v2(weights=None, num_classes=91).cuda().eval()
        x = [torch.randn(3, 640, 640, device="cuda")]
        with torch.no_grad():
            out = m(x)
        print(f"  RetinaNet R50 OK : {len(out)} sortie, keys={list(out[0].keys())}")
    except Exception as e:
        print(f"  RetinaNet R50 FAIL : {type(e).__name__}: {str(e)[:200]}")

    # Test effdet
    print("\n── Test effdet ──────────────────────────────────────────────────")
    try:
        from effdet import create_model
        m = create_model("tf_efficientdet_d0", pretrained=False, num_classes=90,
                         image_size=(640, 640), bench_task="predict").cuda().eval()
        x = torch.randn(1, 3, 640, 640, device="cuda")
        with torch.no_grad():
            out = m(x)
        print(f"  effdet D0 OK     : output shape {tuple(out.shape)}")
    except Exception as e:
        print(f"  effdet D0 FAIL   : {type(e).__name__}: {str(e)[:200]}")

    print("\n" + "=" * 70)
    print("  TEST TERMINÉ")
    print("=" * 70)
    return {"status": "OK"}


@app.local_entrypoint()
def main():
    print("Lancement du test environnement Modal (image vierge Python 3.13)...")
    result = test_env.remote()
    print(f"\nRésultat : {result}")
