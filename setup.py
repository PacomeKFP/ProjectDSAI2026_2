"""
setup.py
--------
Full project initialization.

Actions:
  1. pip dependency installation (requirements.txt) -- including effdet, timm,
     torchvision, pycocotools, etc. (NOT the optimization dependencies: those
     are installed in optimization_full.ipynb).
  2. Creation of project directories.
  3. Download of the COCO val2017 dataset
       images      -> datasets/coco/val2017/
       annotations -> datasets/coco/annotations/
  4. Detectron2 extraction
       clone  -> temp (removed afterwards)
       copy   -> detectron2/   (Python package only)

Note: RetinaNet R101 is now rebuilt from torchvision (resnet101 + FPN).
Detectron2 is kept as a safeguard (usable if needed) but is no longer required
by the models.

Usage:
    python setup.py                   # everything
    python setup.py --skip-coco       # without downloading COCO
    python setup.py --skip-deps       # without installing pip dependencies
    python setup.py --skip-d2         # without configuring Detectron2
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# -- Target paths --------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
COCO_DIR    = ROOT / "datasets" / "coco"
VAL_DIR     = COCO_DIR / "val2017"
ANN_DIR     = COCO_DIR / "annotations"
ANN_FILE    = ANN_DIR  / "instances_val2017.json"
D2_TARGET   = ROOT / "detectron2"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
D2_REPO_URL     = "https://github.com/facebookresearch/detectron2.git"


# -- Utilities ------------------------------------------------------------------

def _print(msg, level=0):
    prefix = "  " * level
    print(f"{prefix}{msg}", flush=True)


def _download(url, dest: Path):
    """Streamed download with a minimal progress bar."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    _print(f"Downloading: {filename}", 1)

    # Try to use tqdm if available
    try:
        from tqdm import tqdm

        class _TqdmUpTo(tqdm):
            def update_to(self, b=1, bsize=1, tsize=None):
                if tsize is not None:
                    self.total = tsize
                self.update(b * bsize - self.n)

        with _TqdmUpTo(unit="B", unit_scale=True, miniters=1,
                       desc=f"  {filename}") as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    except ImportError:
        # Fallback without tqdm
        def _hook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r  {filename} : {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print()

    _print(f"OK -> {dest}", 1)


def _extract(zip_path: Path, target_dir: Path):
    _print(f"Extracting -> {target_dir}", 1)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    zip_path.unlink()
    _print("OK", 1)


# -- Step 1: pip dependencies ---------------------------------------------------

def install_dependencies():
    _print("--- pip dependencies ---------------------------------------")
    if not REQUIREMENTS.exists():
        _print("requirements.txt not found, skipping.", 1)
        return
    _print("pip install -r requirements.txt", 1)
    # Output is not captured -> the user sees pip's progress.
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        text=True,
    )
    if result.returncode != 0:
        _print("ERROR: pip install failed (see output above).", 1)
        sys.exit(1)
    _print("Dependencies OK\n")


# -- Step 2: directories --------------------------------------------------------

def create_dirs():
    _print("Creating directories...")
    for d in [VAL_DIR, ANN_DIR,
              ROOT / "results",
              ROOT / "results" / "profiler" / "pytorch",
              ROOT / "results" / "profiler" / "nsight",
              ROOT / "results" / "optimization",
              ROOT / "outputs" / "models",
              ROOT / "outputs" / "onnx"]:
        d.mkdir(parents=True, exist_ok=True)
    _print("OK\n")


# -- Step 3: COCO val2017 -------------------------------------------------------

def download_coco():
    _print("--- COCO val2017 -------------------------------------------")

    # Annotations
    if ANN_FILE.exists():
        _print("Annotations already present, skipping.", 1)
    else:
        zip_path = COCO_DIR / "annotations_trainval2017.zip"
        _download(COCO_ANN_URL, zip_path)
        _extract(zip_path, COCO_DIR)

    # Images
    if VAL_DIR.exists() and any(VAL_DIR.iterdir()):
        _print("val2017 images already present, skipping.", 1)
    else:
        zip_path = COCO_DIR / "val2017.zip"
        _download(COCO_IMAGES_URL, zip_path)
        _extract(zip_path, COCO_DIR)   # extracts val2017/ directly under datasets/coco/

    _print("COCO OK\n")


# -- Step 4: Detectron2 (kept as a safeguard) ----------------------------------

def setup_detectron2():
    _print("--- Detectron2 ---------------------------------------------")

    if D2_TARGET.exists() and (D2_TARGET / "__init__.py").exists():
        _print("detectron2/ already present, skipping.", 1)
        _print("Detectron2 OK\n")
        return

    # Clone into a temporary directory
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "d2_repo"
        _print(f"Cloning repo -> {tmp_path}", 1)

        result = subprocess.run(
            ["git", "clone", "--depth=1", D2_REPO_URL, str(tmp_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _print(f"ERROR: git clone failed:\n{result.stderr}", 1)
            sys.exit(1)

        # The Python package lives at tmp_path/detectron2/
        pkg_src = tmp_path / "detectron2"
        if not pkg_src.exists():
            _print("ERROR: detectron2/ directory not found in the clone.", 1)
            sys.exit(1)

        _print(f"Copying {pkg_src} -> {D2_TARGET}", 1)
        if D2_TARGET.exists():
            shutil.rmtree(D2_TARGET)
        shutil.copytree(pkg_src, D2_TARGET)
        # tmp_path is removed automatically on exit from the context manager

    _print("Detectron2 OK\n")


# -- Main -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Project initialization")
    parser.add_argument("--skip-deps", action="store_true",
                        help="Do not install pip dependencies")
    parser.add_argument("--skip-coco", action="store_true",
                        help="Do not download COCO")
    parser.add_argument("--skip-d2", action="store_true",
                        help="Do not configure Detectron2")
    args = parser.parse_args()

    print("\n==========================================================")
    print("  Setup ProjectDSAI2026_2")
    print("==========================================================\n")

    if not args.skip_deps:
        install_dependencies()

    create_dirs()

    if not args.skip_coco:
        download_coco()

    if not args.skip_d2:
        setup_detectron2()

    print("==========================================================")
    print("  Initialization complete.")
    print("==========================================================\n")


if __name__ == "__main__":
    main()
