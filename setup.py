"""
setup.py
────────
Initialisation complète du projet.

Actions :
  1. Création des répertoires du projet
  2. Téléchargement du dataset COCO val2017
       images      → datasets/coco/val2017/
       annotations → datasets/coco/annotations/
  3. Extraction de Detectron2
       clone  → temp (supprimé après)
       copie  → detectron2/   (package Python uniquement)

Usage :
    python setup.py                   # tout
    python setup.py --skip-coco       # seulement detectron2
    python setup.py --skip-d2         # seulement COCO
    python setup.py --coco-only       # alias --skip-d2
    python setup.py --d2-only         # alias --skip-coco
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# ── Chemins cibles ─────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
COCO_DIR    = ROOT / "datasets" / "coco"
VAL_DIR     = COCO_DIR / "val2017"
ANN_DIR     = COCO_DIR / "annotations"
ANN_FILE    = ANN_DIR  / "instances_val2017.json"
D2_TARGET   = ROOT / "detectron2"

COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANN_URL    = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
D2_REPO_URL     = "https://github.com/facebookresearch/detectron2.git"


# ── Utilitaires ────────────────────────────────────────────────────────────────

def _print(msg, level=0):
    prefix = "  " * level
    print(f"{prefix}{msg}", flush=True)


def _download(url, dest: Path):
    """Téléchargement streamé avec barre de progression minimale."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    filename = url.split("/")[-1]
    _print(f"Téléchargement : {filename}", 1)

    # Tente d'utiliser tqdm si disponible
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
        # Fallback sans tqdm
        def _hook(count, block_size, total_size):
            if total_size > 0:
                pct = min(100, count * block_size * 100 // total_size)
                print(f"\r  {filename} : {pct:3d}%", end="", flush=True)
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print()

    _print(f"OK → {dest}", 1)


def _extract(zip_path: Path, target_dir: Path):
    _print(f"Extraction → {target_dir}", 1)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    zip_path.unlink()
    _print("OK", 1)


# ── Étape 1 : répertoires ──────────────────────────────────────────────────────

def create_dirs():
    _print("Création des répertoires…")
    for d in [VAL_DIR, ANN_DIR,
              ROOT / "results",
              ROOT / "results" / "profiler" / "pytorch",
              ROOT / "results" / "profiler" / "nsight"]:
        d.mkdir(parents=True, exist_ok=True)
    _print("OK\n")


# ── Étape 2 : COCO val2017 ─────────────────────────────────────────────────────

def download_coco():
    _print("─── COCO val2017 ───────────────────────────────────────────")

    # Annotations
    if ANN_FILE.exists():
        _print("Annotations déjà présentes, skip.", 1)
    else:
        zip_path = COCO_DIR / "annotations_trainval2017.zip"
        _download(COCO_ANN_URL, zip_path)
        _extract(zip_path, COCO_DIR)

    # Images
    if VAL_DIR.exists() and any(VAL_DIR.iterdir()):
        _print("Images val2017 déjà présentes, skip.", 1)
    else:
        zip_path = COCO_DIR / "val2017.zip"
        _download(COCO_IMAGES_URL, zip_path)
        _extract(zip_path, COCO_DIR)   # extrait val2017/ directement sous datasets/coco/

    _print("COCO OK\n")


# ── Étape 3 : Detectron2 ───────────────────────────────────────────────────────

def setup_detectron2():
    _print("─── Detectron2 ─────────────────────────────────────────────")

    if D2_TARGET.exists() and (D2_TARGET / "__init__.py").exists():
        _print("detectron2/ déjà présent, skip.", 1)
        _print("Detectron2 OK\n")
        return

    # Clone dans un répertoire temporaire
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "d2_repo"
        _print(f"Clone du repo → {tmp_path}", 1)

        result = subprocess.run(
            ["git", "clone", "--depth=1", D2_REPO_URL, str(tmp_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            _print(f"ERREUR git clone :\n{result.stderr}", 1)
            sys.exit(1)

        # Le package Python est dans tmp_path/detectron2/
        pkg_src = tmp_path / "detectron2"
        if not pkg_src.exists():
            _print("ERREUR : dossier detectron2/ introuvable dans le clone.", 1)
            sys.exit(1)

        _print(f"Copie {pkg_src} → {D2_TARGET}", 1)
        if D2_TARGET.exists():
            shutil.rmtree(D2_TARGET)
        shutil.copytree(pkg_src, D2_TARGET)
        # tmp_path est supprimé automatiquement à la sortie du context manager

    _print("Detectron2 OK\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Initialisation du projet")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-coco", "--d2-only",  action="store_true",
                       help="Ne pas télécharger COCO")
    group.add_argument("--skip-d2",   "--coco-only", action="store_true",
                       help="Ne pas configurer Detectron2")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════════════════")
    print("  Setup ProjectDSAI2026_2")
    print("══════════════════════════════════════════════════════════\n")

    create_dirs()

    if not args.skip_coco:
        download_coco()

    if not args.skip_d2:
        setup_detectron2()

    print("══════════════════════════════════════════════════════════")
    print("  Initialisation terminée.")
    print("══════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
