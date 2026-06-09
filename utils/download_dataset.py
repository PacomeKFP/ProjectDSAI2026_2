import os
import zipfile
import requests

def download_coco_val(root_dir="datasets/coco"):
    os.makedirs(root_dir, exist_ok=True)

    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "images": "http://images.cocodataset.org/zips/val2017.zip"
    }

    ann_zip_path = os.path.join(root_dir, "annotations_trainval2017.zip")
    ann_file = os.path.join(root_dir, "annotations", "instances_val2017.json")

    if not os.path.exists(ann_file):
        print("Téléchargement des annotations...")
        r = requests.get(urls["annotations"], stream=True)

        with open(ann_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(ann_zip_path, "r") as z:
            z.extractall(root_dir)

        os.remove(ann_zip_path)

    img_zip_path = os.path.join(root_dir, "val2017.zip")
    img_dir = os.path.join(root_dir, "val2017")

    if not os.path.exists(img_dir):
        print("Téléchargement des images (~1 GB)...")
        r = requests.get(urls["images"], stream=True)

        with open(img_zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        with zipfile.ZipFile(img_zip_path, "r") as z:
            z.extractall(root_dir)

        os.remove(img_zip_path)

    return img_dir, ann_file