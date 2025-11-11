# === Imports and Setup ===
import argparse
import os
import torch
import gc
from tqdm import tqdm
from PIL import Image, ImageFile
from io import BytesIO
from datasets import load_dataset
from huggingface_hub import HfFolder
import torch.nn.functional as F
import open_clip
import requests

# Enable loading of truncated images to prevent decoding issues
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Configuration ===
# Model and data parameters
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"
batch_size = 64
SAVE_DIR = "embeddings"
COCO_DIR = "~/coco/images/train2017"   # Path to COCO 2017 training images
CC12M_DIR = "~/cc12m/images"           # Path to CC12M images

# Default sampling limits per dataset
DEFAULT_SAMPLE_LIMITS = {
    "imagenet": 1_200_000,
    "cc12m": 12_000_000,
    "coco": 110_000,
}

# Setup device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    model_name=OPENCLIP_MODEL_BASE,
    pretrained=OPENCLIP_MODEL_PATH
)
model = model.to(device).eval()

# === Argument Parsing ===
# Allows specification of dataset source, offset, and sample limit
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, required=True, choices=["imagenet", "cc12m", "coco"])
parser.add_argument("--offset", type=int, default=0)
parser.add_argument("--limit", type=int, default=None)
args = parser.parse_args()

# === Image Loading Function ===
def load_image(path_or_url, is_url=False):
    """
    Loads an image from a local path or URL and converts it to RGB.
    Returns None if the image cannot be loaded.
    """
    try:
        if is_url:
            response = requests.get(path_or_url, timeout=5)
            return Image.open(BytesIO(response.content)).convert("RGB")
        return Image.open(path_or_url).convert("RGB")
    except Exception:
        return None

# === Feature Extraction Function ===
@torch.no_grad()
def get_vision_feature(source, offset=0, limit=None):
    """
    Extracts and saves normalized CLIP vision features for images from a specified dataset source.
    Supports COCO, CC12M, and ImageNet datasets.
    """
    image_tensors = torch.stack([preprocess_val(img) for img in images]).to(device)
    outputs = model.encode_image(image_tensors)
    outputs = F.normalize(outputs, dim=-1).cpu()

    # Determine sampling range
    full_limit = DEFAULT_SAMPLE_LIMITS[source]
    limit = limit or full_limit
    end_idx = offset + limit

    os.makedirs(f"{SAVE_DIR}/{source}", exist_ok=True)
    image_count, batch_id = 0, 0
    batch = []

    def process_batch(batch_images, batch_id, offset):
        """
        Processes a batch of images: runs preprocessing, model inference, normalization, and saves output.
        """
        valid_images = []
        for img in batch_images:
            try:
                _ = clip_processor(images=img, return_tensors="pt", padding=True)
                valid_images.append(img)
            except Exception as e:
                print(f"Preprocessing failed: {e}")
                continue

        if not valid_images:
            print(f"No valid images - Skipping batch {batch_id}")
            return

        try:
            inputs = processor(images=valid_images, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1).cpu()
            torch.save(features, f"{SAVE_DIR}/{source}/offset{offset}_batch_{batch_id:05d}.pt")
            print(f"Saved batch {batch_id} ({len(valid_images)} images)")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"OOM - Skipping batch {batch_id}")
            else:
                print(f"Runtime error during processing: {e}")
        finally:
            del inputs, features, valid_images
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Starting: {source.upper()} / offset={offset} / limit={limit}")

    # === COCO Dataset Loader ===
    if source == "coco":
        files = sorted([
            os.path.join(COCO_DIR, f)
            for f in os.listdir(COCO_DIR)
            if f.endswith(".jpg")
        ])
        files = files[offset:end_idx]
        for path in tqdm(files):
            img = load_image(path)
            if img:
                batch.append(img)
                image_count += 1
            if len(batch) == batch_size:
                process_batch(batch, batch_id, offset)
                batch, batch_id = [], batch_id + 1
            if image_count >= limit:
                break

    # === CC12M Dataset Loader ===
    elif source == "cc12m":
        # Assumes CC12M is locally available in a folder
        files = sorted([
            os.path.join(CC12M_DIR, f)
            for f in os.listdir(CC12M_DIR)
            if f.endswith(".jpg")
        ])
        files = files[offset:end_idx]
        for i, path in enumerate(tqdm(files, desc="CC12M")):
            img = load_image(path)
            if img:
                batch.append(img)
                image_count += 1
            if len(batch) == batch_size:
                process_batch(batch, batch_id, offset)
                batch, batch_id = [], batch_id + 1
            if image_count >= limit:
                break

    # === ImageNet Dataset Loader ===
    elif source == "imagenet":
        # Loads ImageNet-1k via HuggingFace streaming API
        token = HfFolder.get_token()
        dataset = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=token)
        for i, ex in enumerate(tqdm(dataset, desc="ImageNet")):
            if i < offset:
                continue
            if i >= end_idx:
                break

            try:
                img_data = ex.get("image")
                if isinstance(img_data, Image.Image):
                    img = img_data.convert("RGB")
                elif isinstance(img_data, bytes):
                    img = Image.open(BytesIO(img_data)).convert("RGB")
                else:
                    print(f"[{i}] Unsupported image type: {type(img_data)}")
                    continue

                batch.append(img)
                image_count += 1
            except Exception as e:
                print(f"[{i}] Failed to load image: {e}")
                continue

            if len(batch) == batch_size:
                process_batch(batch, batch_id, offset)
                batch, batch_id = [], batch_id + 1

            if image_count >= limit:
                break

# === Entry Point ===
if __name__ == "__main__":
    get_vision_feature(args.source, args.offset, args.limit)
