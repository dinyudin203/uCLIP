import sys
sys.path.append('..')

import torch
from transformers import AutoTokenizer
from torchvision import transforms
from safetensors.torch import load_file
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from uclip.model import uCLIP, ModalityType
import open_clip
import os
import json
import argparse
import time

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mscoco", choices=["mscoco", "flickr", "xm3600"])
args = parser.parse_args()

# === Configuration ===
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1
LANGS = ["cs", "fi", "hr", "hu", "ro"]
CAPTION_DIR = f"../test_data/{args.dataset}"
IMAGE_ROOT = f"../test_data/{args.dataset}/images"
MAX_SAMPLES = 500

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# === Model Loading (OpenCLIP only) ===
model_args = SimpleNamespace(model='mpnet', vlm_model='openclip', training=False)
text_tokenizer = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)

uclip = uCLIP(model_args).eval()
uclip_checkpoint = load_file("../checkpoint/clip_encoder/model.safetensors")
uclip.load_state_dict(uclip_checkpoint, strict=False)
uclip.to(device).eval()

# OpenCLIP image processor
_, _, clip_processor = open_clip.create_model_and_transforms(
    OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH
)

# === Dataset Loader ===
def load_data(caption_json_path, image_root, max_samples=5000):
    """
    Loads image-caption pairs from JSON and corresponding image files.
    Returns images, captions, filenames, and load errors.
    """
    with open(caption_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)[:max_samples]
    images, captions, filenames, errors = [], [], [], []
    for item in tqdm(data, desc=f"Loading {os.path.basename(caption_json_path)}"):
        try:
            img = Image.open(os.path.join(image_root, item["filename"])).convert("RGB")
            images.append(img)
            captions.append(item["captions_translated"])
            filenames.append(item["filename"])
        except Exception as e:
            errors.append({"filename": item["filename"], "error": str(e)})
    return images, captions, filenames, errors

# === Inference Timing Script ===
results = {}

for lang in LANGS:
    print(f"\nEvaluating inference time for language: {lang}")
    json_path = os.path.join(CAPTION_DIR, f"captions_{lang}.json")
    images, captions, filenames, errors = load_data(json_path, IMAGE_ROOT, max_samples=MAX_SAMPLES)

    image_times, text_times = [], []

    for i in range(0, len(images), BATCH_SIZE):
        images_batch = images[i:i+BATCH_SIZE]
        texts_batch = captions[i:i+len(images_batch)]

        if len(images_batch) == 0 or len(texts_batch) != len(images_batch):
            continue

        # === Image Preprocessing and Timing ===
        start_img = torch.cuda.Event(enable_timing=True)
        end_img = torch.cuda.Event(enable_timing=True)

        start_img.record()
        img_inputs = torch.stack([clip_processor(img) for img in images_batch]).to(device)
        end_img.record()
        torch.cuda.synchronize()
        image_times.append(start_img.elapsed_time(end_img) / len(images_batch))  # ms/sample

        # === Text Preprocessing and Timing ===
        start_txt = torch.cuda.Event(enable_timing=True)
        end_txt = torch.cuda.Event(enable_timing=True)

        start_txt.record()
        text_inputs = text_tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        end_txt.record()
        torch.cuda.synchronize()
        text_times.append(start_txt.elapsed_time(end_txt) / len(texts_batch))  # ms/sample

        # === Forward Pass (Not Timed) ===
        with torch.no_grad():
            flattened = {
                "clip_vision_inputs": img_inputs,
                **{f"mul_{k}": v for k, v in text_inputs.items()}
            }
            _ = uclip.get_test_embeddings(flattened)

    # === Averaged Timing per Language ===
    avg_image_time = sum(image_times) / len(image_times)
    avg_text_time = sum(text_times) / len(text_times)

    results[lang] = {
        "num_images": len(images),
        "num_errors": len(errors),
        "timing": {
            "avg_image_encoding_time_ms": round(avg_image_time, 2),
            "avg_text_encoding_time_ms": round(avg_text_time, 2),
            "total_avg_inference_time_ms": round(avg_image_time + avg_text_time, 2)
        }
    }

# === Save Timing Results ===
save_dir = f"./retrieval_results/{args.dataset}"
os.makedirs(save_dir, exist_ok=True)
save_path = f"{save_dir}/uclip_inference_time_openclip.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved inference timing results to: {save_path}")
