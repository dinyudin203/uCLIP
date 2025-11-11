import sys
sys.path.append('..')

from transformers import AutoTokenizer, AutoProcessor
from torchvision import transforms
from safetensors.torch import load_file
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from types import SimpleNamespace
from uclip.model import uCLIP, ModalityType

import torch
import open_clip
import os
import csv
import argparse
import json


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--vlm_model", type=str, default="openclip", choices=["openclip", "siglip2"])
parser.add_argument("--dataset", type=str, default="mscoco", choices=["mscoco", "flickr", "xm3600"])
args = parser.parse_args()

# model definition
vlm_model = args.vlm_model
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"
SIGLIP_MODEL_PATH = "google/siglip2-base-patch32-256"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocess
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)), 
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()


CAPTION_DIR = f"../test_data/{args.dataset}"
IMAGE_ROOT = f"../test_data/{args.dataset}/images"
BATCH_SIZE = 512
LANGS = ["cs","fi","hr","hu","ro"]


# load model
if vlm_model == 'openclip':
    model_args = {
        'model': 'mpnet',
        'vlm_model': 'openclip',
        'training': False
    }
elif vlm_model == 'siglip2':
    model_args = {
        'model': 'mpnet',
        'vlm_model': 'siglip2',
        'training': False
    }


model_args = SimpleNamespace(**model_args) 
text_processor = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)

uclip = uCLIP(model_args).eval()

if vlm_model == 'openclip':
    uclip_checkpoint = load_file("../checkpoint/clip_encoder/model.safetensors")
    _, _, clip_processor = open_clip.create_model_and_transforms(OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH)
elif vlm_model == 'siglip2':
    uclip_checkpoint = load_file("../checkpoint/siglip_encoder/model.safetensors")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

uclip.load_state_dict(uclip_checkpoint, strict=False)
uclip.to(device)
uclip.eval()


def compute_recall_at_k(image_features, text_features, captions_per_image=1, k_list=[1, 5, 10]):
    """
    image_features: (N, D) tensor, normalized
    text_features: (N*C, D) tensor, normalized
    captions_per_image: how many captions per image (C)
    """
    num_images = image_features.shape[0]
    num_captions = text_features.shape[0]
    assert num_captions == num_images * captions_per_image

    # (N, N*C) similarity matrix
    sims = image_features @ text_features.T

    recalls_image_to_text = {f"Recall@{k}": 0.0 for k in k_list}

    for img_idx in range(num_images):
        # ground truth caption indices
        gt_caption_indices = list(range(img_idx * captions_per_image, (img_idx + 1) * captions_per_image))
        ranking = torch.argsort(sims[img_idx], descending=True)

        # rank is success if any of the ground truth captions are in the top-k
        for k in k_list:
            top_k = ranking[:k].tolist()
            if any(gt in top_k for gt in gt_caption_indices):
                recalls_image_to_text[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_image_to_text[f"Recall@{k}"] /= num_images

    # Text-to-Image: each caption has one ground truth image
    sims_text_to_image = text_features @ image_features.T
    recalls_text_to_image = {f"Recall@{k}": 0.0 for k in k_list}

    for cap_idx in range(num_captions):
        gt_image_idx = cap_idx // captions_per_image
        ranking = torch.argsort(sims_text_to_image[cap_idx], descending=True)
        rank = (ranking == gt_image_idx).nonzero(as_tuple=True)[0].item() + 1

        for k in k_list:
            if rank <= k:
                recalls_text_to_image[f"Recall@{k}"] += 1

    for k in k_list:
        recalls_text_to_image[f"Recall@{k}"] /= num_captions

    return recalls_image_to_text, recalls_text_to_image

# Load image-caption pairs
def load_data(caption_json_path, image_root, max_samples=5000):
    with open(caption_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:max_samples]  # Limit to max_samples if needed
    images, captions, filenames = [], [], []
    errors = []

    for item in tqdm(data, desc=f"Loading {os.path.basename(caption_json_path)}"):
        filename = item["filename"]
        try:
            path = os.path.join(image_root, filename)
            img = Image.open(path).convert("RGB")
            images.append(img)
            captions.append(item["captions_translated"])
            filenames.append(filename)
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})
    return images, captions, filenames, errors

# main loop
results = {}
for lang in LANGS:
    print(f"\nEvaluating language: {lang}")
    json_path = os.path.join(CAPTION_DIR, f"captions_{lang}.json")
    images, captions, filenames, errors = load_data(json_path, IMAGE_ROOT, max_samples=5000)


    multilingual_text_embs_uclip_full = []
    img_embs_uclip_full = []

    for i in range(0, len(images), BATCH_SIZE):
      images_batch = images[i:i+BATCH_SIZE]
      if len(images_batch) == 0:
          continue  # skip empty batch

      # each image has one text
      texts_batch = captions[i : i + len(images_batch)]

      if len(texts_batch) != len(images_batch):
          print(f"[Warning] texts_batch length mismatch at i={i}: got {len(texts_batch)} vs expected {len(images_batch)}")
          continue

      flattened = {}
      if vlm_model == 'openclip':
        clip_vision_inputs = torch.stack([clip_processor(img) for img in images_batch]).to(device)
        flattened["clip_vision_inputs"] = clip_vision_inputs
      elif vlm_model == 'siglip2':
        siglip_vision_inputs = torch.stack([siglip_processor(img) for img in images_batch]).to(device)
        flattened["siglip_vision_inputs"] = siglip_vision_inputs
      mul_inputs = text_processor(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
      for k, v in mul_inputs.items():
          flattened[f"mul_{k}"] = v

      # extract embeddings
      with torch.no_grad():
          outputs = uclip.get_test_embeddings(flattened)
          multilingual_text_embs_uclip, img_embs_uclip = outputs[ModalityType.MULTILINGUAL_TEXT], outputs[ModalityType.VISION]
      
      multilingual_text_embs_uclip_full.append(multilingual_text_embs_uclip.to('cpu'))
      img_embs_uclip_full.append(img_embs_uclip.to('cpu'))

    multilingual_text_embs_uclip_full = torch.cat(multilingual_text_embs_uclip_full, dim=0).to(device)
    img_embs_uclip_full = torch.cat(img_embs_uclip_full, dim=0).to(device)

    # Recall@1,5,10
    recall_i2t, recall_t2i = compute_recall_at_k(img_embs_uclip_full, multilingual_text_embs_uclip_full)
    
    # Store results for this language
    results[lang] = {
        "Image_to_Text": recall_i2t,
        "Text_to_Image": recall_t2i,
        "num_images": len(images),
        "num_errors": len(errors)
    }
    
    print(f"Results for {lang}:")
    print(f"  Image-to-Text: {recall_i2t}")
    print(f"  Text-to-Image: {recall_t2i}")


# After processing all languages, prepare results for CSV
csv_rows = []
for lang_key, res in results.items():
    row = {
        "lang": lang_key,
        "i2t@1": res.get("Image_to_Text", {}).get("Recall@1", 0.0),
        "i2t@5": res.get("Image_to_Text", {}).get("Recall@5", 0.0),
        "i2t@10": res.get("Image_to_Text", {}).get("Recall@10", 0.0),
        "t2i@1": res.get("Text_to_Image", {}).get("Recall@1", 0.0),
        "t2i@5": res.get("Text_to_Image", {}).get("Recall@5", 0.0),
        "t2i@10": res.get("Text_to_Image", {}).get("Recall@10", 0.0),
    }
    csv_rows.append(row)

csv_fieldnames = ["lang", "i2t@1", "i2t@5", "i2t@10", "t2i@1", "t2i@5", "t2i@10"]

save_dir = f"./retrieval_results/{args.dataset}"
os.makedirs(save_dir, exist_ok=True)
out_csv = f"{save_dir}/retrieval_uclip_{vlm_model}.csv"
with open(out_csv, "w", encoding="utf-8", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
    writer.writeheader()
    for row in csv_rows:
        writer.writerow(row)
print(f"Saved retrieval results to {out_csv}")
