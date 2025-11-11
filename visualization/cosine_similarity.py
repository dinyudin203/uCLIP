import sys
sys.path.append('..')

from datasets import load_dataset
from uclip.model import uCLIP, ModalityType
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel, AutoProcessor
from PIL import Image
from safetensors.torch import load_file
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from itertools import islice
import io
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import open_clip
from torchvision import transforms
from types import SimpleNamespace
import json
import os
from tqdm import tqdm


# model definition
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"

# test dataset
CAPTION_DIR = "../test_data/flickr"
IMAGE_ROOT = "../test_data/flickr/images"
LANGS = ["cs", "fi", "hr", "hu", "ro"]


transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),  # CLIP image input size
    transforms.ToTensor()
])

# parameters
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 512

args = {
    'model': 'mpnet',
    'vlm_model': 'openclip',
    'training': False,
}

args = SimpleNamespace(**args) 

# load model
text_processor = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)

uclip = uCLIP(args).eval()
uclip_checkpoint = load_file("../checkpoint/clip_encoder/model.safetensors")
_, _, clip_processor = open_clip.create_model_and_transforms(OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH)
uclip.load_state_dict(uclip_checkpoint, strict=False)
uclip.to(device)
uclip.eval()


# load dataset
def load_data(caption_json_path, image_root, max_samples=50):
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


for lang in LANGS:
    print(f"\nEvaluating language: {lang}")
    json_path = os.path.join(CAPTION_DIR, f"captions_{lang}.json")
    images, captions, filenames, errors = load_data(json_path, IMAGE_ROOT, max_samples=50)

    multilingual_text_embs_uclip_full = []
    img_embs_uclip_full = []
    # data load
    for i in range(0, len(images), BATCH_SIZE):
        images_batch = images[i:i+BATCH_SIZE]
        if len(images_batch) == 0:
            continue  
        texts_batch = captions[i : i + len(images_batch)]
        if len(texts_batch) != len(images_batch):
            print(f"[Warning] texts_batch length mismatch at i={i}: got {len(texts_batch)} vs expected {len(images_batch)}")
            continue
        flattened = {}

        # vision input
        clip_vision_inputs = [clip_processor(img) for img in images_batch]
        clip_vision_inputs = torch.stack(clip_vision_inputs).to(device)
        flattened["clip_vision_inputs"] = clip_vision_inputs

        # text input
        mul_inputs = text_processor(texts_batch, return_tensors="pt", padding=True, truncation=True).to(device)
        for k, v in mul_inputs.items():
            flattened[f"mul_{k}"] = v
            
        # get embeddings from uCLIP
        with torch.no_grad():
            # after projector
            outputs_post = uclip.get_test_embeddings(flattened)
            img_post = outputs_post[ModalityType.VISION]  # shape: (N, d)
            text_post = outputs_post[ModalityType.MULTILINGUAL_TEXT]  # shape: (N, d)

        multilingual_text_embs_uclip_full.append(text_post.to('cpu'))
        img_embs_uclip_full.append(img_post.to('cpu'))

    # cosine similarity
    mul_all = torch.cat(multilingual_text_embs_uclip_full, dim=0)  # shape: [N, 512]
    img_all = torch.cat(img_embs_uclip_full, dim=0) # shape: [N, 512]
    mul_all = F.normalize(mul_all, dim=-1)
    img_all = F.normalize(img_all, dim=-1)
    sim_post = cosine_similarity(mul_all.cpu().numpy(), img_all.cpu().numpy())

    # heatmap visualization
    import matplotlib as mpl
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 70

    sim_matrices = [sim_post]
    titles = ["uclip"]
    save_dir = "./cosine_similarity"
    os.makedirs(save_dir, exist_ok=True)

    for sim_matrix, title in zip(sim_matrices, titles):
        fig, ax = plt.subplots(figsize=(15, 15))

        sns.heatmap(sim_matrix, cmap='coolwarm', ax=ax, cbar=False, square=True)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.axis('off') 

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        plt.savefig(f"{save_dir}/uclip_cosine_similarity_{lang}_{title}.png", 
                    bbox_inches='tight', pad_inches=0) # Ensure no padding when saving
        plt.close()
        