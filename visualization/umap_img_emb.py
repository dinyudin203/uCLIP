import sys
sys.path.append('..')

import open_clip
import torch
from torchvision.datasets import CIFAR10, STL10, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, AutoTokenizer
from uclip.model import uCLIP, ModalityType
from safetensors.torch import load_file
import matplotlib as mpl
from PIL import Image
from types import SimpleNamespace
import os
import argparse


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--vlm_model", type=str, default="openclip", choices=["openclip", "siglip2"])
args = parser.parse_args()

# model definition
vlm_model = args.vlm_model
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"
SIGLIP_MODEL_PATH = "google/siglip2-base-patch32-256"


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

to_pil = transforms.ToPILImage()

dataset = CIFAR10(root="path_to_cifar10_dataset", train=False, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=256, shuffle=False)

# load model
if vlm_model == 'openclip':
    args = {
        'model': 'mpnet',
        'vlm_model': 'openclip',
        'training': False
    }
elif vlm_model == 'siglip2':
    args = {
        'model': 'mpnet',
        'vlm_model': 'siglip2',
        'training': False
    }

args = SimpleNamespace(**args) 
text_processor = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)

uclip = uCLIP(args).eval()

if vlm_model == 'openclip':
    uclip_checkpoint = load_file("../checkpoint/clip_encoder/model.safetensors")
    _, _, clip_processor = open_clip.create_model_and_transforms(OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH)
elif vlm_model == 'siglip2':
    uclip_checkpoint = load_file("../checkpoint/siglip_encoder/model.safetensors")
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)

uclip.load_state_dict(uclip_checkpoint, strict=False)
uclip.to(device)
uclip.eval()


# Embedding extraction
def extract_image_embeddings(model_name):
    embeddings = []
    labels = []
    for images, lbls in tqdm(loader, desc=f"Extracting {model_name}"):
        with torch.no_grad():

            if vlm_model == 'openclip':
                clip_vision_inputs = [clip_processor(to_pil(img)) for img in images]
                clip_vision_inputs = torch.stack(clip_vision_inputs).to(device)
                vision_features = uclip.trunk.get_vision_feature(clip_vision_inputs)
                projected = uclip.project_features({ModalityType.VISION: vision_features})
                embs = projected[ModalityType.VISION].cpu().numpy()
            elif vlm_model == 'siglip2':
                images_pil = [to_pil(img.cpu()) for img in images]
                vision_inputs = siglip_processor(images=images_pil, return_tensors="pt").to(device)
                vision_input = {"pixel_values": vision_inputs["pixel_values"]}
                vision_features = uclip.trunk.get_vision_feature(vision_input)
                projected = uclip.project_features({ModalityType.VISION: vision_features})
                embs = projected[ModalityType.VISION].cpu().numpy()
        embeddings.append(embs)
        labels.extend(lbls.numpy())
    return np.vstack(embeddings), np.array(labels)


# Collect and plot
for model_name in ["uCLIP"]:
    # set up
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 60
    save_dir = "./umap_cifar10"
    os.makedirs(save_dir, exist_ok=True)

    # extract embeddings
    embs, lbls = extract_image_embeddings(model_name)
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced = reducer.fit_transform(embs)

    # plot
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=lbls, cmap='tab10', s=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/cifar10_umap_{model_name}_{vlm_model}.png")
    plt.close()
