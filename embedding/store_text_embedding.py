# === Imports and Configuration ===
import os
import torch
from sentence_transformers import SentenceTransformer
from multiprocessing import Process, set_start_method
import config

# Input/output paths
TEXT_PATH = "nllb_captions_txt"          # Directory containing multilingual caption text files
OUTPUT_PATH = "nllb_captions_txt/e5"     # Output directory for embedding files
MODEL_NAME = config.E5_MODEL_PATH        # Path or name of the pretrained E5 model

# List of target languages (ISO 639-1 codes)
lang_list = ["cs", "hr", "fi", "hu", "ro"]
BATCH_SIZE = 1024                         # Number of sentences per encoding batch

# === Embedding Function ===
def embed_language(l, device_id):
    """
    Encodes sentences from a given language using the E5 model and saves the embeddings.
    Each language is processed on a separate GPU.

    Args:
        l (str): Language code (e.g., "cs", "hu").
        device_id (int): Index of the CUDA device to use.
    """
    device = f"cuda:{device_id}"
    print(f"[{l}] Starting on GPU {device}")

    # Load sentence transformer model on specified GPU
    model = SentenceTransformer(MODEL_NAME, device=device)

    # Load sentences from text file
    with open(f"{TEXT_PATH}/captions_{l}.txt", "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    all_embeddings = []

    # Encode sentences in batches
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i:i + BATCH_SIZE]
        with torch.no_grad():
            embeddings = model.encode(
                batch,
                convert_to_tensor=True,
                normalize_embeddings=True,  # Unit-norm embeddings
                device=device
            )
        all_embeddings.append(embeddings)

    # Concatenate all embeddings and move to CPU
    embeddings = torch.cat(all_embeddings, dim=0)
    torch.save(embeddings.cpu(), f"{OUTPUT_PATH}/embeddings_{l}.pt")

    print(f"[{l}] Saved: {embeddings.shape}, Example norm: ‖{torch.norm(embeddings[0]):.4f}‖")


# === Multiprocessing Controller ===
def main():
    """
    Launches multiple processes to compute multilingual sentence embeddings in parallel across GPUs.
    Each process handles one language.
    """
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    processes = []

    # Spawn one process per language, cycling through available GPUs
    for idx, lang in enumerate(lang_list):
        gpu_id = idx % num_gpus
        p = Process(target=embed_language, args=(lang, gpu_id))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()


# === Entry Point ===
if __name__ == "__main__":
    # Ensures safe multiprocessing with CUDA
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    main()
