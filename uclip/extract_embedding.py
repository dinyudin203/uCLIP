import torch
import torch.nn.functional as F
import os
import numpy as np




@torch.no_grad()
def retrieve_embedding(model, text_embedding_multi, text_embedding_clip, multi_emb, image_emb, temperature=0.01):
    """
    Step 1: Soft Embedding Retrieval
    - English Text ↔ Image Memory
    - English Text ↔ Multilingual Memory
    """

    text_embedding_multi = F.normalize(text_embedding_multi, dim=-1)
    multi_emb = F.normalize(multi_emb, dim=-1)
    similarity_image = torch.matmul(text_embedding_clip, image_emb.T) / temperature
    similarity_multi = torch.matmul(text_embedding_multi, multi_emb.T) / temperature
 
    weights_image = torch.softmax(similarity_image, dim=-1)
    weights_multi = torch.softmax(similarity_multi, dim=-1)

    enhanced_image = F.normalize(torch.matmul(weights_image, image_emb), dim=-1)
    enhanced_multi = F.normalize(torch.matmul(weights_multi, multi_emb), dim=-1)

    return enhanced_multi, enhanced_image





