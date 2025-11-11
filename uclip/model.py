import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from uclip.projector import uCLIP_Head
from uclip.trunks import Trunk
from uclip.type import ModalityType
from uclip.extract_embedding import retrieve_embedding
from uclip.loss import InterLoss, IntraLoss

inter_loss = InterLoss()
intra_loss = IntraLoss()

    
class uCLIP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        self.model = args.model
        self.vlm_model = args.vlm_model
        if args.training == True:
            self.noise = args.noise
            self.image_store_path = args.image_store_path
            self.image_index_path = args.image_index_path
            self.text_store_path = args.text_store_path
            self.text_index_path = args.text_index_path
            self.temperature = args.temperature
            self.extract_method = args.extract_method


            self.text_emb = torch.load(self.text_store_path).to(device=self.device)
            self.image_emb = torch.load(self.image_store_path).to(device=self.device)
        self.trunk = Trunk(args)
        self.uclip_head = uCLIP_Head(args)

        for param in self.trunk.parameters():
            param.requires_grad = False

    def loss_fn(self, mul_emb, eng_emb, text_emb, vision_emb, lamb = 0.1):
        inter_loss_ = inter_loss(mul_emb, eng_emb, text_emb, vision_emb)
        intra_loss_ = lamb * intra_loss(mul_emb, eng_emb, text_emb, vision_emb)
        loss = inter_loss_ + intra_loss_
        return loss.unsqueeze(0), inter_loss_, intra_loss_

    def project_features(self, features: dict) -> dict:
        uclip_embeddings = {}
        for modality in features.keys():
            if modality == ModalityType.VISION:
                uclip_embeddings[modality] = self.uclip_head.forward_clip(features[modality])
            elif modality == ModalityType.TEXT:
                uclip_embeddings[modality] = self.uclip_head.forward_clip(features[modality])
            elif modality == ModalityType.MULTILINGUAL_TEXT:
                uclip_embeddings[modality] = self.uclip_head.forward_text(features[modality])
            elif modality == ModalityType.ENG_TEXT:
                uclip_embeddings[modality] = self.uclip_head.forward_text(features[modality])
        return uclip_embeddings


    def get_embeddings(self, input: dict) -> dict:
        input[ModalityType.VISION] = {
            'pixel_values': input["clip_vision_pixel_values"],
        }
        input[ModalityType.TEXT] = {
            'input_ids': input["clip_text_input_ids"],
            'attention_mask': input["clip_text_attention_mask"],
        }
        input[ModalityType.ENG_TEXT] = {
            'input_ids': input["eng_input_ids"],
            'attention_mask': input["eng_attention_mask"]
        }
        input[ModalityType.MULTILINGUAL_TEXT] = {
            'input_ids': input["mul_input_ids"],
            'attention_mask': input["mul_attention_mask"]
        }
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.TEXT]   = self.trunk.get_text_feature(input[ModalityType.TEXT])

        features[ModalityType.ENG_TEXT] = self.trunk.get_eng_text_feature(input[ModalityType.ENG_TEXT])
        features[ModalityType.MULTILINGUAL_TEXT] = self.trunk.get_multilingual_text_feature(input[ModalityType.MULTILINGUAL_TEXT])
        features = self.project_features(features)
        return features

    def get_test_embeddings(self, input: dict) -> dict:
        if self.vlm_model == 'openclip':
            input[ModalityType.VISION] = input.get("clip_vision_inputs")
        else:
            input[ModalityType.VISION] = {
                'pixel_values': input["clip_vision_pixel_values"],
            }
        input[ModalityType.MULTILINGUAL_TEXT] = {
            'input_ids': input["mul_input_ids"],
            'attention_mask': input["mul_attention_mask"]
        }
        features = {}
        features[ModalityType.VISION] = self.trunk.get_vision_feature(input[ModalityType.VISION])
        features[ModalityType.MULTILINGUAL_TEXT] = self.trunk.get_multilingual_text_feature(input[ModalityType.MULTILINGUAL_TEXT])
        features = self.project_features(features)
        return features
    
    def forward(self, **input) -> Tensor:
        if self.vlm_model == 'openclip':
            input[ModalityType.TEXT] = input.get("clip_inputs")
        else:
            input[ModalityType.TEXT] = {
                'input_ids': input.get("clip_input_ids"),
                'attention_mask': input.get("clip_attention_mask"),
            }
        input[ModalityType.ENG_TEXT] = {
            'input_ids': input.get("eng_input_ids"),
            'attention_mask': input.get("eng_attention_mask")
        }
        features = {}
        features[ModalityType.ENG_TEXT] = self.trunk.get_eng_text_feature(input[ModalityType.ENG_TEXT])
        features[ModalityType.TEXT] = self.trunk.get_text_feature(input[ModalityType.TEXT])

        features[ModalityType.MULTILINGUAL_TEXT], features[ModalityType.VISION] = retrieve_embedding(
                model=self.model,
                text_embedding_multi=features[ModalityType.ENG_TEXT],
                text_embedding_clip=features[ModalityType.TEXT],
                multi_emb=self.text_emb,
                image_emb=self.image_emb,
                temperature=self.temperature)


        # Add noise to the features for robustness
        for key, tensor in features.items():
            noise = torch.randn_like(tensor) * self.noise
            features[key] = F.normalize(tensor + noise, dim=-1)

        features = self.project_features(features)
        

        return {**features}
