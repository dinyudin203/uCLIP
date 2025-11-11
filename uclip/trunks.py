from transformers import AutoTokenizer, AutoModel, AutoProcessor
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import open_clip


MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"
SIGLIP_MODEL_PATH = "google/siglip2-base-patch32-256"


class Trunk(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.model = args.model
        self.vlm_model = args.vlm_model
        if self.vlm_model == 'openclip':
            self.openclip_model, self.openclip_tokenizer, self.openclip_preprocess = get_clip_extractor(self.vlm_model) 
        else:
            self.clip_extractor, self.clip_processor = get_clip_extractor(self.vlm_model) # siglip2
        self.text_extractor, self.text_processor = get_text_extractor(self.model)

        if self.vlm_model == 'openclip':
            self.openclip_model.eval()
        else:
            self.clip_extractor.eval() # siglip2
        self.text_extractor.eval()

    # get CLIP vision feature
    def get_vision_feature(self, inputs) -> Tensor:
        if self.vlm_model == 'openclip':
            image_feature = self.openclip_model.encode_image(inputs)
        else:
            image_feature = self.clip_extractor.get_image_features(**inputs)
        return F.normalize(image_feature, dim=-1)
    
    # get CLIP english text feature
    def get_text_feature(self, inputs) -> Tensor:
        if self.vlm_model == 'openclip':
            text_feature = self.openclip_model.encode_text(inputs)
        else:
            text_feature = self.clip_extractor.get_text_features(**inputs)
        return F.normalize(text_feature, dim=-1)

    # get multilingual encoder english text feature
    def get_eng_text_feature(self, inputs) -> Tensor:
        outputs = self.text_extractor(**inputs)
        pooled = mean_pooling(outputs, inputs['attention_mask'])
        pooled = F.normalize(pooled, dim=-1)
        return pooled

    # get multilingual encoder multilingual text feature
    def get_multilingual_text_feature(self, inputs) -> Tensor:
        outputs = self.text_extractor(**inputs)
        pooled = mean_pooling(outputs, inputs['attention_mask'])
        pooled = F.normalize(pooled, dim=-1)
        return pooled


# get extractor
def get_clip_extractor(vlm_model) -> nn.Module:
    if vlm_model == 'siglip2':
        clip_model = AutoModel.from_pretrained(SIGLIP_MODEL_PATH)
        processor  = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
    elif vlm_model == 'openclip':
        model, _, preprocess = open_clip.create_model_and_transforms(OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH)
        tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL_BASE)
        return model, tokenizer, preprocess

    return clip_model, processor


# get text extractor
def get_text_extractor(model):
    if model == 'mpnet':
        return AutoModel.from_pretrained(MPNET_MODEL_PATH), AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]