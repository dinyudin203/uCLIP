from functools import partial
import torch
from torch import nn
from torch.utils.data import Dataset, random_split, Subset
import torch.distributed as dist
from transformers import Trainer, TrainingArguments

from uclip.model import uCLIP, ModalityType
from uclip.loss import InterLoss, IntraLoss
from uclip.type import ModalityType

from transformers import CLIPProcessor, AutoTokenizer, AutoProcessor
import argparse

import open_clip

# model path
E5_MODEL_PATH = 'intfloat/multilingual-e5-base'
MINILM_MODEL_PATH = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
SIGLIP_MODEL_PATH = "google/siglip2-base-patch32-256"
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"


inter_loss = InterLoss()
intra_loss = IntraLoss()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='mpnet', help='text encoder model')
parser.add_argument('--vlm_model', type=str, default='openclip', help='vlm model')
parser.add_argument('--noise', type=float, default=0.004**0.5, help='standard deviation of noise')
parser.add_argument('--temperature', type=int, default=0.01, help='soft embedding retrieval temperature')
parser.add_argument('--loss', type=str, default='full', help='loss ablation')

parser.add_argument('--dataset', type=str, default="captions_txt/caption_prompts.txt", help='dataset path')
parser.add_argument('--image_store_path', type=str, default="embeddings/openclip_image/image_embedding_openclip_2M.pt", help='image embedding store path')
parser.add_argument('--text_store_path', type=str, default="embeddings/text_embedding_nllb_prompts_mpnet_2M.pt", help='text embedding store path')
parser.add_argument('--extract_method', type=str, default="plain", help='index extraction method')

parser.add_argument('--output_dir', type=str, default="./results", help='output directory')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='lr')
parser.add_argument('--per_device_train_batch_size', type=int, default=512, help='per device train batch size')
parser.add_argument('--per_device_eval_batch_size', type=int, default=64, help='per device eval batch size')
parser.add_argument('--num_train_epochs', type=int, default=5, help='number of training epochs')
parser.add_argument('--eval_strategy', type=str, default="steps", help='evaluation strategy')
parser.add_argument('--eval_steps', type=int, default=100, help='evaluation steps interval')
parser.add_argument('--logging_steps', type=int, default=10, help='logging steps interval')
parser.add_argument('--save_strategy', type=str, default="epoch", help='model save strategy')
parser.add_argument('--save_steps', type=int, default="1500", help='model save steps interval')    
parser.add_argument('--load_best_model_at_end', action='store_true', help='load best model at end of training')
parser.add_argument('--logging_dir', type=str, default="./logs", help='logging directory')
parser.add_argument('--report_to', type=str, default="wandb", help='report to logging service')
parser.add_argument('--run_name', type=str, default="uCLIP", help='run name')
args = parser.parse_args()

if args.model == 'minilm':
    text_processor = AutoTokenizer.from_pretrained(MINILM_MODEL_PATH)
elif args.model == 'e5':
    text_processor = AutoTokenizer.from_pretrained(E5_MODEL_PATH)
elif args.model == 'mpnet':
    text_processor = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)


if args.vlm_model == 'clip':
    clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
elif args.vlm_model == 'siglip2':
    clip_processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
elif args.vlm_model == 'openclip':
    clip_processor = open_clip.get_tokenizer(OPENCLIP_MODEL_BASE)


class EnglishDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]
        print("total count of dataset: ", len(self.lines))

    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        return self.lines[idx]


def collate_fn(batch, text_model):
    flattened = {}
    if args.vlm_model == 'clip':
        clip_inputs = clip_processor(text=batch, max_length=77, truncation=True, return_tensors="pt", padding='max_length')
    elif args.vlm_model == 'siglip2':
        clip_inputs = clip_processor(text=batch, max_length=64, return_tensors="pt", truncation=True, padding=True)
    elif args.vlm_model == 'openclip':
        clip_inputs = clip_processor(batch)
    if text_model == "e5":
        eng_inputs = text_processor(['query: ' + sentence for sentence in batch], return_tensors="pt", padding=True, truncation=True)
    else:
        eng_inputs = text_processor(batch, return_tensors="pt", padding=True, truncation=True)
    if args.vlm_model != 'openclip':
        for k, v in clip_inputs.items():
            flattened[f"clip_{k}"] = v

    else: flattened["clip_inputs"] = clip_inputs
    for k, v in eng_inputs.items():
        flattened[f"eng_{k}"] = v

    return flattened

class uCLIPTrainer(Trainer):
    def __init__(self, text_model, *args, **kwargs):
        kwargs['data_collator'] = partial(collate_fn, text_model=text_model)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, lamb=0.1):
        outputs = model(**inputs)
        z1 = outputs[ModalityType.MULTILINGUAL_TEXT]  # [local_bs, dim]
        z2 = outputs[ModalityType.ENG_TEXT]  # [local_bs, dim]
        z3 = outputs[ModalityType.TEXT]  # [local_bs, dim]
        z4 = outputs[ModalityType.VISION]  # [local_bs, dim]

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()

            # z1 gather
            z1_list = [torch.zeros_like(z1) for _ in range(world_size)]
            dist.all_gather(z1_list, z1.contiguous())
            z1_all = torch.cat(z1_list, dim=0)

            # z2 gather
            z2_list = [torch.zeros_like(z2) for _ in range(world_size)]
            dist.all_gather(z2_list, z2.contiguous())
            z2_all = torch.cat(z2_list, dim=0)

            # z3 gather
            z3_list = [torch.zeros_like(z3) for _ in range(world_size)]
            dist.all_gather(z3_list, z3.contiguous())
            z3_all = torch.cat(z3_list, dim=0)

            # z4 gather
            z4_list = [torch.zeros_like(z4) for _ in range(world_size)]
            dist.all_gather(z4_list, z4.contiguous())
            z4_all = torch.cat(z4_list, dim=0)
        else:
            z1_all = z1
            z2_all = z2
            z3_all = z3
            z4_all = z4

        batch_all = z1_all.size(0)
        inter_loss_text, inter_loss_pseudo = inter_loss(z1, z2, z3, z4)
        intra_loss_ = lamb * intra_loss(z1_all, z2_all, z3_all, z4_all)

        if args.loss == 'full':
            loss = inter_loss_text + inter_loss_pseudo + intra_loss_
        elif args.loss == 'intraX':
            loss = inter_loss_text + inter_loss_pseudo
        elif args.loss == 'textX':
            loss = inter_loss_pseudo + intra_loss_
        elif args.loss == 'pseudoX':
            loss = inter_loss_text + intra_loss_
        outputs['inter_loss'] = inter_loss_text + inter_loss_pseudo
        outputs['intra_loss'] = intra_loss_

        return (loss, outputs) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        self.log({"inter_loss": outputs['inter_loss'].item(), "intra_loss": outputs['intra_loss'].item()})
        return (loss, None, None) if prediction_loss_only else (loss, outputs, None)

    

    

# dataset
dataset = EnglishDataset(args.dataset)
val_size = int(0.001 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, eval_dataset = random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    logging_steps= args.logging_steps,
    save_strategy= args.save_strategy,
    save_steps=args.save_steps,
    load_best_model_at_end=args.load_best_model_at_end,
    logging_dir=args.logging_dir,
    report_to= args.report_to,
    run_name= args.run_name,
    fp16=True,  # 16-bit floating point precision
)


# Trainer initialization
model = uCLIP(args)

trainer = uCLIPTrainer(
    text_model=args.model,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# train
trainer.train()

# evaluation
print(trainer.evaluate())