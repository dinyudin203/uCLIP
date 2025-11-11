import sys
sys.path.append('..')

from transformers import AutoTokenizer, AutoProcessor, AutoModel, AutoImageProcessor, CLIPProcessor
from torchvision import transforms
from torchvision.datasets import CIFAR10, STL10
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from sklearn.metrics import f1_score, accuracy_score
from typing import List, Dict, Tuple
from tqdm import tqdm
from types import SimpleNamespace
from uclip.model import uCLIP, ModalityType

import torch
import torch.nn.functional as F
import open_clip
import numpy as np
import pandas as pd
import os
import csv
import json


LANGUAGES = {
    'czech': 'ces_Latn',
    'finnish': 'fin_Latn', 
    'croatian': 'hrv_Latn',
    'hungarian': 'hun_Latn',
    'romanian': 'ron_Latn'
}

DATASETS = ["cifar10", "stl10"]

STL10_ROOT = "path_to_stl10_dataset"
CIFAR10_ROOT = "path_to_cifar10_dataset"

UCLIP_PATH = "../checkpoint/clip_encoder/model_temp.safetensors"
MPNET_MODEL_PATH = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OPENCLIP_MODEL_BASE = "ViT-B-32"
OPENCLIP_MODEL_PATH = "datacomp_xl_s13b_b90k"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataset_and_labels(dataset_name, args=None):
    """ Get dataset and class names based on the dataset name. """
    transform_224 = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    transform = transform_224
    
    if dataset_name == "cifar10":
        dataset = CIFAR10(root=CIFAR10_ROOT, train=False, download=True, transform=transform)
        class_names = dataset.classes
    
    elif dataset_name == "stl10":
        dataset = STL10(root=STL10_ROOT, split='test', download=True, transform=transform)
        class_names = dataset.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded {len(dataset)} images from {dataset_name} dataset with {len(class_names)} classes.")
    
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    return dataset, loader, class_names


def evaluate_model(dataset_name, translated_labels):
    """ Evaluate the model on the specified dataset and return predictions and labels. """
    _, loader, _ = get_dataset_and_labels(dataset_name)
    to_pil = transforms.ToPILImage()

    all_preds = []
    all_labels = []

    # mpnet, openclip
    text_processor = AutoTokenizer.from_pretrained(MPNET_MODEL_PATH)
    args = {
            'model': 'mpnet',
            'vlm_model': 'openclip',
            'training': False,
        }
    args = SimpleNamespace(**args) 

    uclip = uCLIP(args).eval()
    uclip_checkpoint = load_file(UCLIP_PATH)
    _, _, clip_processor = open_clip.create_model_and_transforms(OPENCLIP_MODEL_BASE, pretrained=OPENCLIP_MODEL_PATH)
    uclip.load_state_dict(uclip_checkpoint, strict=False)
    uclip.to(device)
    uclip.eval()

    text_inputs = text_processor(text=translated_labels, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        mul_input = {
                'input_ids': text_inputs['input_ids'],
                'attention_mask': text_inputs['attention_mask']
            }
        mul_features = uclip.trunk.get_multilingual_text_feature(mul_input)
        projected = uclip.project_features({ModalityType.MULTILINGUAL_TEXT: mul_features})
        text_embeds = projected[ModalityType.MULTILINGUAL_TEXT]
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)

        for images, labels in tqdm(loader):
            images_pil = [to_pil(img.cpu()) for img in images]
            vision_inputs = torch.stack([clip_processor(img) for img in images_pil]).to(device)
            vision_features = uclip.trunk.get_vision_feature(vision_inputs)
            projected = uclip.project_features({ModalityType.VISION: vision_features})
            image_embeds = projected[ModalityType.VISION]
            image_embeds = F.normalize(image_embeds, p=2, dim=-1)
                
            logits = image_embeds @ text_embeds.T
            preds = logits.argmax(dim=1).cpu()
                
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Calculate metrics and return results
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return macro_f1


def main():
    results_data = []
    datasets = DATASETS
    
    print(f"=== Starting evaluation for uCLIP on all datasets ===")
    print(f"Total datasets: {len(datasets)}")
    print(f"Languages per dataset: {len(LANGUAGES)}")
    print(f"Total experiments: {len(datasets) * len(LANGUAGES)}")
    print("=" * 80)
    
    for dataset_idx, dataset_name in enumerate(datasets, 1):
        print(f"\n[DATASET {dataset_idx}/{len(datasets)}] Processing {dataset_name}")
        print("-" * 60)
        try:
            # Get dataset and class names
            dataset, _, class_names = get_dataset_and_labels(dataset_name)
            
            # Load existing translations or generate new ones
            translation_filename = f"../test_data/classification/translations_{dataset_name}.json"
            with open(translation_filename, 'r', encoding='utf-8') as f:
                translations = json.load(f)
            
            # Evaluate the model for each language
            print(f"Evaluating {dataset_name} with {len(class_names)} classes...")
            for lang_idx, (language, lang_code) in enumerate(LANGUAGES.items(), 1):
                experiment_num = (dataset_idx - 1) * len(LANGUAGES) + lang_idx
                total_experiments = len(datasets) * len(LANGUAGES)

                print(f"  [{experiment_num}/{total_experiments}] {language} ({lang_code})...")
                try:
                    # Get translated labels for the current language
                    translated_labels = translations[language]

                    # Evaluate the model
                    macro_f1 = evaluate_model(dataset_name, translated_labels)

                    # Store results
                    results_data.append({
                        'model': 'uclip',
                        'dataset': dataset_name,
                        'language': language,
                        'language_code': lang_code,
                        'class_count': len(class_names),
                        'macro_f1': float(macro_f1),
                        'status': 'success',
                        'error_message': None
                    })
                    
                    print(f"Macro F1 = {macro_f1:.4f}")
                    
                except Exception as e:
                    print(f"Error - {str(e)}")
                    results_data.append({
                        'model': 'uclip',
                        'dataset': dataset_name,
                        'language': language,
                        'language_code': lang_code,
                        'class_count': len(class_names),
                        'macro_f1': None,
                        'status': 'error',
                        'error_message': str(e)
                    })
            
        except Exception as e:
            print(f"\nFailed to process {dataset_name}: {str(e)}")
            # If dataset loading fails, log the error for all languages
            for language, lang_code in LANGUAGES.items():
                results_data.append({
                    'model': 'uclip',
                    'dataset': dataset_name,
                    'language': language,
                    'language_code': lang_code,
                    'class_count': None,
                    'macro_f1': None,
                    'status': 'dataset_error',
                    'error_message': str(e)
                })

    # Save detailed results to CSV
    df = pd.DataFrame(results_data)
    output_dir = f"./classification_results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/results_uclip.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nDetailed results saved to: {output_file}")
    successful_results = df[df['status'] == 'success']
    
    if len(successful_results) > 0:
        # Calculate overall statistics
        overall_stats = {
            'total_experiments': len(df),
            'successful_experiments': len(successful_results),
            'success_rate': len(successful_results) / len(df),
            'overall_average_macro_f1': successful_results['macro_f1'].mean(),
            'overall_std_macro_f1': successful_results['macro_f1'].std(),
        }
        
        # Stats by dataset
        dataset_stats = successful_results.groupby('dataset').agg({
            'macro_f1': ['mean', 'std', 'count'],
        }).round(4)
        
        # Stats by language
        language_stats = successful_results.groupby('language').agg({
            'macro_f1': ['mean', 'std', 'count'],
        }).round(4)
        
        # Save summary statistics to CSV
        summary_file = f"./results_classification/summary_uclip.csv"

        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['=== OVERALL STATISTICS ==='])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Experiments', overall_stats['total_experiments']])
            writer.writerow(['Overall Average Macro F1', f"{overall_stats['overall_average_macro_f1']:.4f}"])
            writer.writerow([])
            
            writer.writerow(['=== DATASET STATISTICS ==='])
            writer.writerow(['Dataset', 'Avg Macro F1'])
            for dataset in dataset_stats.index:
                writer.writerow([
                    dataset,
                    f"{dataset_stats.loc[dataset, ('macro_f1', 'mean')]:.4f}",
                ])
            writer.writerow([])
            
            writer.writerow(['=== LANGUAGE STATISTICS ==='])
            writer.writerow(['Language', 'Avg Macro F1'])
            for language in language_stats.index:
                writer.writerow([
                    language,
                    f"{language_stats.loc[language, ('macro_f1', 'mean')]:.4f}"
                ])
    
    # Print final results
    print("\n" + "=" * 80)
    print("=== FINAL RESULTS ===")
    print(f"Model: uclip")
    print(f"Detailed results saved to: {output_file}")
    if len(successful_results) > 0:
        print(f"Summary statistics saved to: {summary_file}")
        
        print(f"\nOverall Summary:")
        print(f"  Total experiments: {overall_stats['total_experiments']}")
        print(f"  Successful experiments: {overall_stats['successful_experiments']}")
        print(f"  Success rate: {overall_stats['success_rate']:.2%}")
        print(f"  Overall average Macro F1: {overall_stats['overall_average_macro_f1']:.4f} ± {overall_stats['overall_std_macro_f1']:.4f}")
        
        print(f"\nResults by dataset:")
        for dataset in dataset_stats.index:
            success_count = int(dataset_stats.loc[dataset, ('macro_f1', 'count')])
            avg_macro = dataset_stats.loc[dataset, ('macro_f1', 'mean')]
            std_macro = dataset_stats.loc[dataset, ('macro_f1', 'std')]
            print(f"  {dataset:15s}: Macro F1 = {avg_macro:.4f} ± {std_macro:.4f} ({success_count}/{len(LANGUAGES)} languages)")
    
    print("=" * 80)

if __name__ == "__main__":
    main()