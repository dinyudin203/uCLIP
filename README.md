# 📎 uCLIP: Parameter-Efficient Multilingual Extension of Vision-Language Models with Unpaired Data (AAAI 2026)
### [Project](https://dinyudin203.github.io/uCLIP-project) | [Paper](https://arxiv.org/abs/2511.13036) <br>

Note:  
The memory bank files, model checkpoints, and image files in test datasets are not included in this repo due to their large sizes.  
For model training and evaluation, you can reproduce memory bank and model checkpoint using the submitted code.  
We specified which datasets and models were used in the full paper.

## Environment
```bash
conda create -n uCLIP python=3.9
conda activate uCLIP
pip install -r requirments.txt
```
Recommended CUDA version: 11.8

## Constructing Memory Bank
```bash
python embedding/store_image_embedding.py
```
```bash
python embedding/store_text_embedding.py
```

## Train
```bash
accelerate launch train.py
```

## Evaluation
### Retrieval
```bash
cd inference
python multilingual_retrieval.py --vlm_model openclip --dataset xm3600
```
You can change --vlm_model and --dataset arguments.

### Classification
```bash
cd inference
python multilingual_classification.py
```

### Measure Inference Time
```bash
cd inference
python inference_time.py --dataset mscoco
```
You can change the --dataset argument.

### UMAP Visualization
```bash
cd visualization
python umap_img_emb.py --vlm_model openclip
```
You can change the --vlm_model argument.

### Calculate Cosine Similarity
```bash
cd visualization
python cosine_similarity.py
```
