# 🧩Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack

Welcome to the official implementation of **Tag&Tab**, a novel method for detecting pretraining data in Large Language Models (LLMs) through keyword-based membership inference attacks. This repository provides the code, dataset references, and evaluation scripts to replicate the experiments from our paper:  
**"Tag&Tab: Pretraining Data Detection in Large Language Models Using Keyword-Based Membership Inference Attack"**


Tag&Tab determines whether a given text was part of an LLM's pretraining data using a two-stage process:
1. **Tagging**: Selecting high-entropy and named-entity keywords.
2. **Tabbing**: Computing the average log-likelihood of these keywords through the LLM.

The method operates in a black-box setting and is designed for efficiency, requiring no auxiliary models or retraining.

## 📄 Overview

Tag&Tab addresses the task of **pretraining data detection**: given a piece of text and black-box access to an LLM, can we infer whether the model was trained on that text?

Unlike existing MIAs that analyze entire token sequences, Tag&Tab focuses on the most informative words — combining high-entropy tokens with named entities — to detect memorization more effectively. This approach overcomes common weaknesses of MIAs, such as poor generalization and sensitivity to distribution shifts.

## 🔍 Optimal Configuration

We conducted extensive experiments to determine the best configuration of Tag&Tab:

- **Tagging Impact**: Comparing the selection of high-entropy keywords versus random token selection, Tag&Tab achieved an average AUC of 0.80, while random selection only reached 0.64. This demonstrates the effectiveness of entropy-based keyword selection.
  
![K_ent_rand](https://github.com/user-attachments/assets/91bef60b-b182-4a8b-a5f5-6da133c268a6)

- **Choosing K**: Varying the number of selected keywords per input, we found that smaller LLaMa models perform best with 2-3 keywords, while larger models (e.g., Pythia, GPT-3.5 Turbo) prefer 7. For generalization, we use **K=4** as the default, achieving strong performance across models with an average AUC of 0.797.

![optimal_k_long](https://github.com/user-attachments/assets/c4e56bc3-eaa6-4f83-b882-9b7f3f4fb038)

## 🚀 Running Tag&Tab

### Code Structure (`src/` Folder)
- `attacks.py`: Core implementation of Tag&Tab and baseline MIAs.
- `book_list.py`: Lists BookMIA documents excluded from model training.
- `evaluation_BookMIA.py`: Evaluation script for BookMIA dataset.
- `evaluation_THE_PILE.py`: Evaluation script for The Pile dataset.
- `helpers.py`: Common utility functions.
- `run_neighbor_attack.py`: Runs the Neighbor attack baseline.
- `testbed.py`: Main driver script for running experiments.

### Running Tag&Tab
To run Tag&Tab on a target model and dataset:

```bash
python src/testbed.py --model_name <target_model_name> --mode <dataset_name>

```

### Parameters Explained:
- _min_len_line_generate:_ Minimum length of a line to generate (default: 7).
- _max_len_line_generate:_ Maximum length of a line to generate (default: 40).
- _mode:_ Mode of operation, such as 'PILE' or 'BookMIA' (default: 'PILE').
- _quantize:_ Whether to quantize the model (options: 'T' for True, 'F' for False; default: 'F').
- _model_name:_ Model name or path for loading (default: "huggyllama/llama-7b").
- _train_val_pile:_ The Pile section to use, either 'train' or 'validation' (default: 'validation').

### Huggingface Models:

- `huggyllama/LLaMa-7b`
- `huggyllama/LLaMa-13b`
- `huggyllama/LLaMa-30b`
- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-1.4b`
- `EleutherAI/pythia-2.8b`
- `EleutherAI/pythia-6.9b`
- `EleutherAI/pythia-12b`



## 📊 Baselines

Our script supports the following baselines:
- **PPL**: LOSS Attack - uses the model's loss values (in LLMs the perplexity) to infer membership based on higher perplexity indicating unfamiliar text.
- **Zlib**: calculates the ratio between the log of the text's perplexity and its Zlib compression length.
- **Neighbor**: generates neighbor sentences using a different language model and compares the perplexity ratios, though it is computationally expensive. We followed their paper for this attack and used 'Bert' as the reference model.
- **Min-K%**: averaging the lowest k% probabilities.
- **Max-K%**: averaging the highest k% probabilities.
- **Min-K++%**: averaging the lowest k% probabilities while normalizing token log probabilities using mean and variance.
- **RECALL**: measures the relative change in log-likelihood when conditioning the target text on non-member prefixes.
- **DC-PDD**: calibrates token probabilities using divergence from a reference corpus.

## 📘 Datasets

### BookMIA Datasets

The BookMIA datasets serve as a benchmark designed to evaluate membership inference attack (MIA) methods, specifically in detecting pretraining data from OpenAI models released before 2023 (such as text-davinci-003). Access our BookMIA datasets directly on Hugging Face.

**Loading the Datasets:**

```python
from datasets import load_dataset
dataset = load_dataset("swj0419/BookMIA")
```

- **Label 0**: Refers to unseen data during pretraining.
- **Label 1**: Refers to seen data.


### The Pile Dataset

The Pile dataset is a large, diverse, open-source language modeling dataset developed by EleutherAI. It is used to train large language models and evaluate MIA methods.

**Loading the Datasets:**

```python
from datasets import load_dataset
dataset = load_dataset("monology/pile-uncopyrighted")
```

- **Validation**: Refers to unseen data during pretraining.
- **Train**: Refers to seen data.


## ⚙️ Dependencies

Our implementation relies on widely used, stable libraries (standard versions as of May 2024):

- **Python** 3.10+
- **PyTorch** 2.1 (with CUDA 12.1 for GPU experiments)
- **Transformers (Hugging Face)** 4.40
- **Datasets (Hugging Face)** 2.19
- **spaCy** 3.7 (`en_core_web_sm` or `en_core_web_trf` for NER)
- **NLTK** 3.8 (with `punkt`, `stopwords`, and `wordnet` corpora downloaded)
- **wordfreq** 3.0
- **jieba** 0.42 (for Chinese tokenization in PatentMIA)
- **tiktoken** 0.6 (for GPT models)
  

## 🔐 API Key for OpenAI Models

When using OpenAI models, ensure to add your API key at the appropriate line in `attacks.py` and `run_neighbor_attack.py`:

```python
openai.api_key = "YOUR_API_KEY"
```

