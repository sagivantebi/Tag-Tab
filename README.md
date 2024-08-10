# 🧩Tag&Tab: Efficient and Effective Detection of Pretraining Data from Large Language Models

Welcome to the Tag&Tab repository, the official implementation of our novel method for detecting pretraining data from Large Language Models (LLMs). This repository contains the source code, datasets references, and evaluation scripts necessary to replicate our experiments and results as presented in our research paper, "Tag&Tab: Enhancing Data Leakage Detection in Large Language Models through Keyword-Based Membership Inference Attacks"

![Tag Tab Diagram  - Image](https://github.com/user-attachments/assets/1a0266aa-c422-4e3b-92cf-8bbf451d5f0f)

Illustration of the method Tag\&Tab - The process starts by inputting a text, in our example we entered the ending of the famous poem "The Road Not Taken", into the Target LLM to gain its words probabilities distribution (Words Prob). Simultaneously, in the Tag phase, the keywords are selected from the Text Input using the Words Entropy Map and the Spacy library, taking the highest scored words entropy and the NERs. Then in the Tab phase, we calculate the log-likelihood of only the selected keywords. Finally, we compare the average log-likelihood of the chosen keywords against a threshold $\gamma$ to determine if the text was part of the pretraining data of the Target LLM or not.

## Overview

We explore the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? Our approach, Tag&Tab, uses advanced NLP techniques to tag high-entropy keywords and predict their log-likelihoods using the target LLM.

## Optimal Configuration


- We show the impact of our word selection method, Tag, which selects the highest $K$ entropy words, compared to a random selection of words using the same Tab algorithm. For each model, we presented 10 results using the Tag method (Blue) and 10 results using a random selection of words (Orange).
The results indicate that selecting the highest $K$ entropy words improves performance across all models. The Tag method achieved an average AUC of 79%, compared to an average AUC of 64.4% with a random selection of $K$ words. This demonstrates the effectiveness of the Tag method in enhancing model performance by focusing on high-entropy words.
![K_ent_rand](https://github.com/user-attachments/assets/91bef60b-b182-4a8b-a5f5-6da133c268a6)


- The number of chosen keywords that perform best was tested by choosing 1 to 10 keywords from each sentence.
The results show that the optimal number of keywords required to ensure the optimal detection depends also on the model architecture. For different sizes of the LLaMa1 models, the optimal number of keywords ranged from 2 to 3, while for the Pythia models and GPT-3.5 turbo, the optimal number of tagged keywords was 7 keywords.
To generalize our selection we can infer that the best results across all models on average were when the number of highest $K$ entropy words was when $K=4$, resulting in an average AUC score of 79.7\%.

![optimal_k_long](https://github.com/user-attachments/assets/c4e56bc3-eaa6-4f83-b882-9b7f3f4fb038)



## 🚀 Running Tag&Tab

### Files in `src/` Folder

- **attacks.py**: Contains the implementation of various attack methods used for evaluating the model.
- **book_list.py**: Defines the list of book titles not included in the training set.
- **evaluation_BookMIA.py**: Evaluation script for the BookMIA dataset.
- **evalutation_THE_PILE.py**: Evaluation script for the PILE dataset.
- **helpers.py**: Helper functions used throughout the codebase.
- **run_neighbor_attack.py**: Script to run neighbor attack methods.
- **testbed.py**: Testbed for running various experiments and evaluations.

### Running the Model

To run the Tag&Tab method, use the following command:

```sh
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
- `EleutherAI/pythia-6.9b`
- `EleutherAI/pythia-12b`



## 📊 Baselines

Our script supports the following baselines:
- **PPL**: LOSS Attack - uses the model's loss values (in LLMs the perplexity) to infer membership based on higher perplexity indicating unfamiliar text.
- **Zlib**: calculates the ratio between the log of the text's perplexity and its Zlib compression length.
- **Neighbor**: generates neighbor sentences using a different language model and compares the perplexity ratios, though it is computationally expensive. We followed their paper for this attack and used 'Bert' as the reference model.
- **Min-K%**: averaging the lowest k% probabilities.
- **Max-K%**: averaging the highest k% probabilities.
- **Min-K%**: averaging the lowest k% probabilities while normalizing token log probabilities using mean and variance.

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

  

## 🔐 API Key for OpenAI Models

When using OpenAI models, ensure to add your API key at the appropriate line in `attacks.py` and `run_neighbor_attack.py`:

```python
openai.api_key = "YOUR_API_KEY"
```

