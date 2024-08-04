# Tag&Tab: Efficient and Effective Detection of Pretraining Data from Large Language Models

Welcome to the Tag&Tab repository, the official implementation of our novel method for detecting pretraining data from Large Language Models (LLMs). This repository contains the source code, datasets references, and evaluation scripts necessary to replicate our experiments and results as presented in our research paper, "Tag&Tab: Enhancing Data Leakage Detection in Large Language Models through Keyword-Based Membership Inference Attacks"

![Tag Tab Diagram  - Image](https://github.com/user-attachments/assets/1a0266aa-c422-4e3b-92cf-8bbf451d5f0f)

## Overview

We explore the pretraining data detection problem: given a piece of text and black-box access to an LLM without knowing the pretraining data, can we determine if the model was trained on the provided text? Our approach, Tag&Tab, uses advanced NLP techniques to tag high-entropy keywords and predict their log-likelihoods using the target LLM.

⭐ If you find our implementation and paper helpful, please consider citing our work ⭐:

```bibtex
@article{antebi2024tagtab,
  title={The Chosen Words: Keyword-Based Method for Detecting Pretraining Data from Large Language Models},
  author={Sagiv Antebi and [Co-authors]},
  journal={Journal/Conference Name},
  year={2024},
  volume={XX},
  number={YY},
  pages={ZZ-ZZZ},
}
```

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
-top_k_entropy: Top K entropy value to select high-entropy words (default: 5).
-min_len_line_generate: Minimum length of a line to generate (default: 7).
-max_len_line_generate: Maximum length of a line to generate (default: 40).
-mode: Mode of operation, such as 'PILE' or 'BookMIA' (default: 'PILE').
-quantize: Whether to quantize the model (options: 'T' for True, 'F' for False; default: 'F').
-model_name: Model name or path for loading (default: "huggyllama/llama-7b").
-train_val_pile: The Pile section to use, either 'train' or 'validation' (default: 'validation').


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

When using OpenAI models, ensure to add your API key at the appropriate line in `run_neighbor_attack.py`:

```python
openai.api_key = "YOUR_API_KEY"
```

## Contact

For any questions or issues, please open an [issue](https://github.com/username/TagAndTab/issues) or contact us at [email@example.com](mailto:email@example.com).

Thank you for your interest in Tag&Tab! We hope this repository serves as a valuable resource for your research and development efforts.

