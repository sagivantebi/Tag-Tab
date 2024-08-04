# Tag&Tab: Efficient and Effective Detection of Pretraining Data from Large Language Models

Welcome to the Tag&Tab repository, the official implementation of our novel method for detecting pretraining data from Large Language Models (LLMs). This repository contains the source code, datasets, and evaluation scripts necessary to replicate our experiments and results as presented in our research paper, "The Chosen Words: Keyword-Based Method for Detecting Pretraining Data from Large Language Models."

## Introduction

The rapid advancement of generative artificial intelligence has led to the widespread adoption of LLMs for a variety of applications, from conversational agents to content generation. However, the data collection methods used to train these models often raise significant privacy and ethical concerns, particularly regarding the inclusion of personally identifiable information (PII) and copyrighted content.

Tag&Tab is designed to address these concerns by providing a robust and efficient method for detecting whether specific text samples were part of an LLM's training dataset. Our approach leverages advanced natural language processing (NLP) techniques to identify high-entropy keywords and predict their log-likelihoods using the target LLM, offering a novel solution to the challenges faced by existing membership inference attack (MIA) methods.

## Features

- **Tag Phase**: Identifies high-entropy words in the text and selects the most informative keywords.
- **Tab Phase**: Calculates the average log-likelihood of the selected keywords using the target LLM.
- **Inference**: Compares the average log-likelihood against a threshold to determine the text's membership status.
- **Reference-Free**: Achieves robust performance without the need for additional reference models, making it resource and time-efficient.

## Key Contributions

1. **Novel Detection Method**: Tag&Tab focuses on the contextual and semantic relevance of words, leading to a new direction in MIA research for LLMs.
2. **High Performance**: Demonstrates superior performance across various datasets and LLMs, outperforming state-of-the-art (SOTA) methods with over 10% improvement in AUC scores.
3. **Efficiency**: Provides a reference-free solution that is both resource and time-efficient, eliminating the need for additional models or reference-based attacks.

## Repository Contents

- **src/**: Contains the source code for the Tag&Tab method.
- **data/**: Includes the datasets used for evaluation.
- **experiments/**: Scripts for running experiments and reproducing results.
- **docs/**: Documentation and usage guides.

## Getting Started

To get started with Tag&Tab, please refer to the [Installation Guide](docs/installation.md) and [Usage Instructions](docs/usage.md). Detailed documentation is provided to help you understand and utilize the method effectively.

## Method

\textit{Tag&Tab} follows these three stages:

1. **Preprocessing**:
    - **Word Entropy Map**: Calculate the entropy for each word $w_i$ in dataset $D$ using the formula:
    $$ 
    E(w_i) = p(w_i) \cdot \log_2 p(w_i) 
    $$
    - **Sentence Segmentation**: Split the text into individual sentences using the Natural Language Toolkit (NLTK).

2. **Tag**:
    - Select $K$ keywords from each sentence $S$ by targeting Named Entities (NERs) using spaCy and high-entropy words from the entropy map. Keywords are chosen within a specific range $s, e$ to balance context.

3. **Tab**:
    - Calculate the log-likelihood of the selected keywords. For each word $w_i$ in sentence $S$:
    $$
    \log p(w_i \mid w_1, \ldots, w_{i-1}) = \log p(t_{i_1} \mid w_1, \ldots, w_{i-1})
    $$
    - Compute the average log-likelihood for the keywords:
    $$
    \text{Keywords Prob}(S) = \frac{1}{K} \sum_{w_i \in \text{Keywords}(S)} \log p(w_i | w_1, \ldots, w_{i-1})
    $$
    - Compare the average log-likelihood against a threshold to determine text membership.

GitHub supports inserting math expressions in TeX and LaTeX style syntax using the `$` and `$$` delimiters natively in Markdown. This content is then rendered using the highly popular MathJax library. Please see the [documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables) for more information on including mathematical expressions in Markdown on GitHub.


GitHub supports inserting math expressions in TeX and LaTeX style syntax. This content is then rendered using the highly popular MathJax library. Please see the [documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables) for more information on including mathematical expressions in Markdown on GitHub.


## Citation

If you find this repository useful in your research, please consider citing our paper:

