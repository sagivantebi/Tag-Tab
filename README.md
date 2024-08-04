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

We introduce \textit{Tag\&Tab}, a novel approach designed to efficiently and effectively detect pretraining data from LLMs. \textit{Tag\&Tab} leverages advanced NLP techniques to tag keywords and predict them. The method follows simple steps of identifying the high entropy words in the text, then choosing the $K$ words with the highest entropy score, which we refer to as \textit{Keywords}. Then by passing the entire text to the target LLM, we calculate the $K$ keywords' average log-likelihood, and eventually compare it to a threshold and determine the text membership. 

\textit{Tag\&Tab} follows this 3 stages:

1. **Preprocessing**: 
    - Create the Word Entropy Map: Given a dataset \( D \), containing text files \( T \), we create the word entropy map \( E \) by dividing the frequency of each word \( w_i \in D \) by the total number of words in \( D \), resulting in its probability \( p(w_i) \). The entropy for each word is then calculated using the formula: 
    \[
    E(w_i) = p(w_i) \cdot \log_2 p(w_i)
    \]
    - Separate the text into individual sentences. This could be done using the common Python library Natural Language Toolkit (NLTK).

2. **Tag**:
    - Select \( K \) keywords from each sentence \( S \in T \) text file, targeting Named Entities (NERs) and the high entropy words. Our NERs are gathered using the commonly known library spaCy, while the high-entropy words are taken from the words entropy map we built during the previous stage. The selection of words is made from within a specific range \( s, e \) in the sentence \( S_i \) to ensure that keywords have a balanced amount of prefix words. 

3. **Tab**:
    - Calculate the log-likelihood of the previously identified \textbf{keywords} given the probabilities of their preceding words. For each sentence \( S \in T \) consisting of \( n \) words \( w_1, w_2, \ldots, w_n \), where each word \( w_i \) is tokenized into tokens, denoted as \( w_i = t_{i_1}, t_{i_2}, \ldots, t_{i_m} \). Token \( t_{i_j} \), given its preceding tokens, is calculated as \( \log p(t_{i_j} | t_{i_1}, \ldots, t_{i_{j-1}}) \). 
    We define the log-likelihood of a word \( w_i \) using the log-likelihood of its first token \( t_{i_1} \) given its preceding tokens, expressed as:
    \[
    \log p(w_i \mid w_1, \ldots, w_{i-1}) = 
    \log p(t_{i_1} \mid w_1, \ldots, w_{i-1})
    \]
    The algorithm selects the \( K \) keywords from \( S \), and computes the average log-likelihood of the keywords:
    \[
    \text{Keywords Prob}(S) = \frac{1}{K} \sum_{w_i \in \text{Keywords}(S)} \log p(w_i | w_1, \ldots, w_{i-1})
    \]
    We then compute the average of the Keywords Probability results for each sentence $S \in T$, and finally compare it against a threshold to determine whether the text is classified as a member or not.

## Citation

If you find this repository useful in your research, please consider citing our paper:

