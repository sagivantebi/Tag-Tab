# Tag&Tab: Efficient and Effective Detection of Pretraining Data from Large Language Models

Welcome to the Tag&Tab repository, the official implementation of our novel method for detecting pretraining data from Large Language Models (LLMs). This repository contains the source code, datasets references, and evaluation scripts necessary to replicate our experiments and results as presented in our research paper, "Tag&Tab: Enhancing Data Leakage Detection in Large Language Models through Keyword-Based Membership Inference Attacks"
[Tag Tab Diagram - Final + Threshold.pdf](https://github.com/user-attachments/files/16486686/Tag.Tab.Diagram.-.Final.%2B.Threshold.pdf)

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


## Method

**Tag&Tab** follows these three stages:

1. **Preprocessing**:
    - **Word Entropy Map**: Calculate the entropy for each word in the dataset by dividing the frequency of each word by the total number of words, resulting in its probability. The entropy for each word is then calculated.
    - **Sentence Segmentation**: Split the text into individual sentences using the Natural Language Toolkit (NLTK).

2. **Tag**:
    - Select $K$ keywords from each sentence by targeting Named Entities (NERs) using spaCy and high-entropy words from the entropy map. Keywords are chosen within a specific range to balance context.

3. **Tab**:
    - Calculate the log-likelihood of the selected keywords given the probabilities of their preceding words. For each sentence, compute the average log-likelihood for the keywords.
    - Compare the average log-likelihood against a threshold to determine text membership.




GitHub supports inserting math expressions in TeX and LaTeX style syntax. This content is then rendered using the highly popular MathJax library. Please see the [documentation](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/organizing-information-with-tables) for more information on including mathematical expressions in Markdown on GitHub.


## Citation

If you find this repository useful in your research, please consider citing our paper:

