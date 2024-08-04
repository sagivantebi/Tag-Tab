import re
import string
import csv
import math
from pathlib import Path
from collections import Counter
from datasets import load_dataset


def calculate_entropy(word_freq, total_words):
    probability = word_freq / total_words
    return -probability * math.log2(probability)


def create_entropy_map(data_sources, mode='books'):
    word_counts = Counter()
    if mode == 'PILE':
        for dataset in data_sources:
            for item in dataset[0]:
                words = item['text'].split()
                word_counts.update(words)
    elif mode == 'BookMIA':
        print(f"Started processing for mode: {mode}")
        dataset_name = "swj0419/BookMIA"
        db_data = load_dataset(dataset_name, split="train")
        for item in db_data:
            text_field = 'snippet'
            words = item[text_field].split()
            word_counts.update(words)

    total_words = sum(word_counts.values())
    entropy_map = {word: calculate_entropy(freq, total_words) for word, freq in word_counts.items()}

    return entropy_map


def save_entropy_map(entropy_map, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for word, entropy in entropy_map.items():
            file.write(f"{word} {entropy}\n")


def load_entropy_map(filename):
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def sort_entropy_map(entropy_map, descending=True):
    return sorted(entropy_map.items(), key=lambda item: item[1], reverse=descending)


def get_text_files(folder_path):
    return [str(filepath) for filepath in Path(folder_path).rglob('*.txt')]


def load_data_pile(mode, train_val_pile, num_samples=100000):
    if mode == 'PILE':
        data_sources = load_dataset("monology/pile-uncopyrighted", split=train_val_pile, streaming=True)
        if num_samples:
            data_sources = data_sources.take(num_samples)
        return data_sources

def create_entropy_map_func(mode="BookMIA", train_val_pile="validation"):
    if mode == 'BookMIA':
        data_sources = [load_dataset("swj0419/BookMIA", split=f"train")]
    elif mode == 'PILE':
        data_sources = load_data_pile('PILE', train_val_pile, num_samples=100000)
    print("Creating entropy map...")
    entropy_map = create_entropy_map(data_sources, mode=mode)

    filename = "entropy_map.txt"
    print(f"Saving entropy map to {filename}...")
    save_entropy_map(entropy_map, filename)

    print("Loading entropy map...")
    loaded_entropy_map = load_entropy_map(filename)

    print("Sorted Entropy Map:")
    sorted_entropy_map = sort_entropy_map(loaded_entropy_map)
    for word, entropy in sorted_entropy_map[:20]:
        print(f"{word}: {entropy}")

    print("Entropy map loaded and sorted successfully.")
    print("Len of map = ", len(sorted_entropy_map))

    return loaded_entropy_map


def strip_punctuation(word):
    return word.strip(string.punctuation)

def bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY):
    words_in_line = line.split()
    top_k = int(TOP_K_ENTROPY)

    return sorted(words_in_line, key=lambda word: entropy_map.get(word, float('inf')))[:top_k]


def create_line_to_top_words_map(text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY,
                                 nlp_spacy):
    # text = text.replace('\n', '')
    doc = nlp_spacy(text)
    # Debugging: convert iterator to list to check content
    all_sentences = list(doc.sents)
    sentences = [sent.text.strip() for sent in all_sentences if
                 MIN_LEN_LINE_GENERATE <= len(sent.text.split()) <= MAX_LEN_LINE_GENERATE]

    line_to_top_words_map = {}

    for line_num, line in enumerate(sentences, 1):
        if line.strip():
            top_k_words = {re.sub(r'^\W+|\W+$', '', word.strip(string.punctuation)) for word in
                           bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY) if ' ' not in word}

            ners = {strip_punctuation(ent.text) for ent in doc.ents if ' ' not in ent.text and ent.sent.text == line}
            unique_words = top_k_words.union(ners)
            line_to_top_words_map[line_num] = list(unique_words)

    return line_to_top_words_map, sentences

def preprocess_text(text, max_length, tokenizer):
    # Tokenize the text and truncate it to the maximum length
    input_ids = tokenizer.encode(text)
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
    return tokenizer.decode(input_ids)

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def write_to_csv(data, filename):
    # Check if data is not empty
    if data:
        # Determine the headers from the keys of the first dictionary
        headers = data[0].keys()

        # Create or overwrite the CSV file
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            for row in data:
                writer.writerow(row)