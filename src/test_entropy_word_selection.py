import re
import string
import math
import spacy
from datasets import load_dataset
from wordfreq import word_frequency



_PUNC_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_token(w: str) -> str:
    """
    Normalizes a word by converting to lowercase, removing punctuation,
    and stripping whitespace.
    """
    w = w.lower().translate(_PUNC_TABLE)
    w = re.sub(r"\s+", "", w)
    return w


def create_entropy_map(data_source):
    """
    Builds an entropy map from a dataset.
    Entropy is calculated as -p * log2(p), where p is the word's frequency.
    """
    print("Building vocabulary from the dataset...")
    vocab = set()
    # Iterate through the dataset to build a unique vocabulary
    for item in data_source:
        text = item.get('snippet', '')  # Use 'snippet' for BookMIA
        for word in text.split():
            normalized_word = _normalize_token(word)
            if normalized_word:
                vocab.add(normalized_word)

    print(f"Vocabulary built with {len(vocab)} unique words.")
    print("Calculating entropy for each word...")
    entropy_map = {}
    for word in vocab:
        p = word_frequency(word, 'en')
        entropy_map[word] = (-p * math.log2(p)) if p > 0.0 else 0.0

    print("Entropy map calculation complete.")
    return entropy_map


def save_entropy_map(entropy_map, filename):
    """Saves the calculated entropy map to a file."""
    with open(filename, 'w', encoding='utf-8') as f:
        for word, entropy in entropy_map.items():
            f.write(f"{word} {entropy}\n")
    print(f"Entropy map successfully saved to {filename}")


def load_entropy_map(filename):
    """Loads an entropy map from a file."""
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, entropy_str = parts
                entropy_map[word] = float(entropy_str)
    print(f"Entropy map successfully loaded from {filename}")
    return entropy_map


def bottom_k_entropy_words(text, entropy_map, k):
    """
    Finds the k words in a text with the lowest entropy.
    Unknown words are given infinite entropy to ensure they are not selected.
    """
    tokens = [_normalize_token(w) for w in text.split()]
    tokens = [t for t in tokens if t]  # Remove any empty strings

    # Sort tokens by their entropy value, lowest first
    # Use .get(word, float('inf')) to handle words not in our map
    sorted_tokens = sorted(tokens, key=lambda w: entropy_map.get(w, float('inf')))

    return sorted_tokens[:k]


def pick_words_with_ner(text, entropy_map, k, nlp_spacy):
    """
    Picks words based on the union of bottom-k low-entropy words and
    named entities found in the text. This is the method from your code.
    """
    # 1. Get the k lowest entropy words
    low_entropy_k = set(bottom_k_entropy_words(text, entropy_map, k))

    # 2. Get Named Entity Recognition (NER) tokens
    doc = nlp_spacy(text)
    ner_tokens = set()
    for ent in doc.ents:
        # Normalize and add each token from the named entity
        for token_in_entity in ent.text.split():
            normalized_token = _normalize_token(token_in_entity)
            if normalized_token:
                ner_tokens.add(normalized_token)

    # 3. Return the union of the two sets
    return list(low_entropy_k.union(ner_tokens))


if __name__ == "__main__":
    # --- PART 1: CONSTRUCT THE ENTROPY MAP ---
    print("### Step 1: Building the Entropy Map on BookMIA ###\n")

    # Load the BookMIA dataset from Hugging Face
    bookmia_dataset = load_dataset("swj0419/BookMIA", split="train")

    # Create and save the entropy map
    entropy_map_filename = "entropy_map_bookmia.txt"
    bookmia_entropy_map = create_entropy_map(bookmia_dataset)
    save_entropy_map(bookmia_entropy_map, entropy_map_filename)

    # Load the map back (as in the original workflow)
    loaded_entropy_map = load_entropy_map(entropy_map_filename)

    print("\n-------------------------------------------------\n")

    # --- PART 2: DEMONSTRATE WORD PICKING ---
    print("### Step 2: Demonstrating Word Picking with k=3 ###\n")

    # Define K for the demonstration
    K = 3

    # Load the SpaCy model for NER
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("SpaCy model 'en_core_web_sm' not found.")
        print("Please run: python -m spacy download en_core_web_sm")
        exit()

    # Select a sample text from the dataset for demonstration
    # This sample from "Dr. Jekyll and Mr. Hyde" contains a clear named entity ("Utterson")
    sample_text = bookmia_dataset[15]['snippet']

    print(f"Using sample text:\n\"\"\"{sample_text}\"\"\"\n")

    # --- Demonstration A: Picking words WITH NERs ---
    print(f"--- A) Picking Words WITH NERs (k={K}) ---")

    # Split text into sentences to process them individually
    doc = nlp(sample_text)
    sentences = [sent.text for sent in doc.sents]

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        picked_words = pick_words_with_ner(sentence, loaded_entropy_map, K, nlp)
        print(f"  Sentence {i + 1}:")
        print(f"    - Original: \"{sentence.strip()}\"")
        print(f"    - Picked Words: {picked_words}\n")

    # --- Demonstration B: Picking words WITHOUT NERs (low entropy only) ---
    print(f"--- B) Picking Words WITHOUT NERs (k={K}) ---")

    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        # This time, we only get the bottom-k entropy words
        picked_words = bottom_k_entropy_words(sentence, loaded_entropy_map, K)
        print(f"  Sentence {i + 1}:")
        print(f"    - Original: \"{sentence.strip()}\"")

        print(f"    - Picked Words: {picked_words}\n")
