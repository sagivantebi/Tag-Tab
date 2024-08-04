import argparse
import datetime
import os
import re

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, BertTokenizer, BertModel
from datasets import load_dataset
from tqdm import tqdm
import random
from book_list import book_list_not_trained

def load_entropy_map(filename):
    entropy_map = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            word, entropy = line.split()
            entropy_map[word] = float(entropy)
    return entropy_map


def bottom_k_entropy_words(line, entropy_map, TOP_K_ENTROPY):
    words_in_line = line.split()
    top_k = int(TOP_K_ENTROPY)
    return sorted(words_in_line, key=lambda word: entropy_map.get(word, float('inf')))[:top_k]


class ExperimentOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--top_k_entropy', type=int, default=5, help='Top K entropy value')
        self.parser.add_argument('--min_len_line_generate', type=int, default=7,
                                 help='Minimum length of line to generate')
        self.parser.add_argument('--max_len_line_generate', type=int, default=40,
                                 help='Maximum length of line to generate')
        self.parser.add_argument('--mode', type=str, default='BookMIA_Neighborhood', help='Mode of operation')
        self.parser.add_argument('--quantize', type=str, default='F', choices=['T', 'F'],
                                 help='Whether to quantize the model')
        self.parser.add_argument('--model_name', type=str, default="huggyllama/llama-7b",
                                 help='Model name or path for loading')
        self.parser.add_argument('--train_val_pile', type=str, default="validation",
                                 help='The Pile - train or validation section')

    def parse_args(self):
        return self.parser.parse_args()


def is_model_quantized(model):
    for layer in model.modules():
        if isinstance(layer, (nn.quantized.Linear, nn.quantized.Conv2d)):
            return True
    return False


class ExperimentRunner:
    def __init__(self, options):
        self.options = options

    def run(self):
        model_name = self.options.model_name
        quantize = self.options.quantize == 'T'
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto",
                                                     trust_remote_code=True) if quantize else AutoModelForCausalLM.from_pretrained(
            model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_quantized = is_model_quantized(model)
        print(f"Is the model quantized? {model_quantized}")

        self.run_exp_mode(tokenizer, model)

    def preprocess_text(self, text, max_length, tokenizer):
        input_ids = tokenizer.encode(text)
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        return tokenizer.decode(input_ids)

    def calculate_word_entropy(self, text, model, tokenizer, device):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)  # Remove batch dimension

        # Calculate entropy for each token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        # Map token IDs to words and pair them with their entropy
        token_ids = inputs['input_ids'].squeeze(0)
        word_entropy = [(tokenizer.decode([token_id]), entropy[i].item()) for i, token_id in enumerate(token_ids)]

        # Sort words by entropy in descending order
        word_entropy_sorted = sorted(word_entropy, key=lambda x: x[1], reverse=True)
        return [word for word, _ in word_entropy_sorted]

    def run_exp_mode(self, tokenizer, model):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        new_dir_path = f"{self.options.mode}/{self.options.mode}_Results_To_be_covered/to_be_processed_{current_time}"
        os.makedirs(new_dir_path, exist_ok=True)
        print(f"Directory {new_dir_path} created successfully.")

        only_model_name = self.options.model_name.split('/')[-1]
        csv_name = f"{new_dir_path}/M={only_model_name}_K={self.options.top_k_entropy}_T={self.options.threshold}_Q={self.options.quantize}_MIN_LEN={self.options.min_len_line_generate}_MXN_LEN={self.options.max_len_line_generate}_SAMP={self.options.num_samples_db}_{current_time}.csv"
        print("$$$ -- " + self.options.mode + " -- $$$")
        print("CSV_NAME:", csv_name)

        target_model = model
        reference_model = BertModel.from_pretrained("bert-base-uncased").to(device)
        reference_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        data = load_dataset("swj0419/BookMIA", split="train")
        entropy_map = load_entropy_map("entropy_map.txt")

        all_output = []
        progress_bar = tqdm(total=len(data), desc="Processing - " + self.options.mode + " Data", mininterval=0.1)


        for i, ex in enumerate(data):
            if ex["book"].lower() in book_list_not_trained:  # Check if the book is in the list for BookMIA mode
                continue  # Skip to the next sample if the book is in the list

            torch.cuda.empty_cache()
            label = ex['label']

            # Mainly because the parser also change it to lower, and then the compraison of the text has to be with the same text
            text = ex['snippet'].lower()
            text = self.preprocess_text(text, 2048, tokenizer)
            book_id = str(ex["book_id"])

            words_to_change = bottom_k_entropy_words(text, entropy_map, self.options.top_k_entropy)
            modified_sentences = self.replace_words_with_unique_variation(text, words_to_change, reference_model,
                                                                          reference_tokenizer)

            original_loss = self.get_sentence_loss(text, target_model, tokenizer, device)
            neighbourhood_loss = self.calculate_neighbourhood_attack(modified_sentences, original_loss, target_model,
                                                                     tokenizer, device)

            all_output.append({
                "book_id": book_id,
                # "original": text,
                # "modified": modified_sentences,
                "label": label,
                "original_loss": original_loss,
                "neighbourhood_loss": neighbourhood_loss
            })

            if i % 1000 == 0:
                progress_bar.update(1000)

        self.write_to_csv(all_output, csv_name)

    def replace_words_with_unique_variation(self, sentence, words_to_replace, reference_model, reference_tokenizer):
        inputs = reference_tokenizer(sentence, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'][0]
        embeddings_matrix = reference_model.get_input_embeddings().weight

        used_replacements = {}
        modified_sentences = []

        # Helper function to strip punctuation from a word
        def strip_punctuation(word):
            return re.sub(r'[^\w\s]', '', word)

        for _ in range(2):  # Assuming generating 3 sentences
            new_tokens = input_ids.clone()
            for word in words_to_replace:
                stripped_word = strip_punctuation(word)
                word_id = reference_tokenizer.convert_tokens_to_ids(stripped_word)
                if word_id == reference_tokenizer.unk_token_id or (input_ids == word_id).nonzero().size(0) == 0:
                    continue  # Skip if word is unknown or not in sentence
                word_embedding = embeddings_matrix[word_id].unsqueeze(0)
                similar_words = self.find_similar_words(stripped_word, word_embedding, embeddings_matrix,
                                                        reference_tokenizer)
                available_choices = [w for w in similar_words if w not in used_replacements.get(word, [])]
                if available_choices:
                    chosen_word = random.choice(available_choices)
                    used_replacements[word] = used_replacements.get(word, []) + [chosen_word]
                else:
                    chosen_word = random.choice(similar_words)  # Fallback if all similar words are used

                # # Print statements for debugging
                # print(f"Used replacements for '{word}': {used_replacements[word]}")
                # print(f"Chosen word: {chosen_word}")

                # Replace the word in the sentence
                for idx in (i for i, token_id in enumerate(input_ids) if
                            reference_tokenizer.decode([token_id]).strip() == word or reference_tokenizer.decode(
                                [token_id]).strip() == stripped_word):
                    new_tokens[idx] = reference_tokenizer.convert_tokens_to_ids(chosen_word)

            modified_sentence = reference_tokenizer.decode(new_tokens, skip_special_tokens=True)
            modified_sentences.append(modified_sentence)

        return modified_sentences

    def find_similar_words(self, word, embedding, embeddings_matrix, tokenizer, top_k=7):
        word_index = tokenizer.convert_tokens_to_ids(word)
        word_embedding = embeddings_matrix[word_index].unsqueeze(0)
        cos_sim = torch.nn.functional.cosine_similarity(word_embedding, embeddings_matrix)
        cos_sim[word_index] = -1  # Ignore the word itself
        top_k_indices = torch.topk(cos_sim, top_k + 1).indices  # Get one extra in case we need to exclude a used word
        similar_words = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_k_indices if idx != word_index]
        return similar_words

    def get_sentence_loss(self, sentence, model, tokenizer, device):
        try:
            input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, labels=input_ids)
                loss = outputs.loss.item()
        except RuntimeError as e:
            # print(f"Error processing sentence: {sentence}. Error: {str(e)}")
            return 0  # Returning 0 if there's a dimension mismatch or any other RuntimeError

        return loss

    def calculate_neighbourhood_attack(self, modified_sentences, original_loss, model, tokenizer, device):
        len_of_sentences = len(modified_sentences)
        total_loss = 0
        for modified_sentence in modified_sentences:
            sentence_loss = self.get_sentence_loss(modified_sentence, model, tokenizer, device)
            total_loss += sentence_loss
        total_loss /= len_of_sentences
        return original_loss - total_loss

    def write_to_csv(self, output_data, file_path):
        import csv
        keys = output_data[0].keys()
        with open(file_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(output_data)


if __name__ == "__main__":
    options = ExperimentOptions().parse_args()
    runner = ExperimentRunner(options)
    runner.run()
