import numpy as np
from tqdm import tqdm
import openai

from src.helpers import bottom_k_entropy_words, preprocess_text, convert_huggingface_data_to_list_dic, \
    write_to_csv, create_entropy_map_func, load_data_pile, create_line_to_top_words_map
import torch.nn.functional as F
import torch
from datasets import load_dataset
import zlib
import datetime
import random
from book_list import book_list_not_trained


def getPerplexityScore(sentence, model, tokenizer, gpu):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0).to(
        gpu).long()  # Ensure the tensor is of type Long
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, logits, input_ids_processed


def gpt3PerplexityScore(prompt, model_name):
    """
    Calculate perplexity using GPT-3 API
    """
    prompt = prompt.replace('\x00', '')
    responses = None
    openai.api_key = "YOUR_API_KEY"  # YOUR_API_KEY

    while responses is None:
        try:
            responses = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=0,
                temperature=1.0,
                logprobs=5,
                echo=True
            )
        except openai.error.InvalidRequestError:
            print("too long for openai API")

    data = responses["choices"][0]["logprobs"]
    all_prob = [d for d in data["token_logprobs"] if d is not None]
    logits = data["token_logprobs"]
    input_ids_processed = data["tokens"]

    prep = np.exp(-np.mean(all_prob))

    return prep, all_prob, logits, input_ids_processed


def run_all_attacks(model, tokenizer, text, label, name, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE,
                    TOP_K_ENTROPY, nlp_spacy):
    attacks_results = {}
    attacks_results["FILE_PATH"] = name
    attacks_results["label"] = label

    if model is None:
        prep, all_prob, logits, input_ids_processed = gpt3PerplexityScore(text, "davinci")
    else:
        prep, all_prob, logits, input_ids_processed = getPerplexityScore(text, model, tokenizer,
                                                                          gpu=model.device)
    # ppl
    attacks_results["ppl"] = prep

    # Ratio of log ppl of large and zlib
    zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))
    attacks_results["ppl_zlib"] = np.log(prep) / zlib_entropy

    # min-k prob
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[:k_length]
        attacks_results[f"Min_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    # max-k prob
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(all_prob) * ratio)
        topk_prob = np.sort(all_prob)[-k_length:]
        attacks_results[f"Max_{ratio * 100}% Prob"] = -np.mean(topk_prob).item()

    # Min-K++
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
    input_ids = input_ids[0][1:].unsqueeze(-1)
    logits = logits.to(model.device)  # Ensure logits are on the same device
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.1, 0.2, 0.3]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        attacks_results[f"MinK++_{ratio * 100}% Prob"] = np.mean(topk).item()

    tokens = tokenizer.tokenize(text)
    concatenated_tokens = "".join(token for token in tokens)
    mink_plus = mink_plus.cpu()

    # Tag&Tab Attack:


    # Create the map for the Top-K entropy Keywords
    line_to_top_words_map, sentences = create_line_to_top_words_map(
        text, entropy_map, MAX_LEN_LINE_GENERATE, MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY, nlp_spacy
    )

    # Define the values of k you want to iterate over
    k_values = list(range(1, TOP_K_ENTROPY))

    # Create a list of bottom k words once
    all_bottom_k_words = {}

    for line_num, top_words in line_to_top_words_map.items():
        bottom_k_words = bottom_k_entropy_words(" ".join(top_words), entropy_map, max(k_values))
        all_bottom_k_words[line_num] = bottom_k_words

    # Intermediate storage for results
    intermediate_results = {
        "relevant_log_probs": [],
        "relevant_log_probs_one_token": [],
    }

    # This loop finds the indexes of the keywords tokens and calculate them
    for line_num, bottom_k_words in all_bottom_k_words.items():
        for i, word in enumerate(bottom_k_words):
            if word in concatenated_tokens:
                start_index = concatenated_tokens.find(word)
                end_index = start_index + len(word)
                start_token_index = end_token_index = None
                current_length = 0
                for j, token in enumerate(tokens):
                    current_length += len(token)
                    if current_length > start_index and start_token_index is None:
                        start_token_index = j
                    if current_length >= end_index:
                        end_token_index = j
                        break
                if start_token_index is not None and end_token_index is not None:
                    if start_token_index < len(all_prob):
                        intermediate_results["relevant_log_probs_one_token"].append((i, all_prob[start_token_index]))
                    for idx in range(start_token_index, end_token_index + 1):
                        if idx < len(all_prob):
                            intermediate_results["relevant_log_probs"].append((i, all_prob[idx]))

    # Calculate and store results for each k value
    for k in k_values:
        relevant_log_probs = [val for i, val in intermediate_results["relevant_log_probs"] if i < k]
        relevant_log_probs_one_token = [val for i, val in intermediate_results["relevant_log_probs_one_token"] if i < k]

        if relevant_log_probs:
            sentence_log_likelihood = np.mean(relevant_log_probs)
            attacks_results[f"tag_tab_AT_k={k}"] = sentence_log_likelihood

        if relevant_log_probs_one_token:
            sentence_log_probs_one_token = np.mean(relevant_log_probs_one_token)
            attacks_results[f"tag_tab_FT_k={k}"] = sentence_log_probs_one_token

    # Random Sampling of Words
    for k in k_values:
        if k <= len(all_prob):
            random_word_probs = random.sample(all_prob, k)
            attacks_results[f"random_words_mean_prob_k={k}"] = np.mean(random_word_probs)
        else:
            attacks_results[f"random_words_mean_prob_k={k}"] = None  # or handle this case appropriately

    return attacks_results


def run_exp(TOP_K_ENTROPY, MIN_LEN_LINE_GENERATE, MAX_LEN_LINE_GENERATE, tokenizer, model, nlp_spacy, mode,
            only_model_name, train_val_pile):
    entropy_map = None

    if mode == "PILE":
        print("Started ", mode)
        entropy_map = create_entropy_map_func(mode, train_val_pile=train_val_pile)
        db_data = load_data_pile('PILE', train_val_pile, num_samples=100000)
        all_data = []
        for ex in db_data:
            all_data.append(ex)
        all_output = []
        progress_bar = tqdm(total=len(all_data), desc="Processing - " + mode + " Data",
                            mininterval=0.1)  # Initialize the tqdm progress bar
        for i, ex in enumerate(all_data):
            torch.cuda.empty_cache()
            file_name = str(i)
            entropy_map = entropy_map
            text = ex['text']
            text = preprocess_text(text, 2048, tokenizer)
            label = ex['meta']
            new_ex = run_all_attacks(model, tokenizer, text, label, str(i), entropy_map, MAX_LEN_LINE_GENERATE,
                                     MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY, nlp_spacy)
            all_output.append(new_ex)
            if i % 1000 == 0:  # Update progress bar every 1000 iterations
                progress_bar.update(1000)
        progress_bar.close()  # Close the progress bar when done
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        dir_path = f"{mode}/{mode}_Results_To_be_covered/M={only_model_name}_K={str(TOP_K_ENTROPY)}_{train_val_pile}_D={mode}_{current_time}.csv"
        write_to_csv(all_output, dir_path)

    elif mode == "BookMIA":
        print("Started ", mode)
        entropy_map = create_entropy_map_func(mode, train_val_pile=train_val_pile)
        db_data = load_dataset("swj0419/BookMIA", split=f"train")
        data = convert_huggingface_data_to_list_dic(db_data)
        all_output = []
        progress_bar = tqdm(total=len(data), desc="Processing - " + mode + " Data",
                            mininterval=0.1)  # Initialize the tqdm progress bar

        for i, ex in enumerate(data):
            if mode == "BookMIA" and ex["book"].lower() in book_list_not_trained:
                continue

            torch.cuda.empty_cache()
            file_name = str(i) if mode == 'WikiMIA' else str(ex["book_id"])
            entropy_map = entropy_map
            text = ex['input'] if mode == 'WikiMIA' else ex['snippet']
            text = preprocess_text(text, 2048, tokenizer)
            label = ex['label']
            new_ex = run_all_attacks(model, tokenizer, text, label, str(i), entropy_map, MAX_LEN_LINE_GENERATE,
                                     MIN_LEN_LINE_GENERATE, TOP_K_ENTROPY, nlp_spacy)
            all_output.append(new_ex)
            if i % 1000 == 0:  # Update progress bar every 1000 iterations
                progress_bar.update(1000)

        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        dir_path = f"{mode}/{mode}_Results_To_be_covered/M={only_model_name}_K={str(TOP_K_ENTROPY)}_D={mode}_{current_time}.csv"
        write_to_csv(all_output, dir_path)
