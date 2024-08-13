import argparse
import datetime
import os
import torch
import torch.nn as nn
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import attacks
from huggingface_hub import login


class ExperimentOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()
        login(token="Enter Your Hugginface Token")

    def initialize_parser(self):
        self.parser.add_argument('--top_k_entropy', type=int, default=10, help='Top K entropy value')
        self.parser.add_argument('--min_len_line_generate', type=int, default=7,
                                 help='Minimum length of line to generate')
        self.parser.add_argument('--max_len_line_generate', type=int, default=40,
                                 help='Maximum length of line to generate')
        self.parser.add_argument('--mode', type=str, default='PILE', help='Mode of operation')
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
        self.nlp_spacy = spacy.load("en_core_web_sm")

    def run(self):
        model_name = self.options.model_name
        print("Start loading " + model_name + " :*")
        quantize = self.options.quantize == 'T'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = None
        if "davinci" == model_name:
            tokenizer = None
        else:
            if quantize:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.options.model_name,
                    device_map="auto",
                    trust_remote_code=True,
                    quantization_config=bnb_config
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_quantized = is_model_quantized(model)
            print(f"Is the model quantized? {model_quantized}")
        self.run_exp_mode(tokenizer, model)

    def run_exp_mode(self, tokenizer, model):
        current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        new_dir_path = f"{self.options.mode}/{self.options.mode}_Results_To_be_covered/to_be_processed_{current_time}"
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
            print(f"Directory {new_dir_path} created successfully.")
        only_model_name = self.options.model_name.split('/')[-1]
        csv_name = f"{new_dir_path}/M={only_model_name}_K={self.options.top_k_entropy}_T={self.options.threshold}_Q={self.options.quantize}_MIN_LEN={self.options.min_len_line_generate}_MXN_LEN={self.options.max_len_line_generate}_SAMP={self.options.num_samples_db}_{current_time}.csv"
        print("$$$ -- " + self.options.mode + " -- $$$")
        print("CSV_NAME:", csv_name)

        attacks.run_exp(int(self.options.top_k_entropy), int(self.options.min_len_line_generate),
                        int(self.options.max_len_line_generate), tokenizer, model, self.nlp_spacy, self.options.mode,
                        only_model_name, self.options.train_val_pile)
        print("Saved the file:" + csv_name)


if __name__ == "__main__":
    options = ExperimentOptions().parse_args()
    runner = ExperimentRunner(options)
    runner.run()
