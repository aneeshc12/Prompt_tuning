import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM,  DataCollatorForSeq2Seq
from datasets import load_dataset

def load_gpt2_and_tok():
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # add extra special characters
    tokenizer.add_special_tokens({"bos_token": "<|bos|>",
                              "eos_token": "<|eos|>",
                              "unk_token": "<|unk|>",
                              "sep_token": "<|sep|>",
                              "pad_token": "<|pad|>",
                              "cls_token": "<|cls|>"})
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer