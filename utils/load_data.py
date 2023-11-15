import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import json
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM,  DataCollatorForSeq2Seq
from datasets import load_dataset

## QNA functions

# returns tokenized data loaders for the question answering task
def get_squad_data(tokenizer):
    dataset = load_dataset("squad")
    traindata = dataset["train"].shuffle().select(range(10000))
    valdata = dataset["validation"].shuffle().select(range(1000))

    tokenized_traindata = traindata.map(tokenize_qna, batched=True, remove_columns=traindata.column_names, fn_kwargs={"tokenizer": tokenizer})
    tokenized_valdata = valdata.map(tokenize_qna, batched=True, remove_columns=valdata.column_names, fn_kwargs={"tokenizer": tokenizer})

    tokenized_traindata.set_format("torch")
    tokenized_valdata.set_format("torch")

    return tokenized_traindata, tokenized_valdata

# used to map context, question, answer etc.
def tokenize_qna(item, tokenizer):
       lengths = [len(tokenizer(context  + "<|bos|>" + q + "<|sep|>",
                  truncation=True)['input_ids']) \
              for (context, q) in zip(item["context"], item["question"])]


       default_prompt = [tokenizer(context + "<|bos|>" + q + "<|sep|>" + a,
                            truncation=True) \
                     for (context, q, a) in zip(item["context"], item["question"], [k["text"][0] for k in item["answers"]])]
       # default_label = [tokenizer("<|pad|> " * (len(context + "<|bos|>" + q) - 1)

       default_label = [tokenizer(context + "<|bos|>" + q + "<|sep|>" + a + "<|eos|>",
                            truncation=True) \
                     for (context, q, a) in zip(item["context"], item["question"], [k["text"][0] for k in item["answers"]])]

       for i, l in enumerate(lengths):
              # default_label[i]["input_ids"][:l] = [tokenizer.pad_token_id for i in range(l)]
              default_label[i]["input_ids"][:l] = [-100 for i in range(l)]

       ret = {"input_ids": [i["input_ids"] for i in default_prompt],
              "attention_mask": [a["attention_mask"] for a in default_prompt],
              "labels": [i["input_ids"][1:] for i in default_label]}         # shift labels by one

       return ret
