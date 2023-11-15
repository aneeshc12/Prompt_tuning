import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import kaggle

from transformers import AutoTokenizer, AutoModelForCausalLM,  DataCollatorForSeq2Seq
from datasets import load_dataset
from torch import nn
from copy import copy

# Wrapper class that manages the soft prompt and interfaces with the GPT2 model
class PrefixTunedGPT2(nn.Module):       # not named super accurately, any model should work 
    def __init__(self, num_prompts, model, tokenizer, init_prompts=None, pretrained_embs=None):
        super(PrefixTunedGPT2, self).__init__()
        if(init_prompts != None and pretrained_embs != None):
            raise("Either load pretrained weights or use init_prompts")
            exit(1)
        if(pretrained_embs != None):
            assert(num_prompts == pretrained_embs.shape[0])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.soft_prompt_length = num_prompts
        self.soft_prompt = nn.Embedding(num_prompts, 768)
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        # init
        if init_prompts != None and len(init_prompts.split(' ')) == num_prompts:
            idx = torch.tensor([[i[0]] for i in self.tokenizer(init_prompts.split(' '))['input_ids']]).int().flatten().to(self.device)
            print("idx: ", idx)
            self.soft_prompt.weights = self.model.transformer.wte(idx)

        # freeze the internal model
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels):
        # move everything to cuda
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)
        
        # get all embs
        batch_size = input_ids.shape[0]     # assuming batch first

        # batch x N x emb_size
        soft_prompt = self.soft_prompt(torch.arange(self.soft_prompt_length).to(self.device))
        soft_prompt = torch.tile(soft_prompt.unsqueeze(0), (batch_size, 1, 1))

        input_embs = self.model.transformer.wte(input_ids)
        input_embs += self.model.transformer.wpe(torch.arange(input_ids.shape[-1]).to(self.device))

        # k = copy(input_embs)

        # apppend the soft prompt
        input_embs = torch.cat((soft_prompt, input_embs), -2)
        attention_mask = torch.cat((torch.ones((batch_size, self.soft_prompt_length)).int().to(self.device), attention_mask), -1)
        labels = torch.cat(((
            torch.ones((batch_size, self.soft_prompt_length)).long() * -100 
                            ).to(self.device), input_ids), -1)
        
        # print("Shapes: ", input_embs.shape, attention_mask.shape, labels.shape, " | ", input_ids.shape)
        # print("nan check (should be false): ", input_embs.isnan().any(), attention_mask.isnan().any(), labels.isnan().any())
        # print("attention mask:", attention_mask, attention_mask.dtype)

        outputs = self.model(input_ids=None, attention_mask=attention_mask, labels=labels, inputs_embeds=input_embs)
        # return outputs, soft_prompt, input_embs, k
        return outputs

    def check_in(self):
        k = (self.soft_prompt(torch.arange(self.soft_prompt_length).to(self.device))[:,:6])
        return k