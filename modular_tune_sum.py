import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
# from torch.nn import DataParallel
from torch.utils.data import DataLoader

from transformers import GPT2Tokenizer, GPT2Model
from transformers import DataCollatorForSeq2Seq, AdamW, get_scheduler

from utils.load_data import get_cnndm_data
from utils.load_models import load_gpt2_and_tok
from models.prefix_tuner import PrefixTunedGPT2

# Fine tune a GPT2 small model on the SQuAD dataset

# parameters
batch_size = 2
soft_prompt_length = 4
init_soft_prompt = "summarize shorten distill short"

lr = 5e-5
num_epochs = 10
save_interval = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model and tokenizer, prep prefix tuner
model, tokenizer = load_gpt2_and_tok()
ptgpt2 = PrefixTunedGPT2(soft_prompt_length, model, tokenizer, init_soft_prompt).to(device)

# load dataset, tokenize, trim train and val for easier computation
tokenized_traindata, tokenized_valdata = get_cnndm_data(tokenizer)

data_collator =  DataCollatorForSeq2Seq(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_traindata, shuffle=True, batch_size=batch_size, collate_fn=data_collator
)
val_dataloader = DataLoader(
    tokenized_valdata, batch_size=batch_size, collate_fn=data_collator
)


# setup training params
model.to(device)

optimizer = AdamW(ptgpt2.parameters(), lr=lr)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)


progress_bar = tqdm(range(num_training_steps))
step_update_count = 500

# begin finetuning
lastprompt = ptgpt2.check_in()
# ptgpt2 = DataParallel(ptgpt2)

print("REACHED TRAINING")
for epoch in range(num_epochs):
    avg_loss = torch.tensor([0], requires_grad=False).float().cuda()
    avg_val_loss = torch.tensor([0], requires_grad=False).float().cuda()
    last_val_loss = torch.tensor([1e30], requires_grad=False).float().cuda()

    for i, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = ptgpt2(**batch)
        loss = outputs.loss
        loss.backward()

        if(i % step_update_count == 0):
            print(f"Epoch: %d | Step: %d | Loss: %f" % (epoch, i, loss))
            newprompt = ptgpt2.check_in()
            # print("Diff in soft embs: ", torch.sum(newprompt-lastprompt).item())
            # lastprompt = newprompt

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(ptgpt2.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        avg_loss += loss.item()

    print(f"Epoch: %d | Average train loss: %f" % (epoch+1, avg_loss/len(train_dataloader)))

    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = ptgpt2(**batch)
            loss = outputs.loss

            torch.nn.utils.clip_grad_norm_(ptgpt2.parameters(), max_norm=1.0)
            avg_val_loss += loss.item()

    print(f"Epoch: %d | Average val loss: %f" % (epoch+1, avg_val_loss/len(val_dataloader)))

    # early stopping
    if avg_val_loss/len(val_dataloader) > last_val_loss:
        print(f"Early stopping, current val loss: %f | last val loss: %f" % (avg_val_loss.item()/len(val_dataloader), last_val_loss.item()))
        
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"sum_soft_prompts_length{soft_prompt_length}_{current_time}.pt"

        soft_prompt_state_dict = {"soft_prompt.weight": ptgpt2.state_dict()["soft_prompt.weight"]}
        torch.save(soft_prompt_state_dict, save_path)

        break
    else:
        last_val_loss = avg_val_loss/len(val_dataloader)

    # save the model at each epoch
    if (epoch % save_interval == 0 and epoch != 0) or epoch == 1:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_path = f"./weights/sum_soft_prompts_epochs{epoch}_length{soft_prompt_length}_{current_time}.pt"

        soft_prompt_state_dict = {ptgpt2.state_dict()["soft_prompt.weight"]}

        torch.save(soft_prompt_state_dict, save_path)

print("Exiting")