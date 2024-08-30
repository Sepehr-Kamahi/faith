
# %%
from transformers import GPTNeoXForCausalLM, AutoTokenizer,AutoModelForCausalLM
import torch
device = "cuda:1"
model_id = "gptj"

IGNORE_INDEX = -100
if model_id == "pythia2":
    batch_size = 40

elif model_id == "pythia":
    batch_size = 40

elif model_id == "phi3":
    batch_size = 24
elif model_id == "gptj":
    batch_size = 16

max_seq_length = 64

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
if model_id == "pythia":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    model_base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1.4b",torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
elif model_id == "pythia2":
    tokenizer = AutoTokenizer.from_pretrained("pythia-2.8b")
    model_base = AutoModelForCausalLM.from_pretrained("pythia-2.8b",torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

elif model_id == "phi3":
    model_base = AutoModelForCausalLM.from_pretrained(
  model_id,device_map = device, torch_dtype=torch.bfloat16,trust_remote_code = True, attn_implementation="flash_attention_2")
    tokenizer = AutoTokenizer.from_pretrained(
  model_id,
)
elif model_id == "gptj":
    model_base = AutoModelForCausalLM.from_pretrained(
  "EleutherAI/gpt-j-6b",device_map = device, torch_dtype=torch.bfloat16,trust_remote_code = True)
    tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/gpt-j-6b",
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"
# %%
from datasets import load_from_disk
import random
import numpy as np
dataset_org = load_from_disk("sst2")
num_added_toks = tokenizer.add_tokens(["<mask>", "<counterfactual>"])
model_base.resize_token_embeddings(len(tokenizer))

from peft import LoraConfig, TaskType, get_peft_model
if model_id == "pythia2":
    target_modules = ["embed_in","dense","dense_h_to_4h","dense_4h_to_h","o_proj","embed_out"]
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,target_modules = target_modules, r=512, lora_alpha=1024, lora_dropout=0.1)
    model = get_peft_model(model_base, peft_config)
    model.print_trainable_parameters()
elif model_id == "phi3":
    target_modules = ["embed_tokens","o_proj","gate_up_proj","down_proj","lm_head"]
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,target_modules = target_modules, r=512, lora_alpha=1024, lora_dropout=0.1)
    model = get_peft_model(model_base, peft_config)
    model.print_trainable_parameters()
elif model_id == "gptj":
    target_modules = ["wte","out_proj","fc_in","fc_out","lm_head"]
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False,target_modules = target_modules, r=256, lora_alpha=512, lora_dropout=0.1)
    model = get_peft_model(model_base, peft_config)
    model.print_trainable_parameters()
else:
    model = model_base
# %%
model.to(device)

# %%

import copy
def add_mask(x):
    sent = x["sentence"]
    if x["label"] == 0:
        x["sentence"]+= "negative"
    elif x["label"] == 1:
        x["sentence"]+= "positive"
    length = len(tokenizer(sent)["input_ids"])

    x["sentence"] = x["sentence"] + " <counterfactual> " + sent
    x = tokenizer(x["sentence"])
    
    percent = random.uniform(0.05,0.5)
    mask_ids = np.random.choice(np.arange(length)-1,int(np.ceil(percent*length)),replace=False,)
    x["input_ids"] = np.array(x["input_ids"],dtype=np.int64)
    # print(mask_ids)
    MASK_ID = tokenizer.convert_tokens_to_ids("<mask>")
    x["input_ids"][mask_ids] = MASK_ID
    
    x["counter_sent"] = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(x["input_ids"]))
    x["it_labels"] = np.ones(max_seq_length, dtype=np.int64) * IGNORE_INDEX
    x["it_labels"][-length:] = copy.deepcopy(x["input_ids"][-length:])
    return x

def batching(x):
    x = tokenizer(x["counter_sent"], return_tensors="pt",padding="max_length",max_length=max_seq_length,truncation=True)
    return x
def data_processing(shuffled_data):
    data = shuffled_data.filter(lambda x:len(tokenizer(x["sentence"])["input_ids"])<=max_seq_length//2 - 1)
    data = data.map(add_mask)
    data = data.map(batching,batched=True,batch_size=batch_size)
    return data


# %%
from torch.optim import Adam
op = Adam(params=model.parameters(),lr=8e-5)

from tqdm import tqdm
num_epochs = 8
seeds = [10*x for x in range(num_epochs)]

for epoch in range(num_epochs):
    if epoch == 2:
        for param_group in op.param_groups:
            param_group['lr'] = 4e-5
    if epoch == 4:
        for param_group in op.param_groups:
            param_group['lr'] = 2e-5
    if epoch == 6:
        for param_group in op.param_groups:
            param_group['lr'] = 1e-5
    loss = 0
    model.eval()
    shuffled_data = dataset_org.shuffle(seed=seeds[epoch])
    data = data_processing(shuffled_data)
    with torch.no_grad():
        for i in tqdm(range(len(data["validation"])//batch_size)):
            ids = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["input_ids"]).to(device)
            mask = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["attention_mask"]).to(device)
            labels = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["it_labels"]).to(device)
            
            out = model(input_ids=ids,attention_mask=mask,labels=labels)
            l= out.loss

            loss += l.item()
    print(f"epoch: {epoch+1} || val loss:{loss}")
    loss = 0
    model.train()
    for i in tqdm(range(len(data["train"])//batch_size)):
        ids = torch.LongTensor(data["train"][i*batch_size:(i+1)*batch_size]["input_ids"]).to(device)
        mask = torch.LongTensor(data["train"][i*batch_size:(i+1)*batch_size]["attention_mask"]).to(device)
        labels = torch.LongTensor(data["train"][i*batch_size:(i+1)*batch_size]["it_labels"]).to(device)
        
        # with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids,attention_mask=mask,labels=labels)
        l= out.loss
        l.backward()
        if model_id == "phi3":

            accum = 5
        elif model_id=="gptj":
            accum = 10
        
        else:
            accum = 3

        if i%accum==0:

            op.step()
            op.zero_grad()            

        loss += l.item()
    print(f"epoch: {epoch+1} || train loss:{loss}")


    loss = 0
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(data["validation"])//batch_size)):
            ids = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["input_ids"]).to(device)
            mask = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["attention_mask"]).to(device)
            labels = torch.LongTensor(data["validation"][i*batch_size:(i+1)*batch_size]["it_labels"]).to(device)
            out = model(input_ids=ids,attention_mask=mask,labels=labels)
            l= out.loss

            loss += l.item()

    print(f"epoch: {epoch+1} || val loss:{loss}")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': op.state_dict(),
            }, f"counter_{model_id}_sst.pt")

# %%
if model_id != "pythia":
    model.merge_and_unload()
    torch.save(model.base_model.model.state_dict(),
                f"counter_{model_id}_sst2_finished.pt")
else:
    torch.save(model.state_dict(),
                f"counter_{model_id}_sst2.pt")