import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
import logging
import random 
import os 
import time 
import math

from utils.running_utils import get_gpu_device, checkCreateFolder, resetRandomStates, initlogging, readConfigFile, print_gpu_memory
from utils.llm_utils import loadModel, loadTokenizer, getmodelname, getDictsizeAndEOSID
from utils.attack_utils import MLPClassHead

def getActivation(name):
    # the hook signature
    def hook(model, input, output):
        activation[name].append(output)
    return hook
    
def registerHooks(model, layer, modeldir):
    modelname = getmodelname(modeldir)
    # register forward hooks on the layers of choice
    # it will automatically raise an error if cannot find the corresponding attributes in the model instance

    logging.critical(f"Registering hooks for model {modelname} and layer {layer}")
    if modelname == "microsoft/phi-2":
        h1 = model.model.embed_tokens.register_forward_hook(getActivation('token_embeddings'))
        h2 = model.model.layers[layer].register_forward_hook(getActivation(f'decoderlayer{layer}'))
    elif modelname == "gpt2-large":        
        h1 = model.wte.register_forward_hook(getActivation('token_embeddings'))
        h2 = model.h[layer].register_forward_hook(getActivation(f'decoderlayer{layer}'))
    elif modelname == "meta-llama/Llama-3.2-1B":
        h1 = model.model.embed_tokens.register_forward_hook(getActivation('token_embeddings'))
        h2 = model.model.layers[layer].register_forward_hook(getActivation(f'decoderlayer{layer}')) 
    elif modelname == "microsoft/Phi-3.5-mini-instruct":           
        h1 = model.model.embed_tokens.register_forward_hook(getActivation('token_embeddings'))
        h2 = model.model.layers[layer].register_forward_hook(getActivation(f'decoderlayer{layer}'))
    elif modelname == "bert-large-cased":           
        h1 = model.embeddings.word_embeddings.register_forward_hook(getActivation('token_embeddings'))
        h2 = model.encoder.layer[layer].register_forward_hook(getActivation(f'decoderlayer{layer}'))
    else:
        raise ValueError(f'Unsupported model name {modelname}')    
    return h1, h2
    
def reconstructSTR(yhatlst, tokenizer):
    predtokens = []
    for i in range(len(yhatlst)): 
        tkstr = yhatlst[i].argmax().item()
        predtokens.append(tkstr) 
    predstr = tokenizer.decode(predtokens)
    return predstr

# ------------------------------------------------------------------
if __name__ == "__main__":
    parameters = readConfigFile('config.ini')
    logfilename = initlogging(parameters["LogPath"]) 

    ###################################################
    # fix these parameters
    modeldir = "Gpt2"
    basedataset = "Wikitext2"
    targetdataset = "SQuAD2.0"
    layer = 18
    ###################################################
    text = parameters['TestPrompt']
    
    device = get_gpu_device()

    modelfolder_1 = f".{os.sep}pretrained{os.sep}Gpt2-Wikitext2_Train-MLPClassHead-Finetune-WbyW-cla-layer18.mdl"
    modelfolder_2 = f".{os.sep}pretrained{os.sep}Gpt2-Wikitext2_Train-MLPClassHead-Finetune-WbyW-cla-layer18-augSQuAD2.0.mdl"
        
    logging.critical(f"\nEvaluating on layer [{layer}] in LLM [{modeldir}]\n")

    model = loadModel(modeldir).to(device)
    tokenizer = loadTokenizer(modeldir)
    attackmodel_1 = MLPClassHead(model.config.hidden_size, model.config.hidden_size, model.config.vocab_size).to(device)
    attackmodel_1.load_state_dict(torch.load(modelfolder_1, weights_only=False))
    logging.critical(f"\nFor Attack 1, loading model from [{modelfolder_1}]")
    attackmodel_2 = MLPClassHead(model.config.hidden_size, model.config.hidden_size, model.config.vocab_size).to(device)
    attackmodel_2.load_state_dict(torch.load(modelfolder_2, weights_only=False))
    logging.critical(f"\nFor Attack 2, loading model from [{modelfolder_2}]\n")

    # register forward hooks on the layers of choice
    h1, h2 = registerHooks(model, layer, modeldir)

    activation = {"token_embeddings":[], f"decoderlayer{layer}":[]}
    encoded_input = tokenizer(text, return_tensors='pt', padding=True) 
        
    with torch.no_grad():
        output = model(input_ids = encoded_input['input_ids'].to(device), attention_mask = encoded_input['attention_mask'].to(device))

    activation[f"decoderlayer{layer}"] = activation[f"decoderlayer{layer}"][0][0].squeeze()
        
    logging.critical("\n----------------------------------")
    logging.critical("\nGround truth is:")
    logging.critical(f"{text}\n\n")
    
    # ------------------- Attack 1 -------------------
    logging.critical(f"Reconstructed prompt via Attack 1 (trained on {basedataset}) is:")
    with torch.no_grad():
        yhat_1 = attackmodel_1(activation[f"decoderlayer{layer}"].float()).detach()
    logging.critical(f"{reconstructSTR(yhat_1, tokenizer)}\n\n")

    # ------------------- Attack 2 -------------------
    logging.critical(f"Reconstructed prompt via Attack 2 (trained on {basedataset}) is:")
    with torch.no_grad():
        yhat_2 = attackmodel_2(activation[f"decoderlayer{layer}"].float()).detach()
    logging.critical(f"{reconstructSTR(yhat_2, tokenizer)}\n\n")

    logging.critical("All finished!")