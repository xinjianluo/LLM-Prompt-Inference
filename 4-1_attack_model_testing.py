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
from utils.attack_utils import loadAttackModel_1, loadAttackModel_2

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
    LLM_outputs_folder = parameters["ActivationStorePath"]
    modeldir = parameters["LLM"]
    datasetdir_1 = parameters["Dataset"]
    datasetdir_2 = parameters['BaseDataset']
    targetdatasetdir_2 = parameters['TargetDataset_2']
    layer = parameters["AttackLayer"]
    text = parameters['TestPrompt']
    
    device = get_gpu_device()

    fulllayerfolder_1 = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{datasetdir_1}{os.sep}layer{layer}"
    fulllayerfolder_2 = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{datasetdir_2}{os.sep}layer{layer}"
        
    logging.critical(f"\nEvaluating on layer [{layer}] in LLM [{modeldir}]\n")

    # attackmodel_1 = loadAttackModel_1(layer, modeldir, datasetdir_1, fulllayerfolder_1, device)
    attackmodel_1 = loadAttackModel_1(layer, modeldir, datasetdir_2, fulllayerfolder_2, device)
    attackmodel_2 = loadAttackModel_2(layer, modeldir, datasetdir_2, targetdatasetdir_2, fulllayerfolder_2, device)
  
    model = loadModel(modeldir).to(device)
    tokenizer = loadTokenizer(modeldir)

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
    logging.critical("Reconstructed prompt via Attack 1 is:")
    with torch.no_grad():
        yhat_1 = attackmodel_1(activation[f"decoderlayer{layer}"].float()).detach()
    logging.critical(f"{reconstructSTR(yhat_1, tokenizer)}\n\n")

    # ------------------- Attack 2 -------------------
    logging.critical("Reconstructed prompt via Attack 2 is:")
    with torch.no_grad():
        yhat_2 = attackmodel_2(activation[f"decoderlayer{layer}"].float()).detach()
    logging.critical(f"{reconstructSTR(yhat_2, tokenizer)}\n\n")

    logging.critical("All finished!")