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
import sys

import configparser
torch.set_printoptions(precision=4, sci_mode=False) 

def loadModel(modeldir, MustLHead=False, output_hidden_states=False):  
    modelname = getmodelname(modeldir)
    
    logging.critical(f"Loading model {modelname}")
    if modelname == "microsoft/phi-2":
        model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype="auto", trust_remote_code=True)     
    elif modelname == "gpt2-large":
        if MustLHead:
            model = AutoModelForCausalLM.from_pretrained("gpt2", output_hidden_states=output_hidden_states)
        else:
            from transformers import GPT2Model   
            model = GPT2Model.from_pretrained(modelname, output_hidden_states=output_hidden_states)
    elif modelname == "meta-llama/Llama-3.2-1B":           
        model = AutoModelForCausalLM.from_pretrained(modelname)
    elif modelname == "microsoft/Phi-3.5-mini-instruct":           
        model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype="auto", trust_remote_code=True)
    elif modelname == "bert-large-cased":           
        from transformers import BertModel
        model = BertModel.from_pretrained(modelname, output_hidden_states=output_hidden_states)
    else:
        raise ValueError(f'Unsupported model name {modelname}') 
                
    return model

def getmodelname(modeldir):
    # Gpt2, Llama3, Phi3, Bert
    if modeldir == "Phi2":
        modelname = "microsoft/phi-2"
    elif modeldir == "Gpt2":
        modelname = "gpt2-large"
    elif modeldir == "Llama3":
        modelname = "meta-llama/Llama-3.2-1B"  
    elif modeldir == "Phi3":
        modelname = "microsoft/Phi-3.5-mini-instruct"  
    elif modeldir == "Bert":
        modelname = "bert-large-cased" 
    else: 
        raise ValueError(f'Unknown modeldir {modeldir}')
    return modelname
    
def getDictsizeAndEOSID(modeldir):
    # Gpt2, Llama3, Phi3, Bert
    if modeldir == "Phi2":
        DICTSIZE = 51200  
        EOSID = 50256
    elif modeldir == "Gpt2":
        DICTSIZE = 50257
        EOSID = 50256
    elif modeldir == "Llama3":
        DICTSIZE = 128256
        EOSID = 128001
    elif modeldir == "Phi3":
        DICTSIZE = 32064
        EOSID = 32000
    elif modeldir == "Bert":
        DICTSIZE = 28996
        EOSID = 0
    else:
        raise ValueError(f"Unknown model directory {modeldir} in getDictsizeAndEOSID()")  
    return DICTSIZE, EOSID

def loadTokenizer(modeldir):   
    modelname = getmodelname(modeldir)
        
    logging.critical(f"Loading tokenizer for {modelname}")
    if modelname == "microsoft/phi-2":
        tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True, padding_side='left', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif modelname == "gpt2-large":        
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(modelname, padding_side='left', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif modelname == "meta-llama/Llama-3.2-1B":        
        tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='left', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif modelname == "microsoft/Phi-3.5-mini-instruct":           
        tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True, padding_side='left', clean_up_tokenization_spaces=True)
        tokenizer.pad_token = tokenizer.eos_token
    elif modelname == "bert-large-cased":           
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(modelname, trust_remote_code=True, padding_side='left', clean_up_tokenization_spaces=True)
        # tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError(f'Unsupported model name {modelname}') 
                
    return tokenizer  
