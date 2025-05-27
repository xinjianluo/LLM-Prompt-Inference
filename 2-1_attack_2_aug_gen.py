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
import gc

from utils.running_utils import get_gpu_device, initlogging, resetRandomStates, readConfigFile, progressWrapper, print_gpu_memory
from utils.dataset_utils import WordbyWordDataset
from utils.llm_utils import getDictsizeAndEOSID, loadModel, loadTokenizer, getmodelname

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
    
def getIDsandFreqsToAug(base, target, parameters, baselowcount=0):
    #  Find the elements in target that do not appear in base (set difference)
    base_unique, base_counts = base.unique(return_counts=True)
    target_unique = torch.unique(target)
    intersect = torch.tensor([x for x in target_unique if x not in base_unique])
    # print(intersect.shape)
    
    # Select the elements that appear less than baselowcount times in base
    target_less_than_50 = torch.tensor([x.item() for x, count in zip(base_unique, base_counts) if count < baselowcount])
    # print(target_less_than_50.shape)
    
    # Concatenate the two tensors
    concatenated = torch.cat((intersect, target_less_than_50))
    
    # Find the average and standard deviation of the frequencies of elements in base
    avg = base_counts.float().mean()
    std = base_counts.float().std()
    
    # Output the sum of average and standard deviation
    aug_freq = avg + std/3
    
    if parameters["RunningSamples_2"] > 0:
        concatenated = concatenated[:parameters["RunningSamples_2"]]
        
        
    return concatenated.int(), aug_freq

def getAugSeqIDsandTargetPos(tensor_to_interplate, target_token, EOS_token, device):
    """
        tensor_to_interplate: a tensor with size (batch, seq), with each row
                left padded with EOS_token; this function do not 
                directly modify this tensor. Rather, it will clone it.
    """
    # Clone the original tensor to avoid modifying the original one
    modified_tensor = tensor_to_interplate.clone().to(device)
    
    # Create an attention mask tensor
    attention_tensor = (modified_tensor != EOS_token).float().to(device)
    
    # Get the first occurrence of non-EOS token for each row
    non_eos_start_indices = attention_tensor.argmax(dim=1)
    
    # List to store the positions where replacements are made
    replacement_positions = []
    
    # Iterate over each sequence (row)
    for i in range(modified_tensor.size(0)):
        # Get the row (sequence)
        row = modified_tensor[i]
        
        # Determine the start of the actual sequence (non-EOS tokens)
        non_eos_start = non_eos_start_indices[i].item()
        if non_eos_start > 0:
            assert row[non_eos_start-1] == EOS_token and row[non_eos_start] != EOS_token, f"Error row[non_eos_start-1]={row[non_eos_start-1]}"
                
        # Select a random valid position
        random_position = torch.randint(low=non_eos_start, high=modified_tensor.size(1), size=(1,)).item()
        
        # Replace the token at the selected position with the target token
        modified_tensor[i, random_position] = target_token
        
        # Store the row index and the random position
        replacement_positions.append((i, random_position, random_position-non_eos_start))

    return modified_tensor, attention_tensor, replacement_positions
    

# ---------- the main function ----------
device = get_gpu_device()

parameters = readConfigFile('config.ini')
LLM_outputs_folder = parameters["ActivationStorePath"]
modeldir = parameters["LLM"]
layer = parameters["AttackLayer"]
logfilename = initlogging(parameters["LogPath"]) 
basedatasetdir = parameters['BaseDataset']
targetdatasetdir = parameters['TargetDataset_2']
aug_freq = parameters['AugDelta'] * 1.0
fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{basedatasetdir}{os.sep}layer{layer}"
model = loadModel(modeldir).to(device)

# register forward hooks on the layers of choice
h1, h2 = registerHooks(model, layer, modeldir)
########################################

resetRandomStates()
batchsize = 128
logging.critical(f"Processing layer=[{layer}], batchsize=[{batchsize}], modeldir=[{modeldir}], basedatasetdir=[{basedatasetdir}], targetdatasetdir=[{targetdatasetdir}]\n")
DICTSIZE, EOSID = getDictsizeAndEOSID(modeldir) 

# Read training data 
basedatapath = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{basedatasetdir}{os.sep}layer{layer}{os.sep}{modeldir}-{basedatasetdir}-layer{layer}-train-data"
tgtdatapath = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{targetdatasetdir}{os.sep}layer{layer}{os.sep}{modeldir}-{targetdatasetdir}-layer{layer}-test-data"

logging.critical(f"Reading base data from\n {basedatapath}\nand target data from\n {tgtdatapath}")
basedataset = WordbyWordDataset(basedatapath, EOSID)
tgtdataset = WordbyWordDataset(tgtdatapath, EOSID)

# To reduce the computing cost, here we assume the target token set is available
# You can also replace tgtdataset.catids with the LLM token dictionary
ID_to_aug, _ = getIDsandFreqsToAug(basedataset.catids, tgtdataset.catids, parameters)

#-------------------------------
ID_batches = torch.split(ID_to_aug, batchsize)
logging.critical(f"\nTotal {len(ID_to_aug)} target tokens to augment; Will produce [{len(ID_batches)}] files.")

base_padded_tokenid_lst, _, _ = torch.load(basedatapath, map_location=torch.device('cpu'), weights_only=False)

old_batch_size = base_padded_tokenid_lst[0].shape[0]
batch_per_token = round((aug_freq/old_batch_size))
if batch_per_token <= 0:
    batch_per_token = 1
logging.critical(f"The old batch size is {old_batch_size}; {batch_per_token} old batches are needed to augment each token\n")

start_time = time.time()
for batch_idx, batch_ID in enumerate(ID_batches):
    batchfilename = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{basedatasetdir}{os.sep}layer{layer}{os.sep}{modeldir}-layer{layer}-Embedding-Outputs-base{basedatasetdir}-target{targetdatasetdir}-batch{batch_idx}.lst"
    tgtdict_to_save = {}
    
    # for each target token, sample a batch of sequence IDs to interpolate
    for target_token in batch_ID:
        target_token = target_token.item()    # must be an integer
        tensors_to_interpolate = random.sample(base_padded_tokenid_lst, batch_per_token)
        
        # for each batch of tensor, interpolate the target token ID
        for tensor_intp in tensors_to_interpolate:
            # target_pos with [(idx, tensor position, sequence position)]
            interploated_tensor, attention_tensor, target_pos = getAugSeqIDsandTargetPos(tensor_intp, target_token, EOSID, device)
            # Feed interploated_tensor into LLM and get activations
            activation = {"token_embeddings":[], f"decoderlayer{layer}":[]}
                
            # interploated_tensor = interploated_tensor.to(device)
            # attention_tensor = attention_tensor.to(device)
            with torch.no_grad():
                output = model(input_ids = interploated_tensor, attention_mask = attention_tensor)

            act_embds = activation["token_embeddings"][0].detach()    # torch.Size([b, seq, 1600])
            act_output = activation[f"decoderlayer{layer}"][0][0].detach()    # torch.Size([b, seq, 1600])

            seq_num = len(act_embds)
            tgt_output_lst = []
            
            tgt_embd_to_save = act_embds[0][target_pos[0][1]]
            
            # validate embeddings based on target positions
            idx_embd_val = torch.randint(low=1, high=seq_num, size=(1,)).item()
            pos1 = target_pos[idx_embd_val][1] # absolute tensor position
            assert torch.equal(act_embds[idx_embd_val][pos1], tgt_embd_to_save), f"Target embeddings validation abnormal: act_embds[{idx_embd_val}][{pos1}] != tgt_embd_to_save"

            for i in range(seq_num):
                tgtpos = target_pos[i][1] # absolute tensor position
                tgtout = act_output[i][tgtpos:tgtpos+1]
                tgt_output_lst.append(tgtout)
            tgt_output_lst = torch.cat(tgt_output_lst, dim=0).to(device)
            assert tgt_output_lst.shape == (seq_num, act_output.shape[2]), f"Output tensor shape abnormal: tgt_output_lst.shape != ({seq_num}, {act_output.shape[2]})"

            _, _, seq_pos_lst = zip(*target_pos)
            seq_pos_lst = torch.tensor(seq_pos_lst).to(device)
            if target_token in tgtdict_to_save:
                assert torch.equal(tgt_embd_to_save, tgtdict_to_save[target_token][0]), f"Target embeddings validation abnormal"
                tgtdict_to_save[target_token][1].append(tgt_output_lst.clone())
                tgtdict_to_save[target_token][2].append(seq_pos_lst.clone())
            else:
                tgtdict_to_save[target_token] = [tgt_embd_to_save.clone(), [tgt_output_lst.clone(),], [seq_pos_lst.clone(),]]
            # Clear memory
            del output, interploated_tensor, attention_tensor, act_embds, act_output, activation
            gc.collect()
            torch.cuda.empty_cache()  
        tgtdict_to_save[target_token][0] = tgtdict_to_save[target_token][0].cpu()
        tgtdict_to_save[target_token][1] = torch.cat(tgtdict_to_save[target_token][1]).cpu()
        tgtdict_to_save[target_token][2] = torch.cat(tgtdict_to_save[target_token][2]).cpu()
    torch.save(tgtdict_to_save, batchfilename)
    logging.critical(f"File is saved to {batchfilename}")
    print_gpu_memory(device)
    del tgtdict_to_save
    gc.collect()
    logging.critical(f"Running Progress: {progressWrapper(start_time, len(ID_batches), batch_idx+1)}") 
    logging.critical("---------------------")
del model
gc.collect()

logging.critical("All finished!")
