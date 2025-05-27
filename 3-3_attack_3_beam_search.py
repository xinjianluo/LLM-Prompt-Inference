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

from utils.running_utils import get_gpu_device, checkCreateFolder, resetRandomStates, progressWrapper, print_gpu_memory, initlogging, readConfigFile
from utils.llm_utils import loadModel, loadTokenizer, getmodelname, getDictsizeAndEOSID
from utils.dataset_utils import filterDatasetFreqs, readRandomDatasetforAdv3, WordbyWordDataset, readDataset
from utils.attack_utils import MLPRegressor, MLPClassHead
        
def getTopkLabels(center_labels, center_tensors, point_tensors, topk=5):
    pnorm = 1
 
    unique_labels = torch.unique(center_labels)
    num_clusters = len(unique_labels)
    
    cluster_masks = torch.stack([center_labels == label for label in unique_labels])
    index2labelDICT = {i:unique_labels[i].item() for i in range(len(unique_labels))}
    label2indexDICT = {unique_labels[i].item():i for i in range(len(unique_labels))}
    
    logging.critical(f"center_tensors.shape: {center_tensors.shape}")
    logging.critical(f"point_tensors.shape: {point_tensors.shape}")
    
    # do not run p=1 norm on GPU, wrong results
    distances = torch.empty(point_tensors.shape[0], num_clusters, device=point_tensors.device)
    for cluster_idx, mask in enumerate(cluster_masks):
        # Select points belonging to the current cluster
        cluster_points = center_tensors[mask]
        # Compute pairwise distances and average them
        distances[:, cluster_idx] = torch.cdist(point_tensors, cluster_points, p=pnorm).mean(dim=1)

    mindists, minindices = distances.topk(topk, dim=1, largest=False)
    minindices = minindices.cpu()
    topk_labels = torch.zeros_like(minindices)
    for i in range(minindices.shape[1]):
        topk_labels[:, i] = torch.tensor([index2labelDICT[ind.item()] for ind in minindices[:, i]])
  
    return topk_labels 

def compACC(point_labels, point_labels_gt):
    acc = (point_labels_gt == point_labels).sum() / len(point_labels)
    return acc.item()    

def find_pairs(tensor, max_len=20):
    pairs = []
    n = len(tensor)
    i = 0
    # Loop through the tensor to form pairs
    while i < n:
        if tensor[i] == -1:
            i += 1
            continue
        # Now we try to maximize j while ensuring j - i < 20
        max_j = -1
        for k in range(i + 1, min(i + max_len, n)):
            if tensor[k] != -1:
                max_j = k
        
        if max_j != -1:
            pairs.append((i, max_j))
            i = max_j  # Set i to max_j for the next pair
        else:
            if k == n-1:
                pairs.append((i, n-1))
                break
            else:
                for tk in range(k, n):
                    if tensor[tk] != -1:
                        break
                pairs.append((i, tk))
                i = tk  # Set i to max_j for the next pair
    return pairs
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    parameters = readConfigFile('config.ini')
    logfilename = initlogging(parameters["LogPath"]) 
    LLM_outputs_folder = parameters["ActivationStorePath"]
    modeldir_shadow = parameters["LLM"]
    datasetdir_shadow = parameters['TargetDataset_3']
    layer = parameters["AttackLayer"]
    querybudgets = parameters['QueryBudget']
    beam_width = parameters['BeamWidth']
    shadowLLM = parameters['ShadowLLM']
    device = get_gpu_device()

    fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir_shadow}{os.sep}{datasetdir_shadow}{os.sep}layer{layer}"
    semanticModel = loadModel(shadowLLM, MustLHead=True).to(device)
        
    logging.critical(f"\nProcessing layer [{layer}] in LLM [{modeldir_shadow}] for dataset [{datasetdir_shadow}] with querybudgets [{querybudgets}]")
    logging.critical(f"Set the shadow LLM to [{shadowLLM}] and beam width to [{beam_width}]")
    
    
    DICTSIZE_sdw, EOSID_sdw = getDictsizeAndEOSID(modeldir_shadow) 
    logging.critical(f"DICTSIZE is set to [{DICTSIZE_sdw}]; EOSID is set to [{EOSID_sdw}]\n") 
    

    # Read training data 
    trainwwset = readRandomDatasetforAdv3(layer, modeldir_shadow, datasetdir_shadow, EOSID_sdw, fulllayerfolder, promptstr="Randomly Generated Embeddings")
    trainwwset = filterDatasetFreqs(trainwwset, FREQ=querybudgets)
    logging.critical(f"After filtering, random dataset size is {len(trainwwset)}\n")

    testwwset = readDataset(layer, modeldir_shadow, datasetdir_shadow, EOSID_sdw, fulllayerfolder, train=False, promptstr="Real Test Dataset")
    
    resetRandomStates(0)

    gt_len = len(trainwwset) if len(trainwwset) < 30000 else 30000
    randidx = torch.tensor(list(range(gt_len)))
    center_labels = trainwwset.catids.to(device)
    center_tensors = trainwwset.catlayer0s.to(device)
    point_tensors = testwwset.catlayer0s[randidx].clone().to(device)
    gt_labels = testwwset.catids[randidx].clone().to(device)

    label_load_path = os.path.join(fulllayerfolder, f"attack3-{modeldir_shadow}-{datasetdir_shadow}-layer{layer}-budgets{querybudgets}-point_labels-point_labels_gt-After-MLPSearch.ts") 
    point_labels, point_labels_gt, _ = torch.load(label_load_path, map_location=device, weights_only=False)

    assert (point_labels_gt-gt_labels).sum() == 0, f"{(point_labels_gt-gt_labels).sum()} != 0"

    topk_labels = getTopkLabels(center_labels.cpu(), center_tensors.cpu(), point_tensors.cpu()).to(device)

    fill_labels = point_labels
    logging.critical("Before Beam Search:")
    logging.critical(f"\t{(fill_labels!=-1).sum() / len(fill_labels)} percentage points are assigned labels;\n\tAccuracy is {compACC(point_labels, point_labels_gt)}\n")

    beam_labels = -torch.ones(len(fill_labels), dtype=torch.long).to(device)

    index_pairs = find_pairs(point_labels)
    for startidx, endidx in tqdm(index_pairs, miniters=20):
        # Initialize the beams with the start token
        beams = []
        beams.append((torch.tensor([fill_labels[startidx].item()]).to(device), 0))
        # print(beams)
        
        for counter in range(startidx+1, endidx):    
            new_beams = []
            # print(counter)
            for seq, score in beams:
                if point_labels[counter] != -1:
                    new_seq = torch.cat([seq, point_labels[counter].view(1)])
                    new_beams.append((new_seq, score))
                    continue 
                else:
                    # Get model output probabilities for the next token
                    with torch.no_grad():
                        outputs = semanticModel(input_ids=seq.unsqueeze(0).to(device) ) # Shape: (1, seq_len, vocab_size)
                        logits = outputs.logits[:, -1, :]         # Take the last step's output
                        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # Shape: (vocab_size,)
                    
                    # Get top `beam_width` tokens and their log-probabilities
                    top_tokens = topk_labels[counter, :]
                    top_log_probs = log_probs[top_tokens].clone()
                    # print(f"top_tokens {top_tokens}, top_log_probs")
                    
                    # Expand each beam
                    for token, token_log_prob in zip(top_tokens, top_log_probs):
                        new_seq = torch.cat([seq, token.view(1)])
                        new_score = score + token_log_prob.item()
                        new_beams.append((new_seq, new_score))
                # print(new_beams)
                # print("---------\n")
            # Keep only the top `beam_width` beams based on score
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            beams = new_beams
        # add the final token
        cand_scores = torch.zeros(len(beams))
        for i, (seq, score) in enumerate(beams):
            # print(seq, score, counter)
            # Get model output probabilities for the next token
            with torch.no_grad():
                outputs = semanticModel(input_ids=seq.unsqueeze(0).to(device) ) # Shape: (1, seq_len, vocab_size)
                logits = outputs.logits[:, -1, :]         # Take the last step's output
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(0)  # Shape: (vocab_size,)
            
            # Get top `beam_width` tokens and their log-probabilities
            final_token = fill_labels[endidx]
            top_log_probs = log_probs[final_token]
            cand_scores[i] = top_log_probs
        cand_beam = beams[cand_scores.argmax()]
        beam_labels[startidx:endidx] = cand_beam[0]

    # save reconstructed labels
    label_save_path = os.path.join(fulllayerfolder, f"attack3-{modeldir_shadow}-{datasetdir_shadow}-layer{layer}-budgets{querybudgets}-point_labels-point_labels_gt-After-BeamSearch.ts") 
    torch.save((beam_labels, fill_labels, gt_labels), label_save_path)
    logging.critical(f"Beam search and ground truth labels are saved to {label_save_path}")
    
    # compute reconstruction accuracy
    beam_mask = (fill_labels == -1).to(device)
    fill_labels[beam_mask] = beam_labels[beam_mask]
    logging.critical("After Beam Search:")
    logging.critical(f"\t{(fill_labels!=-1).sum() / len(fill_labels)} percentage points are assigned labels;\n\tAccuracy is {compACC(fill_labels, point_labels_gt)}\n")

    logging.critical(f"Logs were written to {logfilename}")
    logging.critical("All finished!")