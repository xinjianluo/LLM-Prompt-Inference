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

class SemiSupervisedDataset(Dataset):

    def __init__(self, center_tensors, center_labels, point_tensors, point_labels):
        # center_tensors = center_tensors.repeat(3, 1)
        # center_labels = center_labels.repeat(3)
        
        mask = point_labels != -1
        fpt = point_tensors[mask]
        fpl = point_labels[mask]
        assert (fpl == -1).sum() == 0, f"{(fpl == -1).sum()} != 0"
        self.X = torch.cat([center_tensors, fpt], dim=0)
        assert self.X.shape[1] == point_tensors.shape[1], f"{self.X.shape[1]} != {point_tensors.shape[1]}"
        self.Y = torch.cat([center_labels, fpl], dim=0)
        assert len(self.X) == len(self.Y), f"{len(self.X)} != {len(self.Y)}"
        
    def getmodeldim(self):
        return self.X.shape[1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.Y[index]
        
class WordbyWordDataset(Dataset):

    def __init__(self, datafilepath, EOSID):
        '''
            token_ids: a list of N tensors, each tensor with size [(256, k)]; k being the number of tokens in each sentence
            token_embeddings: a list of N tensors, each tensor with size [(256, k, 2560)]
            layer0_outputs: a list of N tensors, each tensor with size [(256, k, 2560)]
        '''
        token_ids, token_embeddings, layer0_outputs = torch.load(datafilepath, map_location=torch.device('cpu'), weights_only=False) 
        
        catids_list = []
        catembeddings_list = []
        catlayer0s_list = []
        catpos_list = []
        
        # first, find the last EoS position before each sentence
        for i, pbatchid in enumerate(token_ids): 
            # pbatchid with size (256, k)
            pbatchembd = token_embeddings[i]    # with size (256, k, 2560)
            pbatchlayer0 = layer0_outputs[i]    # with size (256, k, 2560)
             
            for j, pseqid in enumerate(pbatchid):  
                # pseqid with size (k)
                eospos = 0
                for k in range(len(pseqid)):
                    if pseqid[k] != EOSID:
                        eospos = k 
                        break 
                pseqembd = pbatchembd[j] # with size (k, 2560)
                pseqlayer0 = pbatchlayer0[j] # with size (k, 2560)
                assert len(pseqid) == pseqembd.size(0) == pseqlayer0.size(0), f"size not match: {len(pseqid)} != {pseqembd.size()} != {pseqlayer0.size()}"

                catpos_list.append(torch.tensor(range(len(pseqid[eospos:]))))
                catids_list.append(pseqid[eospos:])
                catembeddings_list.append(pseqembd[eospos:]) 
                catlayer0s_list.append(pseqlayer0[eospos:])
        
        # Concatenate once after collecting all data
        self.catpos = torch.cat(catpos_list, dim=0)
        self.catids = torch.cat(catids_list, dim=0)
        self.catembeddings = torch.cat(catembeddings_list, dim=0).float()
        self.catlayer0s = torch.cat(catlayer0s_list, dim=0).float()
        logging.critical(f"Creat Dataset Finished! catpos with {self.catpos.size()}, catids with {self.catids.size()}, catembeddings with {self.catembeddings.size()}, catlayer0s with {self.catlayer0s.size()} ")
                      
    def __len__(self):
        return len(self.catembeddings)
        
    def getmodeldim(self):
        return self.catlayer0s.shape[1]

    def __getitem__(self, index):
        return self.catlayer0s[index], self.catembeddings[index], self.catids[index], self.catpos[index]

def readDataset(layer, modeldir_target, datasetdir_target, EOSID_tgt, fulllayerfolder, train=True, promptstr=""):
    trainstr = "train" if train else "test"
    traindatapath_tgt = os.path.join(fulllayerfolder, f"{modeldir_target}-{datasetdir_target}-layer{layer}-{trainstr}-data")
    logging.critical(f"Reading {promptstr} of {trainstr} from {traindatapath_tgt}")
    tgtwwset = WordbyWordDataset(traindatapath_tgt, EOSID_tgt)
    logging.critical(f"Reading {promptstr} of {trainstr} finished!\n")
    return tgtwwset

class AugmentedWordbyWordDataset(Dataset):

    def __init__(self, modeldir, basedatasetdir, targetdatasetdir, EOSID, n_aug_files, layer, LLM_outputs_folder):
        filesuffix=""
        aug_embed_lst = []
        aug_output_lst = []
        aug_pos_lst = []
        aug_tid_lst = []
        fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{basedatasetdir}{os.sep}layer{layer}"
        logging.critical("Reading Augmented Files...")
        for curr_file_idx in tqdm(range(n_aug_files)):
            currfilename = os.path.join(fulllayerfolder, f"{modeldir}-layer{layer}-Embedding-Outputs-base{basedatasetdir}-target{targetdatasetdir}-batch{curr_file_idx}{filesuffix}.lst")
            
            # A dict with shape {id: [one embedding, N representations, N sequence positions]}
            currdict = torch.load(currfilename, map_location=torch.device('cpu'), weights_only=False)
            for tkid, assodata in currdict.items():
                # print(tkid)
                idembd, idoutputs, idpos = assodata
                n_augs, out_dim = idoutputs.shape
                assert n_augs == len(idpos), f"n_augs {n_augs} != len(idpos) {len(idpos)}"
                assert out_dim == len(idembd), f"out_dim {out_dim} != len(idembd) {len(idembd)}"
                repeated_idembd = idembd.unsqueeze(0).repeat(n_augs, 1)
                aug_embed_lst.append(repeated_idembd)
                aug_output_lst.append(idoutputs)
                aug_pos_lst.append(idpos)
                aug_tid_lst.append(torch.tensor([tkid]).repeat(n_augs))
                # print(idembd.shape, idoutputs.shape, idpos.shape)
                # print()
        aug_embed_lst = torch.cat(aug_embed_lst)
        aug_output_lst = torch.cat(aug_output_lst)
        aug_pos_lst = torch.cat(aug_pos_lst)
        aug_tid_lst = torch.cat(aug_tid_lst)
        logging.critical(f"Reading Augmented files finished!")
        logging.critical(f"aug_tid_lst {aug_tid_lst.shape}, aug_embed_lst {aug_embed_lst.shape}, aug_output_lst {aug_output_lst.shape}, aug_pos_lst {aug_pos_lst.shape}")
        
        traindatapath = os.path.join(fulllayerfolder, f"{modeldir}-{basedatasetdir}-layer{layer}-train-data")  # Train 21.9G; test 5.16G
        logging.critical(f"Reading base data from {traindatapath}")
        wwset = WordbyWordDataset(traindatapath, EOSID)

        self.finalpos = torch.cat((aug_pos_lst, wwset.catpos))
        self.finaltid = torch.cat((aug_tid_lst, wwset.catids))
        self.finalembds = torch.cat((aug_embed_lst, wwset.catembeddings))
        self.finaloutputs = torch.cat((aug_output_lst, wwset.catlayer0s))

        logging.critical(f"Creat Dataset Finished! finalpos {self.finalpos.size()}, finaltid {self.finaltid.size()}, finalembds {self.finalembds.size()}, finaloutputs {self.finaloutputs.size()} ")

    def __len__(self):
        return len(self.finalembds)
        
    def getmodeldim(self):
        return self.finaloutputs.shape[1]

    def __getitem__(self, index):
        return self.finaloutputs[index], self.finalembds[index], self.finaltid[index], self.finalpos[index]
        
def readRandomDatasetforAdv3(layer, modeldir_target, dataset_target, EOSID_tgt, fulllayerfolder, promptstr="Randomly Generated Embeddings"):
    traindatapath_tgt = os.path.join(fulllayerfolder, f"{modeldir_target}-{dataset_target}-layer{layer}-attack3-randomdata")
    logging.critical(f"Reading {promptstr} from {traindatapath_tgt}")
    tgtwwset = WordbyWordDataset(traindatapath_tgt, EOSID_tgt)
    logging.critical(f"Reading {promptstr} finished!")
    return tgtwwset
    
def filterDataset(wwset, filterts=None):
    assert wwset.__class__.__name__ == "WordbyWordDataset", f"Got unexpected dataset type {wwset.__class__.__name__}"
    if filterts is None:
        filterts = torch.tensor([  12,   13,   29,   31,   82,  198,  220,  257,  262,  281,  284,  286,
         287,  290,  318,  319,  326,  329,  339,  340,  351,  355,  357,  366,
         373,  379,  383,  416,  422,  465,  547,  550,  554,  705,  764,  837, 1267, 1279, 2488, 2954])
    filteridxes = torch.isin(wwset.catids, filterts)
    wwset.catembeddings = wwset.catembeddings[filteridxes]
    wwset.catids = wwset.catids[filteridxes]
    wwset.catlayer0s = wwset.catlayer0s[filteridxes]
    wwset.catpos = wwset.catpos[filteridxes]
    return wwset
    
def filterDatasetFreqs(dset, FREQ=None):
    assert dset.__class__.__name__ == "WordbyWordDataset", f"Got unexpected dataset type {dset.__class__.__name__}"
    if FREQ is None:
        return dset
    
    totalids = dset.catids
    uniqids = totalids.unique()
    id2freqs = {uid.item():0 for uid in uniqids}
    
    filterIdx = torch.zeros(len(totalids)).bool()  ## a boolean tensor filled with False
    for i in range(len(totalids)):
        tid = totalids[i].item()
        if id2freqs[tid] < FREQ:
            filterIdx[i] = True
            id2freqs[tid] += 1

    dset.catembeddings = dset.catembeddings[filterIdx]
    dset.catids = dset.catids[filterIdx]
    dset.catlayer0s = dset.catlayer0s[filterIdx]
    dset.catpos = dset.catpos[filterIdx]
    return dset   
        