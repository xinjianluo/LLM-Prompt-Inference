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

from utils.running_utils import progressWrapper, print_gpu_memory
from utils.llm_utils import getDictsizeAndEOSID
from utils.dataset_utils import WordbyWordDataset

import configparser
torch.set_printoptions(precision=4, sci_mode=False) 

'''
    1>8>1 Sigmoid,              train 85.4, test 170
    1>4>8>4>1 Sigmoid,          train 18.8, test 37.5
    1>4>8>4>1 ReLU,             train 0.75, test 0.72
    1>4>8>4>1 ReLU Dropout 0.5, train 2.09, test 2.09
    1>4>8>4>1 GELU,             train 2.09, test 2.14
    1>4>16>4>1 ReLU,            train 0.79, test 0.80
    1>4>8>8>4>1 ReLU,           train 0.71, test 0.70
    1>4>8>8>4>1 ReLU LayerNorm, train 0.13, test 0.15
'''
class MLPRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPRegressor, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(input_dim, input_dim*4),
            nn.LayerNorm(input_dim*4),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(input_dim*4, input_dim*8),
            nn.LayerNorm(input_dim*8),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(input_dim*8, output_dim*8),
            nn.LayerNorm(output_dim*8),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(output_dim*8, output_dim*4),
            nn.LayerNorm(output_dim*4),
            # nn.Dropout(p=0.5),
            nn.ReLU(),
            
            nn.Linear(output_dim*4, output_dim),
        )
        
    def forward(self, x):
        return self.dense(x)
        
class MLPClassHead(nn.Module):
    def __init__(self, input_dim, embed_dim, DICTSIZE, pretrainedModelPath=None): 
        # This parameter "pretrainedModelPath" is for class "MLPRegressor"
        super(MLPClassHead, self).__init__()
        self.mlp = MLPRegressor(input_dim, embed_dim)
        if pretrainedModelPath is not None: 
            logging.critical(f"In MLPClassHead, load pretrained model from: {pretrainedModelPath}")
            self.mlp.load_state_dict(torch.load(pretrainedModelPath, map_location=torch.device('cpu'), weights_only=False))	
        self.linearhead = nn.Linear(embed_dim, DICTSIZE)
        
    def forward(self, x):
        
        return self.linearhead(self.mlp(x))

def loadAttackModel_1(modellayer, modeldir, basedatasetdir, fulllayerfolder, device):
    logging.critical(f"Loading attack model for Attack 1, trained with auxiliary dataset = [{basedatasetdir}]")
 
    # Read dataset class for info of attack input dimension
    DICTSIZE, EOSID = getDictsizeAndEOSID(modeldir)
    testdatapath = os.path.join(fulllayerfolder, f"{modeldir}-{basedatasetdir}-layer{modellayer}-test-data")
    testset = WordbyWordDataset(testdatapath, EOSID)
    mlpModel = MLPClassHead(testset.getmodeldim(), testset.getmodeldim(), DICTSIZE, None).to(device)
    
    modelpath = os.path.join(fulllayerfolder, f"{modeldir}-{basedatasetdir}_Train-MLPClassHead-Finetune-cla-layer{modellayer}.mdl")
    
    mlpModel.load_state_dict(torch.load(modelpath, map_location=device, weights_only=False))
    logging.critical(f"Load attack 1 model from\n {modelpath}\n")
    return mlpModel
    
def loadAttackModel_2(modellayer, modeldir, basedatasetdir, testdatasetdir, fulllayerfolder, device):
    logging.critical(f"Loading attack model for Attack 2, trained with auxiliary dataset = [{basedatasetdir}], and target dataset = [{testdatasetdir}]")

    # Read dataset class for info of attack input dimension
    DICTSIZE, EOSID = getDictsizeAndEOSID(modeldir)
    testdatapath = os.path.join(fulllayerfolder, f"{modeldir}-{basedatasetdir}-layer{modellayer}-test-data")
    testset = WordbyWordDataset(testdatapath, EOSID)
    mlpModel = MLPClassHead(testset.getmodeldim(), testset.getmodeldim(), DICTSIZE, None).to(device)
    
    modelpath = os.path.join(fulllayerfolder, f"{modeldir}-{basedatasetdir}_Train-MLPClassHead-Finetune-cla-layer{modellayer}-aug{testdatasetdir}.mdl")
    mlpModel.load_state_dict(torch.load(modelpath, map_location=device, weights_only=False))
    logging.critical(f"Load attack 2 model from\n {modelpath}\n")
    return mlpModel
    
def tester(model, testloader, loss_fn, device, mode="reg"):
    '''   
        model: reg or cla
    '''
    model.eval()
    accloss = 0.0
    accacc = 0.0 
    accloss_base = 0
    with torch.no_grad():
        for lyout, embd, ids, _ in testloader:
            if mode == "reg":
                x = lyout 
                y = embd 
            elif mode == "cla":
                x = lyout 
                y = ids 
            else:
                raise ValueError(f'Unsupported mode type {mode} in testing') 
                
            x = x.to(device) 
            y = y.to(device) 
            
            yhat = model(x)
            loss = loss_fn(yhat, y)
            if mode == "cla":
                accacc += ((yhat.argmax(dim=1))==y).sum()
            accloss += loss.detach().cpu()
            accloss_base += x.shape[0]

    return accloss / accloss_base, accacc / accloss_base

def trainer(model, optimizer, traindataset, testdataset, device, epochs=1, test_interval=1, mode="reg", loss_graph=False):
    '''
        model: reg or cla
    '''   
    trainloss_list = [] 
    testloss_list = []
    if mode == "reg":
        loss_fn = torch.nn.MSELoss(reduction="sum")
    elif mode == "cla": 
        loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
        trainacc_list = [] 
        testacc_list = [] 
    else: 
        raise ValueError(f'Unsupported mode type {mode} in training') 
        
    logging.critical(f"In trainer(), model={model.__class__.__name__}, traindataset={traindataset.__class__.__name__}, testdataset={testdataset.__class__.__name__}, device={device}, epochs={epochs}, test_interval={test_interval}, mode={mode}")
    if testdataset is None:
        resetRandomStates(0)
        train_set, test_set = torch.utils.data.random_split(traindataset, [0.8, 0.2])
    else:
        train_set = traindataset
        test_set = testdataset
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)  # small batch like 128 and 256 can help improve performance
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
    
    model.train()
    start_time = time.time() 
    for epoch in (range(epochs)):
        accloss = 0.0
        accacc = 0.0 
        accloss_base = 0.0
        for lyout, embd, ids, _ in trainloader:     
            optimizer.zero_grad()
            if mode == "reg":
                x = lyout 
                y = embd 
            elif mode == "cla":
                x = lyout 
                y = ids 
            else:
                raise ValueError(f'Unsupported mode type {mode} in training') 
            x = x.to(device) 
            y = y.to(device)
            yhat = model(x)
            loss = loss_fn(yhat, y)
            
            loss.backward()
            optimizer.step()
            if mode == "cla": 
                accacc += ((yhat.argmax(dim=1))==y).sum()
            accloss += loss.detach().cpu()
            accloss_base += x.shape[0]
            
        if (epoch % test_interval == 0) or epoch == epochs - 1:
            # for each epoch, print information
            trainloss = accloss / accloss_base
            testloss, testacc = tester(model, testloader, loss_fn, device, mode)
            trainloss_list.append(trainloss)
            testloss_list.append(testloss)
            logging.critical("Epoch {}, train loss is {}, test loss is {}.".format(epoch, trainloss, testloss))
            if mode == "cla": 
                trainacc = accacc / accloss_base
                trainacc_list.append(trainacc)
                testacc_list.append(testacc)
                logging.critical("         train acc is {}, test acc is {}.".format(trainacc, testacc))
            model.train()
            logging.critical(f"Training Progress: {progressWrapper(start_time, epochs, epoch+1)}") 
            print_gpu_memory(device)
            logging.critical("--------------------------")

    if loss_graph:
        import matplotlib.pyplot as plt
        xticks = range(len(trainloss_list))
        plt.figure()
        # Plot the first line (red dotted line with square markers)
        plt.plot(xticks, trainloss_list, 'r-.s', label='Train Losses') 
        # Plot the second line (green solid line with circle markers)
        plt.plot(xticks, testloss_list, 'g-o', label='Test Losses')  
        plt.legend()
        plt.show()
        if mode == "cla": 
            plt.figure()
            plt.plot(xticks, trainacc_list, 'r-.s', label='Train Accuracy') 
            plt.plot(xticks, testacc_list, 'g-o', label='Test Accuracy')  
            plt.legend()
            plt.show()

    logging.critical("Training Finished!")        
