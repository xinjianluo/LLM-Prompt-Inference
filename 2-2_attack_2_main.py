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

from utils.llm_utils import getDictsizeAndEOSID
from utils.running_utils import get_gpu_device, progressWrapper, initlogging, resetRandomStates, readConfigFile, print_gpu_memory
from utils.attack_utils import tester, trainer, MLPRegressor, MLPClassHead
from utils.dataset_utils import WordbyWordDataset, AugmentedWordbyWordDataset  

def count_files_with_prefix(folder_path, prefix):
    return sum(
        1 for filename in os.listdir(folder_path)
        if filename.startswith(prefix) and os.path.isfile(os.path.join(folder_path, filename))
    )
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    
    parameters = readConfigFile('config.ini')
    LLM_outputs_folder = parameters["ActivationStorePath"]
    modeldir = parameters["LLM"]
    datasetdir = parameters['BaseDataset']
    targetdatasetdir = parameters['TargetDataset_2']
    layer = parameters["AttackLayer"]
    logfilename = initlogging(parameters["LogPath"]) 

    fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{datasetdir}{os.sep}layer{layer}"
    fulltestfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{targetdatasetdir}{os.sep}layer{layer}"

    augpathstr = f"-aug{targetdatasetdir}" 
    
    for mode in ("reg", "cla", ):    # two-stage training
        scratch = True if mode == "reg" else False
        n_aug_files = count_files_with_prefix(fulllayerfolder, f"{modeldir}-layer{layer}-Embedding-Outputs-base{datasetdir}-target{targetdatasetdir}-batch")

        logging.critical(f"Processing mode = [{mode}] for layer = [{layer}] in LLM = [{modeldir}]; base dataset = [{datasetdir}], target dataset = [{targetdatasetdir}] with [{n_aug_files}] augmented files")

        print()

        device = get_gpu_device() 
        DICTSIZE, EOSID = getDictsizeAndEOSID(modeldir) 
            
        logging.critical(f"DICTSIZE is set to {DICTSIZE}; EOSID is set to {EOSID}")    
        print()
        # Read training data 
        traindatapath = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}-layer{layer}-train-data")  
        logging.critical(f"Reading base train data from {traindatapath}")
        wwset = AugmentedWordbyWordDataset(modeldir, datasetdir, targetdatasetdir, EOSID, n_aug_files, layer, LLM_outputs_folder)
        
        testdatapath = os.path.join(fulltestfolder, f"{modeldir}-{targetdatasetdir}-layer{layer}-test-data")  
        logging.critical(f"\nReading test data from {testdatapath}")
        wwtestset = WordbyWordDataset(testdatapath, EOSID)
            
        if mode == "cla":
            if scratch:
                pretrainedModelPath = None 
            else:
                pretrainedModelPath = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}_Train-MLPRegressor-Scratch-reg-layer{layer}{augpathstr}.mdl")
                
            mlpModel = MLPClassHead(wwset.getmodeldim(), wwset.getmodeldim(), DICTSIZE, pretrainedModelPath).to(device)
        elif mode == "reg":
            assert scratch, "You should train MLPRegressor from scratch"
            mlpModel = MLPRegressor(wwset.getmodeldim(), wwset.getmodeldim()).to(device)
        else:
            raise ValueError(f'Unsupported training mode {mode}') 

        if scratch:  
            scratchstr = "Scratch"  
        else:
            scratchstr = "Finetune"
        epochs = 4
        test_interval=1
        
        suffix = f"-{mlpModel.__class__.__name__}-{scratchstr}-{mode}-layer{layer}{augpathstr}"    
        logging.critical(f"Model name suffix is {suffix}\n")

        optimizer = torch.optim.Adam(mlpModel.parameters())

        trainer(mlpModel, optimizer, wwset, wwtestset, device, epochs=epochs, test_interval=test_interval, mode=mode, loss_graph=False)       
        
        # Save trained model
        modelsavepath = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}_Train{suffix}.mdl")
        logging.critical(f"Save model to {modelsavepath}")
        torch.save(mlpModel.state_dict(), modelsavepath)
        print()
        
        del wwset, wwtestset, mlpModel
        gc.collect()
    logging.critical(f"Writing log to {logfilename}")
    logging.critical("All finished!")
