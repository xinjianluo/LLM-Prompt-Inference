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

def readConfigFile(configfile):
    parameters = {}
    # read parameters from config file
    config = configparser.ConfigParser()
    config.read(configfile)

    p_default = config['DEFAULT']
    p_attack1 = config['Attack1']
    p_attack2 = config['Attack2']
    p_attack3 = config['Attack3']
    p_test = config['AttackTest']

    # DEFAULT Section
    parameters['LLM'] = p_default['LLM']
    parameters['AttackLayer'] = p_default.getint('AttackLayer')
    parameters['ActivationStorePath'] = p_default['ActivationStorePath']
    checkCreateFolder(p_default['ActivationStorePath'])
    checkCreateFolder(p_default['LogPath'])
    parameters['LogPath'] = p_default['LogPath'] + os.sep + os.path.splitext(os.path.basename(sys.modules['__main__'].__file__))[0]
    
    # Attack1 Section
    parameters['Dataset'] = p_attack1['Dataset']
    parameters['RunningSamples_1'] = p_attack1.getint('RunningSamples_1')
    
    # Attack2 Section
    parameters['RunningSamples_2'] = p_attack2.getint('RunningSamples_2')
    parameters['BaseDataset'] = p_attack2['BaseDataset']
    parameters['TargetDataset_2'] = p_attack2['TargetDataset_2']
    parameters['AugDelta'] = p_attack2.getint('AugDelta')
    
    # Attack3 Section
    parameters['RunningSamples_3'] = p_attack3.getint('RunningSamples_3')
    parameters['TargetDataset_3'] = p_attack3['TargetDataset_3']
    parameters['QueryBudget'] = p_attack3.getint('QueryBudget')
    parameters['DistThres'] = p_attack3.getfloat('DistThres')
    parameters['LogitsThres'] = p_attack3.getfloat('LogitsThres')
    parameters['BeamWidth'] = p_attack3.getint('BeamWidth')
    parameters['ShadowLLM'] = p_attack3['ShadowLLM']
    
    # AttackTest Section
    parameters['TestPrompt'] = p_test['TestPrompt']
    return parameters
    
def initlogging(logfilename):
    def getTimeStamp():
        from datetime import datetime
        return datetime.now().strftime("_%Y%m%d_%H_%M_%S")
        
    import logging
    logfile = f"{logfilename}_{getTimeStamp()}.log"
    # debug, info, warning, error, critical
    # set up logging to file
    logging.shutdown()
    
    logger = logging.getLogger()
    logger.handlers = []
    
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=logfile,
                        filemode='w')
    
    # create console handler
    ch = logging.StreamHandler()
    
    # Sets the threshold for this handler. 
    # Logging messages which are less severe than this level will be ignored, i.e.,
    # logging messages with < critical levels will not be printed on screen
    # E.g., 
    # logging.info("This should be only in file") 
    # logging.critical("This shoud be in both file and console")
    ch.setLevel(logging.CRITICAL)
    
    # add formatter to ch
    ch.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(ch)  
    return logfile
    
def checkCreateFolder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.critical(f"Folder '{folder_path}' created.\n")
    else:
        logging.critical(f"Folder '{folder_path}' already exists.\n")
        
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dm %ds' % (m, s) if h==0 else '%dh %dm %ds' % (h, m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs)) 

def progressWrapper(start_timer, total_iter, current_iter): 
    """
        total_iter and current_iter should start from 1
    """
    incurrent_iter = current_iter  
    if current_iter == 0: 
        incurrent_iter = 1 
    assert total_iter > 0, f"total_iter must > 0, now is {total_iter}"
    percentage = (incurrent_iter*1.0) / (total_iter)
    time_str = timeSince(start_timer, percentage) 
    
    return "%s (%d / %d -> %d%%)" % (time_str, current_iter, total_iter, percentage * 100)
    
def get_gpu_device():
    n_gpus = torch.cuda.device_count()
    logging.critical(f"Total GPUS: {n_gpus}")
    if n_gpus <= 0:
        logging.critical("Decide to use CPU")
        return torch.device("cpu")
    idx = -1
    temfree = 0
    for i in range(n_gpus):
        cdevice = torch.device(f"cuda:{i}")
        free, total = torch.cuda.mem_get_info(cdevice)
        free = free * 1.0 / 1024 / 1024 / 1024
        total = total * 1.0 / 1024 / 1024 / 1024
        allocated = torch.torch.cuda.memory_allocated(cdevice)
        allocated = allocated * 1.0 / 1024 / 1024 / 1024
        if free > temfree:
            temfree = free 
            idx = i 
        logging.critical(f"For GPU {i}, MEMORY Info Free {free:.3f}, Total {total:.3f}, Allocated {allocated:.3f}")
    
    logging.critical(f"Decide to use GPU {idx}")
    return torch.device(f"cuda:{idx}")
    
def print_gpu_memory(device):
    if device.type == "cpu":
        logging.critical(f"For cpu device {device}, got nothing to print!")
        return device
    free, total = torch.cuda.mem_get_info(device)
    free = free * 1.0 / 1024 / 1024 / 1024
    total = total * 1.0 / 1024 / 1024 / 1024
    allocated = torch.torch.cuda.memory_allocated(device)
    allocated = allocated * 1.0 / 1024 / 1024 / 1024
    
    logging.critical(f"For device {device}, MEMORY Info Free {free:.3f}, Total {total:.3f}, Allocated {allocated:.3f}")
    
    return device      

def resetRandomStates(manualseed=47):
    random.seed(manualseed)
    torch.manual_seed(manualseed)
    np.random.seed(manualseed)


    
  
