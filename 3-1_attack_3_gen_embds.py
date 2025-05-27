import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import random
import gc
import logging
    
from utils.running_utils import get_gpu_device, checkCreateFolder, resetRandomStates, print_gpu_memory, initlogging, readConfigFile
from utils.llm_utils import getDictsizeAndEOSID, loadModel, loadTokenizer, getmodelname

########################################
# DONOT move these two functions, cause they need global variable 'activation'
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

def generate_shuffled_batches(K, m, s, randomseed=0):
    """
    Args:
        K (int): The range of tokens (0 to K-1).
        m (int): The number of repetitions for each token.
        s (int): The sequence size for splitting.

    Returns:
        torch.Tensor: The concatenated tensor of shuffled batches.
    """
    resetRandomStates(randomseed)
    sequence = torch.arange(K).repeat(m)

    # Shuffle the sequence
    shuffled_sequence = sequence[torch.randperm(sequence.size(0))]

    # Split the sequence into batches of sequence s
    splittensors = torch.split(shuffled_sequence, s, dim=0)
    if len(splittensors[0]) != len(splittensors[-1]):
        splittensors = splittensors[:-1]
    splittensors = [x.unsqueeze(0) for x in splittensors]
    
    attentiontensors = [torch.ones_like(x) for x in splittensors]
    tokentypeidtensors = [torch.zeros_like(x) for x in splittensors]

    return splittensors, attentiontensors, tokentypeidtensors


# ---------- the main function ----------
device = get_gpu_device()

parameters = readConfigFile('config.ini')
LLM_outputs_folder = parameters["ActivationStorePath"]
modeldir = parameters["LLM"]
datasetdir = parameters["TargetDataset_3"]
layer = parameters["AttackLayer"]
logfilename = initlogging(parameters["LogPath"]) 

fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{datasetdir}{os.sep}layer{layer}"
checkCreateFolder(fulllayerfolder)
model = loadModel(modeldir).to(device)

# register forward hooks on the layers of choice
h1, h2 = registerHooks(model, layer, modeldir)
########################################

batchszDICT = {"Wikitext2": 64, "MidjourneyPrompts": 128, "SQuAD2.0": 128, "PrivatePrompts": 128}
batchsize = batchszDICT[datasetdir]    
DICTSIZE, EOSID = getDictsizeAndEOSID(modeldir)

logging.critical(f"Processing layer [{layer}] for LLM [{modeldir}], dataset [{datasetdir}], and batchsize [{batchsize}]")

if parameters["RunningSamples_3"] <= 0:
    K = DICTSIZE # token range to generate
else:
    K = parameters["RunningSamples_3"]
m = 64 # repetitions for each token
s = 256 # sequence size
splittensors, attentiontensors, tokentypeidtensors = generate_shuffled_batches(K, m, s, randomseed=0)

tbar = tqdm(range(0, len(splittensors), batchsize), miniters=20)
logging.critical(f"Batchsize is set to [{batchsize}].")
logging.critical(f"Total [{K}] tokens repeating [{m}] times to augment,  [{len(tbar)}] files to generate.")
endidx = 0

for i, start_id in enumerate(tbar):
    if start_id + batchsize < len(splittensors):
        end_id = start_id+batchsize
    else:
        end_id = len(splittensors)

    tbar.set_description(f"batch {i}: {start_id} - {end_id}")
    
    filename = f"{modeldir}-{datasetdir}-layer{layer}-attack3-Embedding-Outputs-batch{i}.lst"
    savepath = os.path.join(fulllayerfolder, filename)
    endidx = i
    
    encoded_input = {}
    encoded_input['input_ids'] = torch.cat(splittensors[start_id:end_id]).clone().to(device)
    encoded_input['attention_mask'] = torch.cat(attentiontensors[start_id:end_id]).clone().to(device)
    assert len(encoded_input['input_ids']) == len(encoded_input['attention_mask']) == end_id - start_id, f"{len(encoded_input['input_ids'])} == {len(encoded_input['attention_mask'])} == {end_id - start_id}"
    if modeldir == "Bert":
        encoded_input['token_type_ids'] = torch.cat(tokentypeidtensors[start_id:end_id]).clone().to(device)
        
    '''For gpt2:
        inputid: torch.Size([b, 12])
        activation["token_embeddings"][0]: torch.Size([b, 12, 1600])
        activation["decoderlayer0"][0][0]: torch.Size([b, 12, 1600])
    '''
    activation = {"token_embeddings":[], f"decoderlayer{layer}":[]}
    
    with torch.no_grad():
        output = model(**encoded_input)
    
    activation["token_embeddings"] = activation["token_embeddings"][0].cpu()
    activation[f"decoderlayer{layer}"] = activation[f"decoderlayer{layer}"][0][0].cpu()
    
    torch.save((encoded_input["input_ids"].cpu(), activation), savepath)
    
    # Clear memory
    del output, encoded_input
    gc.collect()
    torch.cuda.empty_cache()
print_gpu_memory(device)

########### Merge Generated Files ###########

tt_batch = endidx + 1    
token_ids = []
token_embeddings = []
layer_outputs = []
for n_batch in tqdm(range(tt_batch), miniters=50):
    filename = f"{modeldir}-{datasetdir}-layer{layer}-attack3-Embedding-Outputs-batch{n_batch}.lst"
    loadpath = os.path.join(fulllayerfolder, filename)
    '''
        inputs:                             size (batchsize, sequencelen)
        activation["token_embeddings"]:     size (batchsize, sequencelen, 2560)
        activation[f"decoderlayer{layer}"]: size (batchsize, sequencelen, 2560)
    '''
    inputs, activation = torch.load(loadpath, map_location=torch.device('cpu'), weights_only=False)
    assert inputs.size(1) == activation["token_embeddings"].size(1), "squence lengthes do not match"
    token_ids.append(inputs)
    token_embeddings.append(activation["token_embeddings"])
    layer_outputs.append(activation[f"decoderlayer{layer}"])
    if n_batch % 70 == 0:
        logging.critical(f"Load file {loadpath}")
resetRandomStates(0)
combined = list(zip(token_ids, token_embeddings, layer_outputs))
# Shuffle the combined list
random.shuffle(combined)
# Unzip the shuffled list back into the original lists
id_shuffled, embd_shuffled, out_shuffled = zip(*combined)

# Convert back to lists (since zip returns tuples)
token_ids = list(id_shuffled)
token_embeddings = list(embd_shuffled)
layer_outputs = list(out_shuffled)


train_data = (token_ids, token_embeddings, layer_outputs)

savetrainfile = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}-layer{layer}-attack3-randomdata")
logging.critical(f"Save files to\n {savetrainfile} \n")
torch.save(train_data, savetrainfile)
logging.critical("Merging files finished!\n")

########### Delete Generated Files ###########

for n_batch in tqdm(range(tt_batch), miniters=50):
    filename = f"{modeldir}-{datasetdir}-layer{layer}-attack3-Embedding-Outputs-batch{n_batch}.lst"
    loadpath = os.path.join(fulllayerfolder, filename)
    os.remove(loadpath)
logging.critical("Cleaning files finished!\n")

logging.critical(f"Logs were written to {logfilename}") 
logging.critical("All finished!")

