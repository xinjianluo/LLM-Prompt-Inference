import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
import logging
import random
from utils.running_utils import get_gpu_device, checkCreateFolder, resetRandomStates, print_gpu_memory, readConfigFile, initlogging
from utils.llm_utils import loadModel, loadTokenizer, getmodelname

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

def readTestPrompts(datasetdir, parameters):
    # load dataset
    if datasetdir == "SQuAD2.0":
        traindatapath = f".{os.sep}datasets{os.sep}{datasetdir}{os.sep}SQuAD2.0-train-QandA-Pairs.lst"
    elif datasetdir == "Wikitext2":
        traindatapath = f".{os.sep}datasets{os.sep}{datasetdir}{os.sep}Wikitext2-Merge-Articles-19771.lst"  # Sorted, from short to long
    elif datasetdir == "MidjourneyPrompts":
        traindatapath = f".{os.sep}datasets{os.sep}{datasetdir}{os.sep}MidjourneyPrompts-Prompt-Sorted-123642.lst"  # Sorted, from short to long
    elif datasetdir == "PrivatePrompts":
        traindatapath = f".{os.sep}datasets{os.sep}{datasetdir}{os.sep}ProvatePrompts-Length-Sorted-251270.lst"  # Sorted, from short to long    
    else: 
        raise ValueError(f'Unsupported datasetdir {datasetdir}')
    logging.critical(f"Load dataset from {traindatapath}")

    testprompts = torch.load(traindatapath, weights_only=False)
    if parameters["RunningSamples_1"] <= 0:
        return testprompts
    else:
        return testprompts[:parameters["RunningSamples_1"]]
        
# ---------- the main function ----------
parameters = readConfigFile('config.ini')
logfilename = initlogging(parameters["LogPath"]) 
device = get_gpu_device()
LLM_outputs_folder = parameters["ActivationStorePath"]
modeldir = parameters["LLM"]
datasetdir = parameters["Dataset"]
layer = parameters["AttackLayer"]

fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir}{os.sep}{datasetdir}{os.sep}layer{layer}"
checkCreateFolder(fulllayerfolder)
model = loadModel(modeldir).to(device)
tokenizer = loadTokenizer(modeldir)

# register forward hooks on the layers of choice
h1, h2 = registerHooks(model, layer, modeldir)
########################################

testprompts = readTestPrompts(datasetdir, parameters)
batchsize = 64
logging.critical(f"Processing layer [{layer}] for model [{modeldir}], dataset [{datasetdir}], and batchsize [{batchsize}]\n")
     
logging.critical(f"Test Prompts Size = {len(testprompts)}, batchsize = {batchsize}\n")

tbar = tqdm(range(0, len(testprompts), batchsize), miniters=20)

endidx = 0
for i, start_id in enumerate(tbar):
    # Some constraints incase the gpu memory is not big enough
    # if datasetdir == "MidjourneyPrompts" and batchsize == 128:
        # if i > 910:
            # break
    # elif datasetdir == "Wikitext2" and batchsize == 64:
        # if modeldir == "Bert" and i > 270:
            # break
        # if i > 305:
            # break
    if start_id + batchsize < len(testprompts):
        end_id = start_id + batchsize
    else:
        end_id = len(testprompts)

    if datasetdir == "SQuAD2.0":
        questions = [x[0] for x in testprompts[start_id:end_id]]  # for SQuAD2.0
    else:
        questions = [x for x in testprompts[start_id:end_id]]  # for Others


    tbar.set_description(f"batch {i}: {start_id}({len(questions[0])}) - {end_id}({len(questions[-1])})")
    
    filename = f"{datasetdir}-train-QandA-Embedding-Outputs-{modeldir}-layer{layer}-batch{i}.lst"
    savepath = os.path.join(fulllayerfolder, filename)
    endidx = i

    encoded_input = tokenizer(questions, return_tensors="pt", padding=True)
    encoded_input['input_ids'] = encoded_input['input_ids'].to(device)
    encoded_input['attention_mask'] = encoded_input['attention_mask'].to(device)
    if modeldir == "Bert":
        encoded_input['token_type_ids'] = encoded_input['token_type_ids'].to(device)
        
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
    torch.cuda.empty_cache()
print_gpu_memory(device)

########### Merge Generated Files ###########

tt_batch = endidx + 1    # change this
token_ids = []
token_embeddings = []
layer_outputs = []
for n_batch in tqdm(range(tt_batch), miniters=50):
    filename = f"{datasetdir}-train-QandA-Embedding-Outputs-{modeldir}-layer{layer}-batch{n_batch}.lst"
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
    if n_batch % 200 == 0:
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

split = int(tt_batch*0.8)

train_data = (token_ids[:split], token_embeddings[:split], layer_outputs[:split])
test_data = (token_ids[split:], token_embeddings[split:], layer_outputs[split:])

savetrainfile = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}-layer{layer}-train-data")
savetestfile = os.path.join(fulllayerfolder, f"{modeldir}-{datasetdir}-layer{layer}-test-data")
logging.critical(f"Save files to\n {savetrainfile} \nand\n {savetestfile} \n")
torch.save(train_data, savetrainfile)
torch.save(test_data, savetestfile)
logging.critical("Merging files finished!\n")

########### Delete Intermediate Files ###########

for n_batch in tqdm(range(tt_batch), miniters=50):
    filename = f"{datasetdir}-train-QandA-Embedding-Outputs-{modeldir}-layer{layer}-batch{n_batch}.lst"
    loadpath = os.path.join(fulllayerfolder, filename)
    os.remove(loadpath)
logging.critical("Cleaning files finished!\n")

logging.critical(f"Logs were written to {logfilename}")    
logging.critical("All finished!")





