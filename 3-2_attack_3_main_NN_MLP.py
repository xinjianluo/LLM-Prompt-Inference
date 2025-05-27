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
    
from utils.running_utils import get_gpu_device, checkCreateFolder, resetRandomStates, progressWrapper, print_gpu_memory, readConfigFile, initlogging
from utils.llm_utils import loadModel, loadTokenizer, getmodelname, getDictsizeAndEOSID
from utils.dataset_utils import WordbyWordDataset, readDataset, filterDatasetFreqs, readRandomDatasetforAdv3, SemiSupervisedDataset
from utils.attack_utils import MLPRegressor, MLPClassHead

def tester(model, testloader, loss_fn, device, mode="reg"):
    '''   
        model: reg or cla
    '''
    model.eval()
    accloss = 0.0
    accacc = 0.0 
    accloss_base = 0
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device) 
            y = y.to(device) 
            
            yhat = model(x)
            loss = loss_fn(yhat, y)
            if mode == "cla":
                accacc += ((yhat.argmax(dim=1))==y).sum()
            accloss += loss.detach().cpu()
            accloss_base += x.shape[0]

    return accloss / accloss_base, accacc / accloss_base

def trainer(model, optimizer, tdataset, device, epochs=1, test_interval=1, mode="reg", loss_graph=False):
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
        
    logging.critical(f"In trainer(), model={model.__class__.__name__}, tdataset={tdataset.__class__.__name__}, device={device}, epochs={epochs}, test_interval={test_interval}, mode={mode}")
    train_set, test_set = torch.utils.data.random_split(tdataset, [0.99, 0.01])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)  # small batch like 128 and 256 can help improve performance
    testloader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=True)
    
    model.train()
    start_time = time.time() 
    for epoch in (range(epochs)):
        accloss = 0.0
        accacc = 0.0 
        accloss_base = 0.0
        for x, y in trainloader:     
            optimizer.zero_grad()
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
            
        if (epoch+1 % test_interval == 0) or epoch == epochs - 1:
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
            logging.critical("\n")

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

    # print("Training Finished!")
    
def perLogitsEntropy(logits):
    # logits with size (N) before softmax
    # Convert logits to probabilities using softmax
    probs = torch.nn.functional.softmax(logits, dim=0)
    # Compute the entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-9))
    return entropy

def batchEntropy(dists, normalize=True):
    if normalize:
        normaldists = dists.mean() - dists
        # normaldists = (dists.mean() - dists) / dists.std()
        # normaldists = dists[:, -1].reshape(-1, 1) - dists
        # normaldists = normaldists / normaldists.mean()
    else:
        normaldists = dists
    distentropy = torch.zeros(len(dists))
    for i, ndist in enumerate(normaldists):
        distentropy[i] = perLogitsEntropy(ndist)
    return distentropy
    
def compACC(point_labels, point_labels_gt):
    acc = (point_labels_gt == point_labels).sum() / len(point_labels)
    return acc.item()
    
def NNAssignLabels(center_labels, center_tensors, point_tensors, parameters):
    pnorm = 1
    point_labels = -torch.ones(point_tensors.shape[0], dtype=torch.long)  # -1 means unassigned
    
    unique_labels = torch.unique(center_labels)
    num_clusters = len(unique_labels)
    
    cluster_masks = torch.stack([center_labels == label for label in unique_labels])
    index2labelDICT = {i:unique_labels[i].item() for i in range(len(unique_labels))}
    label2indexDICT = {unique_labels[i].item():i for i in range(len(unique_labels))}
    
    logging.critical(f"center_tensors.shape: {center_tensors.shape}")
    logging.critical(f"point_tensors.shape: {point_tensors.shape}")
    
    topk = 5
    distances = torch.empty(point_tensors.shape[0], num_clusters, device=point_tensors.device)
    for cluster_idx, mask in enumerate(cluster_masks):
        # Select points belonging to the current cluster
        cluster_points = center_tensors[mask]
        # Compute pairwise distances and average them
        distances[:, cluster_idx] = torch.cdist(point_tensors, cluster_points, p=pnorm).mean(dim=1)
    
    enthres = 0.1  # v1: 0.001
    print_iterval = 100
    max_iterations = 10000
    entthreshold = parameters['DistThres']
    logging.critical(f"In Nearest Neighbor Search, the distance entropy threshold is set to [{entthreshold}]")
    for iteration in (range(max_iterations)):
        if enthres > entthreshold:    # v1: 0.05
            break
        mindists, minindices = distances.topk(topk, dim=1, largest=False)
        minindices = minindices.cpu()
        minlabels = torch.tensor([index2labelDICT[ind.item()] for ind in minindices[:, 0]])
        
        dist_entropy = batchEntropy(mindists, normalize=True).cpu()
        # adaptively choose a threshold
        # entthreshold = dist_entropy.median()*1.2

        # Find points where entropy is below the threshold and they are unassigned
        valid_mask = (dist_entropy < enthres) & (point_labels == -1)
        # logging.critical(f"In iteration {iteration}, got {valid_mask.sum()} points to assign label.")
        if not valid_mask.any():
            enthres += 0.1 # v1: 0.001
            continue
            # logging.critical("Find no valid points to assign labels. Break.")
            # break  # Stop if no new points are valid
        # Assign labels to valid points (use the top-1 closest center)
        point_labels[valid_mask] = minlabels[valid_mask]
    
        # Update distances
        mod_center_labels = point_labels[valid_mask].unique()
        # logging.critical(f"Got {len(mod_center_labels)} labels to update")
        for modlabel in mod_center_labels:
            labelindex = label2indexDICT[modlabel.item()]
            curcenter = center_tensors[cluster_masks[labelindex]]
            curcenteraug = torch.cat([curcenter, point_tensors[point_labels == modlabel.item()]], dim=0)
            assert curcenteraug.shape[1] == point_tensors.shape[1], f"{curcenteraug.shape[1]} != {point_tensors.shape[1]}"
            distidx = labelindex
            moddistances = torch.cdist(point_tensors, curcenteraug, p=pnorm, compute_mode='donot_use_mm_for_euclid_dist').mean(dim=1)
            assert len(moddistances) == len(point_tensors), f"{len(moddistances)} != {len(point_tensors)}"
            distances[:, distidx] = moddistances
        if iteration % print_iterval == 0:
            logging.critical(f"In iteration {iteration}, assigned labels percentage {1 - (point_labels == -1).sum() / len(point_labels)}")
            # print_gpu_memory(device)
    
    logging.critical(f"\nIn the final iteration {iteration}, {1 - (point_labels == -1).sum() / len(point_labels)} percentage points are assigned labels")
    return point_labels.cpu() 
    
# ------------------------------------------------------------------
if __name__ == "__main__":
    parameters = readConfigFile('config.ini')
    LLM_outputs_folder = parameters["ActivationStorePath"]
    modeldir_shadow = parameters["LLM"]
    datasetdir_shadow = parameters['TargetDataset_3']
    layer = parameters["AttackLayer"]
    querybudgets = parameters['QueryBudget']
    logfilename = initlogging(parameters["LogPath"]) 

    fulllayerfolder = f"{LLM_outputs_folder}{os.sep}{modeldir_shadow}{os.sep}{datasetdir_shadow}{os.sep}layer{layer}"
        
    logging.critical(f"Processing layer [{layer}] in LLM [{modeldir_shadow}] for dataset [{datasetdir_shadow}] with querybudgets [{querybudgets}]")
    
    device = get_gpu_device()
    
    DICTSIZE_sdw, EOSID_sdw = getDictsizeAndEOSID(modeldir_shadow) 
    logging.critical(f"DICTSIZE is set to [{DICTSIZE_sdw}]; EOSID is set to [{EOSID_sdw}]\n") 
    
    # Read training data 
    trainwwset = readRandomDatasetforAdv3(layer, modeldir_shadow, datasetdir_shadow, EOSID_sdw, fulllayerfolder, promptstr="Randomly Generated Embeddings")
    trainwwset = filterDatasetFreqs(trainwwset, FREQ=querybudgets)
    logging.critical(f"After filtering, random dataset size is {len(trainwwset)}\n")

    testwwset = readDataset(layer, modeldir_shadow, datasetdir_shadow, EOSID_sdw, fulllayerfolder, train=False, promptstr="Real Test Dataset")

    # Nearest neighbor search
    resetRandomStates(0)
    gt_len = len(trainwwset) if len(trainwwset) < 30000 else 30000
    randidx = torch.tensor(list(range(gt_len)))
    center_labels = trainwwset.catids
    center_tensors = trainwwset.catlayer0s.to(device)
    point_tensors = testwwset.catlayer0s[randidx].clone().to(device)

    point_labels = NNAssignLabels(center_labels, center_tensors, point_tensors, parameters)

    point_labels_gt = testwwset.catids[randidx].clone()
    logging.critical(f"After NNAssignLabels(), total acc is: {compACC(point_labels, point_labels_gt)}\n")

    # Semi-supervised MLP training
    label_save_path = os.path.join(fulllayerfolder, f"attack3-{modeldir_shadow}-{datasetdir_shadow}-layer{layer}-budgets{querybudgets}-point_labels-point_labels_gt-After-MLPSearch.ts") 
    model_save_path = os.path.join(fulllayerfolder, f"attack3-{modeldir_shadow}-{datasetdir_shadow}-layer{layer}-budgets{querybudgets}-SSL-Classifier.mdl")

    epochs = 6
    test_interval = 4

    save_interval = 10

    # train model
    threshold = 0.4
    logitsthres = parameters['LogitsThres']
    logging.critical(f"In MLP training, logits entropy threshold is set to [{logitsthres}]")
    for i in range(12):
        clamodel = MLPClassHead(point_tensors.shape[1], point_tensors.shape[1], DICTSIZE_sdw, None).to(device)
        optimizer = torch.optim.Adam(clamodel.parameters(), weight_decay=0.01)
        
        wwset = SemiSupervisedDataset(center_tensors, center_labels, point_tensors, point_labels)
        trainer(clamodel, optimizer, wwset, device, epochs=epochs, test_interval=test_interval, mode="cla", loss_graph=False) 

        # reassign labels
        mask_unasg = point_labels == -1
        with torch.no_grad():
            pred_logits = clamodel(point_tensors[mask_unasg].to(device))
        pred_entropy = batchEntropy(pred_logits.cpu(), normalize=False)
        pred_labels = pred_logits.argmax(dim=1).cpu()
        mask_entropy = pred_entropy < threshold

        new_assign_labels = -torch.ones(len(mask_entropy), dtype=torch.long)  # -1 means unassigned
        new_assign_labels[mask_entropy] = pred_labels[mask_entropy]
        point_labels[mask_unasg] = new_assign_labels

        logging.critical("\n----------------------------------------------")
        logging.critical(f"In MLP iteration {i}, {mask_entropy.sum().item()} points are assigned labels; Remain {(point_labels == -1).sum().item()} unlabeled points.")
        logging.critical(f"In MLP iteration {i}, total acc is: {compACC(point_labels, point_labels_gt)}")
        logging.critical("----------------------------------------------\n")
        if i % save_interval == 0:
            with torch.no_grad():
                model_labels = clamodel(point_tensors).argmax(dim=1).cpu()

            torch.save((point_labels.cpu().clone(), point_labels_gt.cpu().clone(), model_labels.cpu().clone()), label_save_path)
            # torch.save(clamodel.state_dict(), model_save_path)
        if mask_entropy.sum().item() <= 20 and threshold < logitsthres:
            threshold += 0.15
            logging.critical(f"In MLP iteration {i}, increase temporary threshold to {threshold}")
            
        if mask_entropy.sum().item() <= 5:
            break
    
    logging.critical("\n----------------------------------------------")
    logging.critical(f"After MLP iterations, {1 - (point_labels == -1).sum() / len(point_labels)} percentage points are assigned labels; total acc is: {compACC(point_labels, point_labels_gt)}")
    logging.critical("----------------------------------------------\n")

    with torch.no_grad():
        model_labels = clamodel(point_tensors).argmax(dim=1).cpu()

    torch.save((point_labels.cpu().clone(), point_labels_gt.cpu().clone(), model_labels.cpu().clone()), label_save_path)
    logging.critical(f"Reconstructed and ground truth labels are saved to {label_save_path}")

    # # train a good model on available data-label pairs
    # clamodel = MLPClassHead(point_tensors.shape[1], point_tensors.shape[1], DICTSIZE_sdw, None).to(device)
    # optimizer = torch.optim.Adam(clamodel.parameters(), weight_decay=0.001)

    # wwset = SemiSupervisedDataset(center_tensors, center_labels, point_tensors, point_labels)
    # trainer(clamodel, optimizer, wwset, device, epochs=12, test_interval=4, mode="cla", loss_graph=False)
    # torch.save(clamodel.state_dict(), model_save_path)
    
    logging.critical(f"Logs were written to {logfilename}") 
    logging.critical("All finished!")


