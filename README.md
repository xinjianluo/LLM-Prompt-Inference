# Prompt Inference Attack on Distributed Large Language Model Inference Frameworks

## Overview
This repository contains the official implementation of the experiments described in the paper:

> *Prompt Inference Attack on Distributed Large Language Model Inference Frameworks*


### Remarks
1. The original experiments were conducted on a private swarm of [Petals](https://github.com/bigscience-workshop/petals), deployed on a high-performance server with **4× NVIDIA A100-SXM4-40GB GPUs**.
2. This repository provides a **single-machine** version of the code that runs on a **single GPU**. The results should align with those obtained using the Petals framework.
3. All code was tested on **A100 GPUs (40GB memory)**. If you use a GPU with less memory, **out-of-memory** issues may occur.


## Getting Started
### Step 1: Install Dependencies
Create a new conda environment named `prompt-inference-env` (adjust `pytorch-cuda` version if necessary):

```bash
conda env create -f conda_env.yml
```

And activate it:
```bash
conda activate prompt-inference-env
```
    

### Step 2: Config Running Options
You can modify the LLM, attack layer, target dataset, and other hyperparameters in `config.ini`. The key names are self-explanatory.


### Step 3: Run Attack 1
First, generate the training data:
```bash
python 1-1_gen_training_data.py
```

Then, train an attack model:
```bash
python 1-2_attack_1_main.py
```

### Step 4: Run Attack 2
> In this demo, to reduce compute cost, we assume access to the token set of the target dataset.
Alternatively, you can choose to replace `tgtdataset.catids` in `2-1_attack_2_aug_gen.py` with the full LLM vocabulary, this will produce similar results but significantly increase runtime.


First, we generate the token set of the target dataset:
1. In `config.ini`, set `Attack1 -> Dataset = SQuAD2.0` (i.e., match `TargetDataset_2` in `Attack2`).
1. Generate the token set for the target dataset `SQuAD2.0` in Attack 2:
```bash
python 1-1_gen_training_data.py
```

Then, we perform Attack 2 as follows:
1. Generate synthetic training data:
```bash
python 2-1_attack_2_aug_gen.py
```

2. Train an attack model using the synthetic data:
```bash
python 2-2_attack_2_main.py
```

### Step 5: Run Attack 3
1. Generate anchor embeddings:
```bash
python 3-1_attack_3_gen_embds.py
```
2. Perform the first two reconstruction phases:
```bash
python 3-2_attack_3_main_NN_MLP.py
```
3. Perform the complementary beam search phase:
```bash
python 3-3_attack_3_beam_search.py
```

### Step 6: Test Attack Models
Before testing, you may replace the value of `TestPrompt` in `config.ini` with any prompt of your choice.

After that, run:
```bash
python 4-1_attack_model_testing.py
```

> **Tip**. Inaccurate prompt reconstructions are expected in this step, if you only train the attack model on a limited number of samples. Try to set `RunningSamples` to `-1` in `config.ini` for a full evaluation.

Or, you can also test our pretrained attack models:
```bash
python 4-2_attack_model_pretrained_testing.py
```

> **Tip**. The reconstruction performance may not be perfect, as the models are trained only on a small dataset, `Wikitext2`. However, for most prompts, the attack performance should align with the results reported in our paper, i.e., achieving over `90%` token-level reconstruction accuracy.


## Citation
If you use this codebase or build upon our work, please cite the following:

    @article{luo2025prompt,
        title={Prompt Inference Attack on Distributed Large Language Model Inference Frameworks},
        author={Luo, Xinjian and Yu, Ting and Xiao, Xiaokui},
        journal={arXiv preprint arXiv:2503.09291},
        year={2025}
    }

