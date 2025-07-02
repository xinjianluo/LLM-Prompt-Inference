# Instructions on Reproducing Figures 9 and 11


To fully reproduce Figure 9, `15` attack models (60 in total) are required for each of the four LLMs: Phi-3.5, Llama-3.2, GPT-2, and BERT. Similarly, reproducing Figure 11 requires `5` attack models (20 in total) per LLM. We summarize the attack model sizes in the following table:

LLM | Attack Model Size
------|------
Phi-3.5| 5.14G
Llama-3.2| 3.10G
GPT2 |1.07G
BERT |0.64G




In total, sharing all models requires approximately 199 GB of Google Drive space. Unfortunately, we may not have sufficient quota for this. As an alternative, we suggest that interested researchers run our code with specific configurations to reproduce the results in Figures 9 and 11. The code has been comprehensively tested and is expected to match the results reported in the paper. Detailed instructions are provided below.




## Train Attack Models in Figure 9
First, modify `config.ini`:

1. Set `Attack1` -> `RunningSamples_1` to `-1`.
2. Choose `DEFAULT` -> `LLM` from `{Gpt2, Llama3, Phi3, Bert}`.
3. Choose `DEFAULT` -> `AttackLayer` from: GPT2 `{0, 9, 18, 27, 35}`, Llama3 `{0, 4, 8, 12, 15}`, Phi3 `{0, 8, 16, 24, 31}`, Bert `{0, 6, 12, 18, 23}`.
4. Choose `Attack1` -> `Dataset` from `{SQuAD2.0, Wikitext2, PrivatePrompts, MidjourneyPrompts}`.

Then, run the following commands to train attack models:
```bash
python 1-1_gen_training_data.py
python 1-2_attack_1_main.py
```

## Train Attack Models in Figure 11
Similarly, modify `config.ini`:

1. Set `Attack2` -> `RunningSamples_2` to `-1`.
2. Choose `DEFAULT` -> `LLM` from `{Gpt2, Llama3, Phi3, Bert}`.
3. Choose `DEFAULT` -> `AttackLayer` from: GPT2 `{0, 9, 18, 27, 35}`, Llama3 `{0, 4, 8, 12, 15}`, Phi3 `{0, 8, 16, 24, 31}`, Bert `{0, 6, 12, 18, 23}`.
4. Choose `Attack2` -> `BaseDataset` from `{SQuAD2.0, Wikitext2, PrivatePrompts, MidjourneyPrompts}`.
5. Choose `Attack2` -> `TargetDataset_2` from `{SQuAD2.0, Wikitext2, PrivatePrompts, MidjourneyPrompts}`/`BaseDataset`.

Then, run the following commands to train attack models:
```bash
python 1-1_gen_training_data.py
python 2-1_attack_2_aug_gen.py
python 2-2_attack_2_main.py
```

## Evaluate Attack Performance
After training all attack models, run the following commands to evaluate the performance (may need to modify `Attack1` -> `Dataset`, `Attack2` -> `BaseDataset`, and `Attack2` -> `TargetDataset_2` to specify attack models):
```bash
python 4-1_attack_model_testing.py
```
