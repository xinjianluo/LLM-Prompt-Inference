# Instructions to Run the Baseline vec2text


We kindly note the following:

> The attached vec2text code (`vec2text.tar.gz`) is considerably different from the original code (https://github.com/vec2text/vec2text), as we have made a number of modifications to make it compatible with our experimental settings, as well as with the conda environment `prompt-inference-env` created for our code. Nevertheless, *the steps for training vec2text models remain similar to those released in the original repository. Please follow the instructions below step by step to ensure a smooth vec2text testing experience*.


-----------------------------



## Preliminary

Download the attached `vec2text.tar.gz` and move it to the directory `LLM-Prompt-Inference` (i.e, the root directory of our artifact). Then, decompress it:

```bash
tar xvf vec2text.tar.gz
cd vec2text/
```

Next, install necessary python packages:
```bash
conda activate prompt-inference-env
pip install vec2text
conda install -c conda-forge huggingface_hub
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

## Train the vec2text Models
First, train the zero-step model for an LLM:
```bash
python run.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 48 --dataset_name one_million_instructions --embedder_model_name gpt2-large  --num_repeat_tokens 4 --embedder_no_grad True --num_train_epochs 80 --max_eval_samples 50 --use_less_data 20000 --eval_steps 10000 --warmup_steps 10000 --bf16=1 --use_wandb=0 --use_frozen_embeddings_as_input True --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001  --save_steps 2000 --embeddings_from_layer_n 1 --output_dir ./vec2text_cache/saves/gpt2-large_layer_1 --overwrite_output_dir --experiment inversion
```

- Choose `--embedder_model_name` from `{gpt2-large, bert, phi3, llama3.2}`
- Choose `--embeddings_from_layer_n` from: GPT2 `{1, 10, 19, 28, 36}`, Llama3 `{1, 5, 9, 13, 16}`, Phi3 `{1, 9, 17, 25, 32}`, Bert `{1, 7, 13, 19, 24}`.
- Make `--output_dir` the format `{embedder_model_name}_layer_{embeddings_from_layer_n}`.

After the model is trained, add its path (i.e., the `--output_dir` parameter `./vec2text_cache/saves/gpt2-large_layer_1`) to the file `vec2text/vec2text/aliases.py` along with the key `gpt_inst_msl48_layer1_80epoch` (Note: *the key is self-defined but try to make it consistent with the training parameters*).

Then run the following command to train the corrector:


```bash
python run.py --per_device_train_batch_size 128 --per_device_eval_batch_size 128 --max_seq_length 48 --dataset_name one_million_instructions --embedder_model_name gpt2-large  --num_repeat_tokens 4 --embedder_no_grad True --num_train_epochs 80 --max_eval_samples 50 --use_less_data 20000 --eval_steps 10000 --warmup_steps 10000 --bf16=1 --use_wandb=0 --use_frozen_embeddings_as_input True --lr_scheduler_type constant_with_warmup --exp_group_name oct-gtr --learning_rate 0.001  --save_steps 2000 --embeddings_from_layer_n 1 --corrector_model_alias gpt_inst_msl48_layer1_80epoch --output_dir ./vec2text_cache/saves/gpt2-large_layer_1_corrector --overwrite_output_dir --experiment corrector
```

- Make `--embedder_model_name` and `--embeddings_from_layer_n` the same as in the last command.
- Make `--corrector_model_alias` the same as the key `gpt_inst_msl48_layer1_80epoch` written in file `./vec2text/aliases.py`.
- Make `--output_dir` the format `{embedder_model_name}_layer_{embeddings_from_layer_n}_corrector`.

## Evaluate Model Performance
After the corrector is trained, run the following command to evaluate its performance:

```bash
python vec2text_test.py --embedder_model_name gpt2-large --embeddings_from_layer_n 1
```

