#!/bin/bash

#SBATCH --job-name=sum_max
#SBATCH --nodes=1
#SBATCH --time=0-72:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_output/%j.out

python main_reranker.py --epochs=20 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.01 --inputmark --model=./biencoder_wiki_large.bin --type_cands=mixed_negative --token_type --eval_batch_size=8 --blink --identity_bert --fixed_initial_weight --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_anncur_top1024_blink_retrieved --type_bert=large --B=2 --gradient_accumulation_steps=4 --distill_training --alpha=0.2 --gpus=0