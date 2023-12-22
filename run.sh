#!/bin/bash

#SBATCH --job-name=cocondenser
#SBATCH --nodes=1
#SBATCH --time=0-72:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_output/%j.out
python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.1 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=8 --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_top1024 --type_bert=base --B=2 --gradient_accumulation_steps=4 --gpus=3 --model=./cocondenser.bin --cocondenser 
# python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=8 --identity_bert --fixed_initial_weight --save_dir=./models --distill_training --alpha=0.2 --num_layers=2 --num_heads=4 --cands_dir=./data/cands_anncur_top1024 --type_bert=large --B=2 --gradient_accumulation_steps=8 --gpus=1,3
# python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=8 --identity_bert --fixed_initial_weight --save_dir=./models --distill_training --alpha=0.2 --num_layers=2 --num_heads=4 --cands_dir=./data/cands_anncur_top1024 --type_bert=base --B=4 --gradient_accumulation_steps=4 --gpus=2
# python main_retriever.py --data_dir ./data --B 4 --gradient_accumulation_steps 1 --logging_steps 1000 --k 64 --epochs 4 --lr 0.00001 --num_cands 64 --type_cands mixed_negative --cands_ratio 0.5 --gpus 2,3 --type_model dual --num_mention_vecs 128 --num_entity_vecs 128 --type_model dual --num_mention_vecs 128 --num_entity_vecs 128  --type_bert=large

# python main_reranker.py --epochs=20 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.01 --inputmark --model=./biencoder_wiki_large.bin --type_cands=mixed_negative --token_type --eval_batch_size=8 --blink --identity_bert --fixed_initial_weight --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_anncur_top1024_blink_retrieved --type_bert=large --B=2 --gradient_accumulation_steps=4 --distill_training --alpha=0.2 --gpus=0