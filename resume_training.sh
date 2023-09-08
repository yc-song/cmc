#!/bin/bash

#SBATCH --job-name=extend
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=200000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
python main_reranker.py --epochs=20 --bert_lr=1e-06 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --model=./dual_encoder_zeshel.ckpt --type_cands=fixed_negative --token_type --eval_batch_size=8 --num_eval_cands=256 --training_one_epoch
