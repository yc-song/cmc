#!/bin/bash
#SBATCH --job-name=czituqt9
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
python main_reranker.py --epochs=10 --lr=0.00001 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --type_cands=fixed_negative --identity_bert --resume_training --run_id=czituqt9