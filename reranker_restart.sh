#!/bin/bash
#SBATCH --job-name=lambda_scheduler
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=cas_v100nv_8
#SBATCH --comment pytorch 
export PYTHONPATH=$PYTHONPATH:.
python main_reranker.py --epochs=20 --lr=0.00001 --type_model=extend_multi --warmup_proportion=0.1 --inputmark --type_cands=fixed_negative --lambda_scheduler
