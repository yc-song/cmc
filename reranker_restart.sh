#!/bin/bash

#SBATCH --job-name=extend
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=100000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
file_contents=$(cat ./models/$1/$2/training_params.txt)
lr=$(echo "$file_contents" | grep -oP "lr=\K[\d.]+")
epochs=$(echo "$file_contents" | grep -oP "epochs=\K[\d.]+")
warmup_proportion=$(echo "$file_contents" | grep -oP "warmup_proportion=\K[\d.]+")

echo ${lr}
echo ${epochs}
echo ${warmup_proportion}

python main_reranker.py --model ./models/reranker --data ./data --B 2 --gradient_accumulation_steps 4 --num_workers 2 --warmup_proportion=${warmup_proportion} --epochs=${epochs} --gpus 0,1 --lr=${lr} --cands_dir=./data/cands_anncur --eval_method macro --type_model=$1 --type_bert base --inputmark --type_cands=fixed_negative --training_one_epoch --resume_training --run_id=$2
# python main_reranker.py --model ./models/reranker --data ./data --B 2 --gradient_accumulation_steps 4 --num_workers 2 --warmup_proportion 0.2 --epochs 50 --gpus 0,1 --lr 0.0001626752188247764 --cands_dir=./data/cands_anncur --eval_method macro --type_model extend_multi_dot --type_bert base --inputmark --type_cands=hard_negative --training_one_epoch --resume_training --run_id=2gpzhjsj