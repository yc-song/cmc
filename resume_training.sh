#!/bin/bash

#SBATCH --job-name=extend
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=200000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
python main_retriever.py --model ./models/model_dual.pt --data_dir data --B 1 --gradient_accumulation_steps 2 --logging_steps 1000 --k 64 --epochs 4 --lr 0.00001 --num_cands 64 --type_cands mixed_negative --cands_ratio 0.5 --gpus 0,1 --type_model full --num_mention_vecs 128 --num_entity_vecs 128 --entity_bsz 512 --mention_bsz 512 --cands_dir=data/cands_dual --eval_method=micro --retriever_path=/home/jylab_share/jongsong/hard-nce-el/models/dual/q0z280i9/pytorch_model.bin --retriever_type=dual --resume_training --run_id=x3qtjw4n