#!/bin/bash

#SBATCH --job-name=save
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=120000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
python save_candidates.py --model /home/jylab_share/jongsong/hard-nce-el/models/dual/otf2fqy6/pytorch_model.bin --data_dir ./data --pre_model Bert --type_model dual --num_mention_vecs 1 --num_entity_vecs 1 --entity_bsz 1024  --mention_bsz 200 --store_en_hiddens --en_hidden_path ./data/en_hidden_dual  --num_cands 200 --cands_dir ./data/cands_dual_top200 --gpus 0,1
