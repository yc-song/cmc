#!/bin/bash
#SBATCH --job-name=full
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --model_top=/home/jylab_share/jongsong/hard-nce-el/models/BLINK/epoch_27_0 --type_cands=fixed_negative --training_one_epoch --run_id=f4kopkqp