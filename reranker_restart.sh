#!/bin/bash
#SBATCH --job-name=fa4j4smq
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=0-12:00:00
#SBATCH --mem=80000MB
#SBATCH --cpus-per-task=16
#SBATCH --partition=P1
# file_contents=$(cat ./models/extend_multi/smjwpt5i/training_params.txt)
#  file_contents=$(cat ./models/$1/$2/training_params.txt)/
#python main_reranker.py --epochs=10 --lr=0.00001 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --type_cands=hard_negative --resume_training --run_id=rardg79v
# lr=$(echo "$file_contents" | grep -oE 'type_model=\K[^,]+')
# epochs=$(echo "$file_contents" | grep -oP "epochs=\K[\d.]+")
# warmup_proportion=$(echo "$file_contents" | grep -oP "warmup_proportion=\K[\d.]+")

#  lr=$(echo "$file_contents" | awk -F 'lr=' '{split($2, a, ","); print a[1]}')
# epochs=$(echo "$file_contents" | awk -F 'epochs=' '{split($2, a, ","); print a[1]}')
# warmup_proportion=$(echo "$file_contents" | awk -F 'warmup_proportion=' '{split($2, a, ","); print a[1]}')
# num_layers=$(echo "$file_contents" | awk -F 'num_layers=' '{split($2, a, ","); print a[1]}')
# num_heads=$(echo "$file_contents" | awk -F 'num_heads=' '{split($2, a, ","); print a[1]}')

#  echo ${lr}
#  echo ${epochs}
#  echo ${warmup_proportion}
#  echo ${num_layers}
#  echo ${num_heads}
# echo ${epochs}
# echo ${warmup_proportion}pytho
python main_reranker.py --epochs=10 --lr=0.00001 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --type_cands=fixed_negative --identity_bert --resume_training --run_id=czituqt9    
# python main_reranker.py --epochs=50 --lr=2e-03 --num_heads=4 --num_layers=4 --type_model=mlp --warmup_proportion=0.1 --inputmark --type_cands=fixed_negative --simpleoptim --freeze_bert --resume_training --run_id=eft2pwys
# python main_reranker.py --epochs=50 --lr=2e-03 --num_heads=4 --num_layers=4 --type_model=mlp --warmup_proportion=0.1 --inputmark --type_cands=fixed_negative --simpleoptim --freeze_bert
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --type_cands=hard_negative --token_type --resume_training --run_id=v41e88xr
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --model_top=/home/jylab_share/jongsong/BLINK/models/zeshel/crossencoder/extend_extension/m62cweg5/Epochs/epoch_85 --type_cands=hard_negative
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --num_layers=12 --num_heads=12 --resume_training --run_id=m3gfre49
# python main_reranker.py --epochs=10 --lr=0.00001 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_/epoch --type_cands=hard_negative --resume_training --run_id=rardg79v
#  python main_reranker.py --warmup_proportion=${warmup_proportion} --num_layers=${num_layers} --num_heads=${num_heads} --epochs=${epochs} --lr=${lr} --cands_dir=./data/cands_anncur_top1024 --type_model=$1 --type_bert base --inputmark --training_one_epoch --resume_training --run_id=$2
# python main_reranker.py --model ./models/reranker --data ./data --B 2 --gradient_accumulation_steps 4 --num_workers 2 --warmup_proportion 0.2 --epochs 50 --gpus 0,1 --lr 0.0001626752188247764 --cands_dir=./data/cands_anncur --eval_method macro --type_model extend_multi_dot --type_bert base --inputmark --type_cands=hard_negative --training_one_epoch --resume_training --run_id=2gpzhjsj
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --model_top=/home/jylab_share/jongsong/BLINK/models/zeshel/crossencoder/extend_extension/m62cweg5/Epochs/epoch_85 --type_cands=hard_negative --resume_training --run_id=xswbidbz
#python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --model_top=/home/jylab_share/jongsong/BLINK/models/zeshel/crossencoder/extend_extension/m62cweg5/Epochs/epoch_85  --type_cands=hard_negative --resume_training --run_id=xswbidbz
#python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --type_cands=hard_negative --resume_training --run_id=8ermfyy6
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --num_layers=12 --num_heads=12 --type_hard_negative=hard_negative
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt  --type_cands=hard_negative --token_type --resume_training --run_id=92q08dif
# python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --token_type
#python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --anncur --model=./dual_encoder_zeshel.ckpt --num_layers=12 --num_heads=12 --resume_training --run_id=m3gfre49
#python main_reranker.py --epochs=20 --lr=2e-05 --type_model=extend_multi_dot --warmup_proportion=0.2 --inputmark --training_one_epoch --type_cands=hard_negative --resume_training --run_id=kou0htvy
