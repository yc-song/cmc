## requirements
conda env create --file environment.yml [1] 

Download the public zeshel data [here](https://github.com/lajanugen/zeshel) [2].


## Reproducibility

### training
'''
python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.1 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=1 --save_topk_nce --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_top1024 --type_bert=base --B=2 --gradient_accumulation_steps=2 --gpus=0 --model=./cocondenser.bin --cocondenser --distill_training --alpha=0.2 --identity_bert --fixed_initial_weight
'''

### Evaluation
```
python main_reranker.py --epochs=5 --bert_lr=1e-05 --lr=1e-05 --type_model=extend_multi_dot --warmup_proportion=0.1 --inputmark --type_cands=mixed_negative --token_type --eval_batch_size=1 --save_topk_nce --save_dir=./models --num_layers=2 --num_heads=4 --cands_dir=./data/cands_top1024 --type_bert=base --B=2 --gradient_accumulation_steps=2 --gpus=0 --model=./cocondenser.bin --cocondenser --distill_training --alpha=0.2 --identity_bert --fixed_initial_weight --training_finished --resume_training --run_id=[run_id]

```



