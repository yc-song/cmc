import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
from data_reranker import get_loaders, load_zeshel_data, Logger
from datetime import datetime
from reranker import FullRanker
from retriever import UnifiedRetriever
from data_retriever import get_all_entity_hiddens
import shutil

from transformers import BertTokenizer, BertModel, \
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
import wandb
import deepspeed
from deepspeed.profiling.flops_profiler import get_model_profile
from deepspeed.accelerator import get_accelerator
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def load_model(model_path, device, eval_mode=False, dp=False, type_model='full'):
    package = torch.load(model_path) if device.type == 'cuda' else \
        torch.load(model_path, map_location=torch.device('cpu'))

    if args.type_bert == 'base':
        encoder = BertModel.from_pretrained('bert-base-uncased')
    else:
        encoder = BertModel.from_pretrained('bert-large-uncased')
    if type_model == 'full':
        model = FullRanker(encoder, device).to(device)
    else:
        if args.type_model == 'poly':
            attention_type = 'soft_attention'
        else:
            attention_type = 'hard_attention'
        if args.type_model == 'dual':
            num_mention_vecs = 1
            num_entity_vecs = 1
        elif args.type_model == 'multi_vector':
            num_mention_vecs = 1
            num_entity_vecs = args.num_entity_vecs
        else:
            num_mention_vecs = args.num_mention_vecs
            num_entity_vecs = args.num_entity_vecs
        model = UnifiedRetriever(encoder, device, num_mention_vecs,
                                 num_entity_vecs,
                                 args.mention_use_codes, args.entity_use_codes,
                                 attention_type, None, False).to(device)
    if dp:
        from collections import OrderedDict
        state_dict = package['sd']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(package['sd'])

    if eval_mode:
        model.eval()
    return model


def main(args):

    with get_accelerator().device(0,1):
        device = torch.device('cuda')
        if args.type_bert == 'base':
            encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            encoder = BertModel.from_pretrained('bert-large-uncased')
        if args.type_model == 'full':
            model = FullRanker(encoder, device, args)
        else:
            if args.type_model == 'poly':
                attention_type = 'soft_attention'
            else:
                attention_type = 'hard_attention'
            if args.type_model == 'dual':
                num_mention_vecs = 1
                num_entity_vecs = 1
            elif args.type_model == 'multi_vector':
                num_mention_vecs = 1
                num_entity_vecs = args.num_entity_vecs
            elif args.type_model == 'extend_multi':
                attention_type = 'extend_multi'
                num_mention_vecs = 1
                num_entity_vecs = 1
            elif args.type_model == 'extend_multi_dot':
                attention_type = 'extend_multi_dot'
                num_mention_vecs = 1
                num_entity_vecs = 1
            else:
                num_mention_vecs = args.num_mention_vecs
                num_entity_vecs = args.num_entity_vecs
            model = UnifiedRetriever(encoder, device, num_mention_vecs,
                                num_entity_vecs,
                                args.mention_use_codes, args.entity_use_codes,
                                attention_type, None, False, args, num_heads = args.num_heads, num_layers = args.num_layers)
        if args.type_model != "full":
            batch = {"mention_token_ids": torch.randint(1,3, (1, 128)),\
            "mention_masks": torch.randint(1,3, (1, 128)),\
            "candidate_token_ids": torch.randint(1,3, (1, 256, 128)),\
            "candidate_masks": torch.randint(1,3, (1, 256, 128)),\
            "args": args}
        else:
            batch = {
                "encoded_pairs":torch.randint(1,3, (1, 128, 256)),\
                "type_marks":torch.randint(0,2, (1,128, 256)),\
                "input_lens": torch.randint(0,2, (1, 128)),
                "args": args}

        model = model.to(device)
        flops, macs, params = get_model_profile(model=model, # model
                                    kwargs=batch, # list of positional arguments to the model.
                                    ignore_modules=None) # the list of modules to ignore in the profiling
        print(flops, macs, params)
#  writer.close()
def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,default='./models/reranker',
                        help='model path')
    parser.add_argument('--model_top', type=str,
                        help='when loading top model path')
    parser.add_argument('--data', type=str,default='./data',
                        help='data path (zeshel data directory)')
    parser.add_argument('--cands_dir', type=str, default = './data/cands_anncur_top1024',
                        help='candidates directory')
    parser.add_argument('--L', type=int, default=256,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--C', type=int, default=64,
                        help='max number of candidates [%(default)d]')
    parser.add_argument('--C_eval', type=int, default=64,
                        help='max number of candidates [%(default)d] when eval')
    parser.add_argument('--B', type=int, default=4,
                        help='batch size [%(default)d]')
    parser.add_argument('--freeze_bert', action = 'store_true',
                        help='freeze encoder BERT')
    parser.add_argument('--identity_bert', action = 'store_true',
                        help='assign same query and key matrix')
    parser.add_argument('--eval_batch_size', type=int, default=8,
                        help='evaluation batch size [%(default)d]')

    parser.add_argument('--lr', type=float, default=2e-5,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--epochs', type=int, default=3,
                        help='max number of epochs [%(default)d]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--init', type=float, default=0,
                        help='init (default if 0) [%(default)g]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num workers [%(default)d]')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--eval_method', default='macro', type=str,
                        choices=['macro', 'micro', 'skip'],
                        help='the evaluate method')
    parser.add_argument('--type_model', type=str,
                        default='full',
                        choices=['dual',
                                 'sum_max',
                                 'multi_vector',
                                 'poly',
                                 'full',
                                 'extend_multi',
                                 'extend_multi_dot',
                                 'mlp_with_som'],
                        help='the type of model')
    parser.add_argument('--debug', action = 'store_true',
                        help='debugging mode')
    parser.add_argument('--save_dir', type = str, default = '/shared/s3/lab07/jongsong/hard-nce-el/models',
                        help='debugging mode')
    parser.add_argument('--training_one_epoch', action = 'store_true',
                        help='stop the training after one epoch')
    parser.add_argument('--type_cands', type=str,
                        default='fixed_negative',
                        choices=['fixed_negative',
                                'mixed_negative',
                                 'hard_negative',
                                 'self_negative',
                                 'self_fixed_negative',
                                 'self_mixed_negative'],
                        help='fixed: top k, hard: distributionally sampled k')
    parser.add_argument('--run_id', type = str,
                        help='run id when resuming the run')
    parser.add_argument('--num_training_cands', default=64, type=int,
                        help='# of candidates')
    parser.add_argument('--num_recall_cands', default=1024, type=int,
                        help='# of candidates')
    parser.add_argument('--train_one_epoch', action = 'store_true',
                        help='training only one epoch')
    parser.add_argument('--cands_ratio', default=0.5, type=float,
                        help='ratio of sampled negatives')
    parser.add_argument('--num_heads', type=int, default=16,
                        help='the number of multi-head attention in extend_multi ')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='the number of atten                ion layers in extend_multi ')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from checkpoint?')
    parser.add_argument('--type_bert', type=str,
                        default='base',
                        choices=['base', 'large'],
                        help='the type of encoder')
    parser.add_argument('--num_mention_vecs', type=int, default=8,
                        help='the number of mention vectors ')
    parser.add_argument('--num_entity_vecs', type=int, default=8,
                        help='the number of entity vectors  ')
    parser.add_argument('--mention_use_codes', action='store_true',
                        help='use codes for mention embeddings?')
    parser.add_argument('--entity_use_codes', action='store_true',
                        help='use codes for entity embeddings?')
    parser.add_argument('--inputmark', action='store_true',
                        help='use mention mark at input?')
    parser.add_argument('--store_en_hiddens', action='store_true',
                        help='store entity hiddens?')
    parser.add_argument('--en_hidden_path', type=str,
                        help='all entity hidden states path')
    parser.add_argument('--beam_ratio', type=float, default=0.25,
                        help='all entity hidden states path')
    parser.add_argument('--too_large', action='store_true',
                        help='all entity hidden states path')
    parser.add_argument('--entity_bsz', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--lambda_scheduler', action='store_true',
                        help='the batch size')
    parser.add_argument('--fixed_initial_weight', action='store_true',
                        help='fixed weight for identity weight')
    parser.add_argument('--recall_eval', action='store_true',
                        help='the batch size')
    parser.add_argument('--val_random_shuffle', action = 'store_true',
                        help='training only one epoch')
    parser.add_argument('--distill', action='store_true',
                        help='getting score distribution for distillation purpose')
    parser.add_argument('--distill_training', action='store_true',
                        help='training smaller model from crossencoder scores')
    parser.add_argument('--bert_lr', type=float, default=1e-5,
                        help='the batch size')
    parser.add_argument('--num_sampled', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--mlp_layers', type=str, default="1536",
                        help='num of layers for mlp or mlp-with-som model (except for the first and the last layer)')
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) "
             "instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', "
             "'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument('--anncur', action='store_true', help = "load anncur ckpt")
    parser.add_argument('--token_type', action='store_false', help = "no token type id when specified")

    args = parser.parse_args()

    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior

    main(args)
