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
from transformers import BertTokenizer, BertModel, \
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from main_reranker import macro_eval, recall_eval, count_parameters, set_seed, write_to_file, load_model
from tqdm import tqdm
import traceback
import wandb
from collections import OrderedDict
import json

recall_preds = []
preds_list =[]

def micro_eval(model, loader_eval, num_total_unorm, args = None, mode = "valid"):
    model.eval()
    debug = args.debug
    num_total = 0
    num_correct = 0
    loss_total = 0
    if args.distill: cands = []
    print(args.cands_dir)
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader_eval), total = len(loader_eval)):
            if args.distill:
                batch[-2] = np.array(batch[-2])
                batch[-1] = np.array(batch[-1]).T
            if debug and step > 10: break
            # try:
            result = model.forward(*batch, args = args)
            if type(result) is dict:
                preds = result['predictions']
                loss_total += result['loss'].sum()
                scores = result['scores']
            else:
                preds = result[1]
                loss_total += result[0].sum()
                scores = result[2]
            num_total += len(preds)
            num_correct += (preds == 0).sum().item()
            # except:
            #     for i in range(4):
            #         print(step, batch[i].shape)
            # if step == 0: scores_list = scores
            # else: scores_list = torch.cat([scores_list, scores], dim = 0)
            if args.distill:
                for i, id in enumerate(batch[3]):
                    candidate = {}
                    candidate['mention_id'] = id
                    candidate['candidates'] = list(batch[4][i])
                    candidate['scores'] = scores[i].tolist()
                    cands.append(candidate)
                with open(os.path.join(args.save_dir, 'candidates_train_distillation_anncur.json'), 'w') as f:
                    for item in cands:
                        f.write('%s\n' % json.dumps(item))

    loss_total /= len(loader_eval)
    model.train()
    acc_norm = num_correct / num_total * 100
    acc_unorm = num_correct / num_total_unorm * 100
    return {'acc_norm': acc_norm, 'acc_unorm': acc_unorm,
            'num_correct': num_correct,
            'num_total_norm': num_total,
            'val_loss': loss_total}


def main(args):
    print(args)
    set_seed(args)
    run = wandb.init(project = "hard-nce-el", resume = "must", id = args.run_id, config = args)
    args.save_dir = '{}/{}/{}/'.format(args.save_dir, args.type_model, run.id)
    
    # writer = SummaryWriter()
    if not args.anncur and not args.resume_training:
        write_to_file(
            os.path.join(args.save_dir, "training_params.txt"), str(args)
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
                                 attention_type, None, False, args, num_heads=args.num_heads,
                                 num_layers=args.num_layers)
    count = 0
    # for each_file_name in os.listdir(args.model):
    #     if each_file_name.startswith('epoch'):
    #         if count == 1: raise ('More than two bin files found in the model directory')
    #         cpt = torch.load(os.path.join(args.model, each_file_name)) if device.type == 'cuda' \
    #             else torch.load(os.path.join(args.model, each_file_name), map_location=torch.device('cpu'))
    #         count += 1
    #         print(os.path.join(args.model, each_file_name))
    # try:
    cpt = torch.load(os.path.join(args.save_dir, "pytorch_model.bin"))

    model.load_state_dict(cpt['sd'])
    print("checkpoint from ", cpt['epoch'])

    # except:
        # pass
    dp = torch.cuda.device_count() > 1
    if dp:
        print('Data parallel across %d GPUs: %s' %
                   (len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)

    if args.type_bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    use_full_dataset = (args.type_model == 'full')
    macro_eval_mode = (args.eval_method == 'macro')
    data = load_zeshel_data(args.data, args.cands_dir, macro_eval_mode, args.debug)

    model.to(device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)


    loader_train, loader_val, loader_test, \
        num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                        args.C, args.B,
                                                        args.num_workers,
                                                        args.inputmark,
                                                        args.C_eval,
                                                        use_full_dataset,
                                                        macro_eval_mode, args=args)

    if args.type_model != "full":
        print('recall eval')

        recall_result = recall_eval(model, loader_val, num_val_samples, eval_mode=args.eval_method, args=args)
        beam_list = [0.0625, 0.25, 0.5]
        print(recall_result)
        for beam in beam_list:
            print({"recall_result/beam{}_unnorm_acc".format(str(beam)), recall_result['acc_unorm'][beam], \
            "recall_result/beam{}_norm_acc".format(str(beam)), recall_result['acc_norm'][beam],
            "recall_result/beam{}_recall".format(str(beam)), recall_result['recall'][beam]})
            wandb.log({"recall_result/beam{}_unnorm_acc".format(str(beam)): recall_result['acc_unorm'][beam], \
            "recall_result/beam{}_norm_acc".format(str(beam)): recall_result['acc_norm'][beam],
            "recall_result/beam{}_recall".format(str(beam)): recall_result['recall'][beam]})
    print('start evaluation')
    if args.eval_method == 'micro':
        test_result = micro_eval(model, loader_test, num_test_samples, args)
    elif args.eval_method == 'macro':
        test_result = macro_eval(model, loader_test, num_test_samples, args)

    print(
               'test acc unormalized  {:8.4f} ({}/{})|'
               'test acc normalized  {:8.4f} ({}/{}) '.format(
        test_result['acc_unorm'],
        test_result['num_correct'],
        num_test_samples,
        test_result['acc_norm'],
        test_result['num_correct'],
        test_result['num_total_norm'],
        newline=False))
    wandb.log({"test/unnormalized acc (cands {})".format(args.num_eval_cands): test_result['acc_unorm'], "test/normalized acc": test_result['acc_norm']
            ,"test/val_loss(cands {})".format(args.num_eval_cands): test_result['val_loss']\
            ,"test/micro_unnormalized_acc(cands {})".format(args.num_eval_cands): sum(test_result['num_correct'])/sum(test_result['num_total_unorm'])\
            ,"test/micro_normalized_acc(cands {})".format(args.num_eval_cands): sum(test_result['num_correct'])/sum(test_result['num_total_norm'])})
    
    if args.distill:
        train_result = micro_eval(model, loader_train, num_val_samples, args, mode = "train")
        print(train_result)
    if args.eval_method == 'micro':
        val_result = micro_eval(model, loader_val, num_val_samples, args)
    elif args.eval_method == 'macro':
        val_result = macro_eval(model, loader_val, num_val_samples, args)
    print('val acc unormalized  {:8.4f} ({}/{})|'
               'val acc normalized  {:8.4f} ({}/{}) '.format(
        val_result['acc_unorm'],
        val_result['num_correct'],
        num_val_samples,
        val_result['acc_norm'],
        val_result['num_correct'],
        val_result['num_total_norm'],
        newline=False))
    
    print("micro_unnormalized_acc", sum(val_result['num_correct'])/sum(val_result['num_total_unorm'])\
    ,"micro_normalized_acc", sum(val_result['num_correct'])/sum(val_result['num_total_norm']))
    wandb.log({"valid/unnormalized acc(cands {})".format(args.num_eval_cands): val_result['acc_unorm'], "valid/normalized acc(cands {})".format(args.num_eval_cands): val_result['acc_norm']
    ,"valid/val_loss(cands {})".format(args.num_eval_cands): val_result['val_loss']\
    ,"valid/micro_unnormalized_acc(cands {})".format(args.num_eval_cands): sum(val_result['num_correct'])/sum(val_result['num_total_unorm'])\
    ,"valid/micro_normalized_acc(cands {})".format(args.num_eval_cands): sum(val_result['num_correct'])/sum(val_result['num_total_norm'])})


    # wandb.log(
    #     {"rereanker_recall": recall_result['recall'], "tournament_normalized_acc": recall_result['acc_norm']
    #         , "tournament_unnormalized_acc": recall_result['acc_unorm']})


    # wandb.log({"test_unnormalized acc": test_result['acc_unorm'], "test_normalized acc": test_result['acc_norm']})


#  writer.close()


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
    parser.add_argument('--fixed_initial_weight', action='store_true',
                        help='fixed weight for identity weight')
    parser.add_argument('--save_dir', type = str, default = '/shared/s3/lab07/jongsong/hard-nce-el_garage/models',
                        help='debugging mode')

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
    parser.add_argument('--training_one_epoch', action = 'store_true',
                        help='stop the training after one epoch')
    parser.add_argument('--type_cands', type=str,
                        default='fixed_negative',
                        choices=['fixed_negative',
                                'mixed_negative',
                                 'hard_negative',
                                 'self_negative',
                                 'self_fixed_negative'],
                        help='fixed: top k, hard: distributionally sampled k')
    parser.add_argument('--run_id', type = str,
                        help='run id when resuming the run')
    parser.add_argument('--num_training_cands', default=64, type=int,
                        help='# of candidates')
    parser.add_argument('--num_eval_cands', default=64, type=int,
                        help='# of candidates')
    parser.add_argument('--num_recall_cands', default=1024, type=int,
                        help='# of candidates')
    parser.add_argument('--final_k', default=64, type=int,
                        help='# of candidates')
    parser.add_argument('--train_one_epoch', action = 'store_true',
                        help='training only one epoch')
    parser.add_argument('--distill_training', action = 'store_true',
                        help='training only one epoch')
    parser.add_argument('--val_random_shuffle', action = 'store_true',
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
    parser.add_argument('--recall_eval', action='store_true',
                        help='the batch size')
    parser.add_argument('--distill', action='store_true',
                        help='the batch size')
    parser.add_argument('--bert_lr', type=float, default=1e-5,
                        help='the batch size')
    parser.add_argument('--num_sampled', type=int, default=256,
                        help='the batch size')
    parser.add_argument('--mlp_layers', type=str, default=1536,
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
    preds_list = torch.tensor(preds_list)
    recall_preds = torch.tensor(recall_preds)

    print(preds_list)
    print(recall_preds)
    print((preds_list == recall_preds).sum().item())
