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
from tqdm import tqdm
import traceback
import wandb
from collections import OrderedDict

recall_preds = []
preds_list =[]
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    transformer = ['extend_multi']
    for n, p in model.named_parameters():
        count = 0
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
         'weight_decay': args.weight_decay, 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': 1e-5},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in transformer) and not any(nd in n for nd in no_decay)],
         'lr': args.lr, 'weight_decay': args.weight_decay},
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      eps=args.adam_epsilon)
    print(optimizer)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps)

    return optimizer, scheduler, num_train_steps, num_warmup_steps


def configure_optimizer_simple(args, model, num_train_examples):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = 0

    scheduler = get_constant_schedule(optimizer)

    return optimizer, scheduler, num_train_steps, num_warmup_steps

def micro_eval(model, loader_eval, num_total_unorm, args = None):
    model.eval()
    debug = args.debug
    num_total = 0
    num_correct = 0
    loss_total = 0
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader_eval), total = len(loader_eval)):
            if debug and step > 10: break
            # try:
            result = model.forward(*batch, args = args)
            if type(result) is dict:
                preds = result['predictions']
                loss_total += result['loss'].sum()

            else:
                preds = result[1]
                loss_total += result[0].sum()
            num_total += len(preds)
            num_correct += (preds == 0).sum().item()
            # except:
            #     for i in range(4):
            #         print(step, batch[i].shape)
    loss_total /= len(loader_eval)
    model.train()
    acc_norm = num_correct / num_total * 100
    acc_unorm = num_correct / num_total_unorm * 100
    return {'acc_norm': acc_norm, 'acc_unorm': acc_unorm,
            'num_correct': num_correct,
            'num_total_norm': num_total,
            'val_loss': loss_total}


def macro_eval(model, val_loaders, num_total_unorm, args = None):
    model.eval()
    assert len(val_loaders) == len(num_total_unorm)
    acc_norm = 0
    acc_unorm = 0
    num_correct_list = []
    num_total_norm = []
    loss_total = 0
    for i in tqdm(range(len(val_loaders))):
        val_loader = val_loaders[i]
        num_unorm = num_total_unorm[i]
        micro_results = micro_eval(model, val_loader, num_unorm, args = args)
        acc_norm += micro_results['acc_norm']
        acc_unorm += micro_results['acc_unorm']
        loss_total += micro_results['val_loss']
        num_correct_list.append(micro_results['num_correct'])
        num_total_norm.append(micro_results['num_total_norm'])
    acc_norm /= len(val_loaders)
    acc_unorm /= len(val_loaders)
    loss_total /= len(val_loaders)
    model.train()
    return {'acc_norm': acc_norm, 'acc_unorm': acc_unorm,
            'num_correct': num_correct_list,
            'num_total_norm': num_total_norm,
            'val_loss': loss_total}

def recall_eval(model, val_loaders, num_total_unorm, eval_mode = "micro", k = 64, all_candidates_embeds = None,  args = None):

    # if hasattr(model, 'module'):
    #     model.module.evaluate_on = True
    #     if all_candidates_embeds is not None:
    #         model.module.candidates_embeds = all_candidates_embeds
    # else:
    #     model.evaluate_on = True
    #     if all_candidates_embeds is not None:
    #         model.candidates_embeds = all_candidates_embeds
    def micro_recall_eval(model, val_loaders, num_total_unorm, k, all_candidates_embeds, args = None):
        model.eval()
        if args is not None: debug = args.debug
        nb_samples = 0
        r_k = 0
        acc_unorm = 0
        acc_norm = 0
        num_correct = 0
        recall_correct = 0
        if args.type_cands == "self_negative" or args.type_cands == "self_fixed_negative":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            scores_tensor = torch.tensor([]).to(device)
        with torch.no_grad(): 
            for i, batch in tqdm(enumerate(val_loaders), total = len(val_loaders)):
                if i > 3 and debug: break
                result = model.forward(*batch, recall_eval = True, beam_ratio = args.beam_ratio, args = args)
                if type(result) is dict:
                    scores = result['scores']
                else:
                    scores = result[2]
                if args.type_cands == "self_negative" or args.type_cands == "self_fixed_negative":
                    scores_tensor = torch.cat([scores_tensor, scores], dim = 0)
                top_k = scores.topk(k, dim=1)[1]
                preds = top_k[:, 0]
                r_k += (top_k == 0).sum().item()
                recall_correct += (top_k == 0).sum().item()
                num_correct += (preds == 0).sum().item()
                nb_samples += scores.size(0)
        r_k = recall_correct/nb_samples * 100
        acc_norm = num_correct/nb_samples * 100
        acc_unorm = num_correct/num_total_unorm * 100
        model.train()
        if hasattr(model, 'module'):
            model.module.evaluate_on = False
            model.module.candidates_embeds = None
        else:
            model.evaluate_on = False
            model.candidates_embeds = None
        if args.type_cands != "self_negative" and args.type_cands != "self_fixed_negative":
            return {"recall": r_k, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}
        else:
            return {"recall": r_k, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, \
                    "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}, scores_tensor

    if eval_mode == "micro":
        return micro_recall_eval(model, val_loaders, num_total_unorm, k, all_candidates_embeds, args)
    elif eval_mode == "macro":
        acc_norm = 0
        acc_unorm = 0
        recall = 0
        num_correct_list = []
        num_total_norm = []
        for i in range(len(val_loaders)):
            val_loader = val_loaders[i]
            num_unorm = num_total_unorm[i]
            if args.type_cands != 'self_negative' and args.type_cands != "self_fixed_negative":
                micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            else:
                micro_results, scores_tensor = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            recall += micro_results['recall']
            acc_unorm += micro_results['acc_unorm']
            acc_norm += micro_results['acc_norm']
            num_correct_list.append(micro_results['num_correct'])
            num_total_norm.append(micro_results['num_total_norm'])
        acc_norm /= len(val_loaders)
        acc_unorm /= len(val_loaders)
        recall /= len(val_loaders)
        model.train()
        return {"acc_norm": acc_norm, "acc_unorm":acc_unorm, "recall": recall, "num_total_norm": num_total_norm, "num_correct": num_correct_list}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)


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
                                 attention_type, None, False, args=args).to(device)
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
    print(args)
    set_seed(args)
    args.model = './models/{}/{}/'.format(args.type_model, args.run_id)
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    # writer = SummaryWriter()
    if not args.anncur and not args.resume_training:
        write_to_file(
            os.path.join(args.model, "training_params.txt"), str(args)
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
    cpt = torch.load(os.path.join(args.model, "pytorch_model.bin"))

    model.load_state_dict(cpt['sd'])
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

    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, len(data[1][0]))
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, len(data[1][0]))
    optimizer.load_state_dict(cpt['opt_sd'])
    scheduler.load_state_dict(cpt['scheduler_sd'])
    if device.type == 'cuda':
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    model.to(device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level=args.fp16_opt_level)

    step_num = cpt['step_num']
    tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
    best_val_perf = cpt['perf']

    loader_train, loader_val, loader_test, \
        num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                        args.C, args.B,
                                                        args.num_workers,
                                                        args.inputmark,
                                                        args.C_eval,
                                                        use_full_dataset,
                                                        macro_eval_mode, args=args)
    recall_result = recall_eval(model, loader_val, num_val_samples, eval_mode=args.eval_method, args=args)
    print('Done with epoch | recall {:8.4f} | '
               'val acc unormalized  {:8.4f} ({}/{})|'
               'val acc normalized  {:8.4f} ({}/{}) '.format(
        
        recall_result['recall'],
        recall_result['acc_unorm'],
        recall_result['num_correct'],
        num_val_samples,
        recall_result['acc_norm'],
        recall_result['num_correct'],
        recall_result['num_total_norm'],
        newline=False))
    # wandb.log(
    #     {"rereanker_recall": recall_result['recall'], "tournament_normalized_acc": recall_result['acc_norm']
    #         , "tournament_unnormalized_acc": recall_result['acc_unorm']})
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
    parser.add_argument('--num_eval_cands', default=256, type=int,
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
    parser.add_argument('--recall_eval', action='store_true',
                        help='the batch size')
    parser.add_argument('--distil', action='store_true',
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
