import argparse
import numpy as np
import os
import random
import torch
import torch.nn as nn
from data_reranker import get_loaders, load_zeshel_data, Logger, preprocess_data
from datetime import datetime
from reranker import FullRanker
from retriever import UnifiedRetriever
from main_retriever import evaluate
from scipy.stats import rankdata
from data_retriever import get_all_entity_hiddens, Data
from transformers import BertTokenizer, BertModel, \
    AdamW, get_linear_schedule_with_warmup, get_constant_schedule
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
import traceback
import wandb
from collections import OrderedDict
import json

self_negative_methods=[
    'self_fixed_negative',
    'self_mixed_negative',
    'self_negative'
]
depth = [2,4,8,16,32,64]
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic=True

def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    transformer = ['extend_multi', 'mlp']
    identity_init = ['transformerencoderlayer.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
        'weight_decay': args.weight_decay, 'lr': args.bert_lr},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0, 'lr': args.bert_lr},
    {'params': [p for n, p in model.named_parameters()
                if any(nd in n for nd in transformer) and not any(nd in n for nd in no_decay)],
        'lr': args.lr, 'weight_decay': args.weight_decay},
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                    eps=args.adam_epsilon)
    # except:
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in transformer) and any(nd in n for nd in no_decay) and not any(nd in n for nd in identity_init)],
        'lr': args.lr, 'weight_decay': 0.0,
        'names': [n for n, p in model.named_parameters()
                    if any(nd in n for nd in transformer) and any(nd in n for nd in no_decay) and not any(nd in n for nd in identity_init)]},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
        'weight_decay': 0.0, 'lr': args.bert_lr,
        'names': [n for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)]},
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in transformer) and not any(nd in n for nd in identity_init) ],
        'lr': args.lr, 'weight_decay': args.weight_decay,
        'names': [n for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and any(nd in n for nd in transformer) and not any(nd in n for nd in identity_init)]},
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
        'weight_decay': args.weight_decay, 'lr': args.bert_lr,
        'names': [n for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)]},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in identity_init)],
        'weight_decay': args.weight_decay, 'lr': args.weight_lr,
        'names': [n for n, p in model.named_parameters()
                    if any(nd in n for nd in identity_init)]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                    eps=args.adam_epsilon)

    print(optimizer)

    num_train_steps = int(num_train_examples / args.B /
                          args.gradient_accumulation_steps * args.epochs)
    num_warmup_steps = int(num_train_steps * args.warmup_proportion)
    if args.lambda_scheduler:
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / num_train_steps)
    else:
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


def micro_eval(model, loader_eval, num_total_unorm, args = None, mode = None):
    if args.store_reranker_score: assert mode is not None

    model.eval()
    debug = args.debug
    num_total = 0
    num_correct = 0
    loss_total = 0
    recall_correct_list = [0]*len(depth)
    mrr_total = 0
    if args.store_reranker_score:
        cands = []
        with open(os.path.join(args.cands_dir, 'candidates_%s.json' % mode), 'r') as f:
            for line in f:
                cands.append(json.loads(line))

    if args.distill: cands = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(loader_eval), total = len(loader_eval)):
            if args.distill:
                batch[-2] = np.array(batch[-2]) 
                batch[-1] = np.array(batch[-1]).T
            if debug and step > 50: break
            # try:
            # batch["args"] = args
            result = model.forward(**batch, args = args)
            label_ids = batch["label_idx"]
            if type(result) is dict:
                preds = result['predictions']
                loss_total += result['loss'].sum()
                scores = result['scores']
            else:
                preds = result[1]
                loss_total += result[0].sum()
                scores = result[2]
            if args.store_reranker_score:
                for i in range(scores.size(0)):
                    cands[step*scores.size(0)+i]['reranker_scores'] = scores[i].tolist()
            # if args.debug:
            #     scores = torch.zeros_like(scores)
            #     scores[:, 0]=1
            #     preds = torch.zeros_like(scores)
            #     label_ids = torch.zeros_like(label_ids).int()
            num_total += len(preds)
            num_correct += (preds.cpu() == label_ids).sum().item()
            for k in range(len(depth)):
                recall_correct_list[k] += (scores.topk(depth[k], dim=1)[1].cpu()==label_ids.unsqueeze(-1)).sum().item()
            # recall_correct += (scores.topk(k, dim=1)[1].cpu()==label_ids.unsqueeze(-1)).sum().item()
            # getting mrr
            # after ranking scores, finding out the rank of lables
            idx_array = rankdata(-scores.cpu().detach().numpy(), axis = 1, method = 'min')
            rank = np.take_along_axis(idx_array, label_ids[:, None].cpu().detach().numpy(), axis=1)
            mrr_total += np.sum(1/rank)
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
            if args.test_at_first:
                print(batch, scores)
        if args.distill:
            with open(os.path.join(args.cands_dir, 'candidates_train_distillation.json'), 'w') as f:
                for item in cands:
                    f.write('%s\n' % json.dumps(item))
    if args.store_reranker_score:
        with open(os.path.join(args.cands_dir, 'candidates_%s_w_reranker_score.json' % mode), 'w') as f:
            for item in cands:
                f.write('%s\n' % json.dumps(item))
    loss_total /= len(loader_eval)
    model.train()
    acc_norm = num_correct / num_total * 100
    if args.debug: num_total_unorm = num_total
    print(num_total_unorm)
    acc_unorm = num_correct / num_total_unorm * 100
    recall_correct_list = [item / num_total_unorm * 100 for item in recall_correct_list]
    mrr = mrr_total / num_total * 100
    return {'acc_norm': acc_norm, 'acc_unorm': acc_unorm,
            'num_correct': num_correct,
            'num_total_norm': num_total,
            'num_total_unorm': num_total_unorm,
            'val_loss': loss_total,
            "recall": recall_correct_list,
            'mrr': mrr}


def macro_eval(model, val_loaders, num_total_unorm, args = None):
    model.eval()
    assert len(val_loaders) == len(num_total_unorm)
    acc_norm = 0
    acc_unorm = 0
    num_correct_list = []
    num_total_norm = []
    loss_total = 0
    recall = [0]*len(depth)
    mrr_total = 0
    mrr_list = []
    recall_correct_list = []
    for i in tqdm(range(len(val_loaders))):
        val_loader = val_loaders[i]
        num_unorm = num_total_unorm[i]
        micro_results = micro_eval(model, val_loader, num_unorm, args = args)
        print(micro_results)
        acc_norm += micro_results['acc_norm']
        acc_unorm += micro_results['acc_unorm']
        loss_total += micro_results['val_loss']
        num_correct_list.append(micro_results['num_correct'])
        num_total_norm.append(micro_results['num_total_norm'])
        for k in range(len(depth)):
            recall[k] += micro_results['recall'][k] 
        print(recall)
        recall_correct_list.append(micro_results['recall'])
        mrr_total += micro_results['mrr']
        mrr_list.append(micro_results['mrr'])
    acc_norm /= len(val_loaders)
    acc_unorm /= len(val_loaders)
    loss_total /= len(val_loaders)
    for i in range(len(recall)):
        recall[i] /= len(val_loaders)
    mrr_total /= len(val_loaders)

    model.train()
    return {'acc_norm': acc_norm, 'acc_unorm': acc_unorm,
            'num_correct': num_correct_list,
            'num_total_norm': num_total_norm,
            'num_total_unorm': num_total_unorm,
            'val_loss': loss_total,
            "recall": recall,
            "recall_correct": recall_correct_list,
            "mrr": mrr_total,
            "mrr_correct": mrr_list}

def recall_eval(model, val_loaders, num_total_unorm, eval_mode = "micro", k = 64, all_candidates_embeds = None,  args = None):

    # if hasattr(model, 'module'):
    #     model.module.evaluate_on = True
    #     if all_candidates_embeds is not None:
    #         model.module.candidates_embeds = all_candidates_embeds
    # else:
    #     model.evaluate_on = True
    #     if all_candidates_embeds is not None:
    #         model.candidates_embeds = all_candidates_embeds
    def micro_recall_eval(model, val_loaders, num_total_unorm, k, all_candidates_embeds, args = None, beam = None):
        model.eval()
        if args is not None: debug = args.debug
        nb_samples = 0
        recall = 0
        acc_unorm = 0
        acc_norm = 0
        num_correct = 0
        recall_correct = 0
        if args.type_cands in self_negative_methods:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            scores_tensor = torch.tensor([]).to(device)
        with torch.no_grad(): 
            for i, batch in tqdm(enumerate(val_loaders), total = len(val_loaders)):
                top_k = model.forward(**batch, recall_eval = True, beam_ratio = args.beam_ratio, args = args).int()
                # print(top_k, batch["label_idx"])
                preds = top_k[:, 0]
                recall_correct += (top_k.cpu() == batch["label_idx"].unsqueeze(-1)).sum().item() 
                num_correct += (preds.cpu() == batch["label_idx"]).sum().item()
                nb_samples += preds.size(0)
                # print(recall_correct, num_correct)
            recall= recall_correct / num_total_unorm * 100
            acc_unorm = num_correct / num_total_unorm * 100
            acc_norm = num_correct / nb_samples * 100
        model.train()
        if hasattr(model, 'module'):
            model.module.evaluate_on = False
            model.module.candidates_embeds = None
        else:
            model.evaluate_on = False
            model.candidates_embeds = None
        if not args.type_cands in self_negative_methods:
            return {"recall": recall, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, \
                    "num_correct": num_correct}

        else:
            return {"recall": recall, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}

            # return {"recall": r_k, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, \
                    # "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}, scores_tensor

    if eval_mode == "micro":
        return micro_recall_eval(model, val_loaders, num_total_unorm, k, all_candidates_embeds, args)
    elif eval_mode == "macro":
        acc_norm = 0
        acc_unorm = 0
        recall = 0
        num_correct_list =[]
        num_total_norm = []   
        for i in range(len(val_loaders)):
            print("micro eval world {}".format(i))
            val_loader = val_loaders[i]
            num_unorm = num_total_unorm[i]
            if not args.type_cands in self_negative_methods:
                micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            else:
                micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            print(micro_results)
            recall += micro_results['recall']
            acc_unorm += micro_results['acc_unorm']
            acc_norm += micro_results['acc_norm']
            num_correct_list.append(micro_results['num_correct'])
            num_total_norm.append(micro_results['num_total_norm'])
        acc_norm /= len(val_loaders)
        acc_unorm /= len(val_loaders)
        recall /= len(val_loaders)
        num_correct = num_correct_list
        nb_samples = num_total_norm
        model.train()
        return {"recall": recall, "acc_norm": acc_norm, "acc_unorm":acc_unorm,\
        "num_correct": num_correct_list, "num_total_norm": num_total_norm}
            # return {"recall": r_k, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, \
                    # "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}, scores_tensor

    recall_dict = {}
    num_correct_dict = {}
    nb_samples_dict = {}
    acc_norm_dict = {}
    acc_unorm_dict = {}
    beam_ratios = [0.0625,0.25, 0.5]
    if eval_mode == "micro":
        return micro_recall_eval(model, val_loaders, num_total_unorm, k, all_candidates_embeds, args)
    elif eval_mode == "macro":
        acc_norm = 0
        acc_unorm = 0
        recall = 0
        for beam in beam_ratios:
            recall_dict[beam] = 0.
            acc_norm_dict[beam] = 0.
            acc_unorm_dict[beam] = 0.   
        num_correct_list =[]
        num_total_norm = []   
        for i in range(len(val_loaders)):
            print("micro eval world {}".format(i))
            val_loader = val_loaders[i]
            num_unorm = num_total_unorm[i]
            if not args.type_cands in self_negative_methods:
                micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            else:
                micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
            print(micro_results)
            for beam in beam_ratios:
                recall_dict[beam] += micro_results['recall'][beam]
                acc_unorm_dict[beam] += micro_results['acc_unorm'][beam]
                acc_norm_dict[beam] += micro_results['acc_norm'][beam]
                num_correct_list.append(micro_results['num_correct'][beam])
                num_total_norm.append(micro_results['num_total_norm'][beam])
        for beam in beam_ratios:
            acc_norm_dict[beam] /= len(val_loaders)
            acc_unorm_dict[beam] /= len(val_loaders)
            recall_dict[beam] /= len(val_loaders)
            num_correct_dict[beam] = num_correct_list
            nb_samples_dict[beam] = num_total_norm
        model.train()
        return {"recall": recall_dict, "acc_norm": acc_norm_dict, "acc_unorm":acc_unorm_dict,\
        "num_correct": num_correct_list, "num_total_norm": num_total_norm}
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def write_to_file(path, string, mode="w"):
    with open(path, mode) as writer:
        writer.write(string)

def load_model(model_path, device, eval_mode=False, dp=False, type_model='full', model = None):
    package = torch.load(model_path) if device.type == 'cuda' else \
        torch.load(model_path, map_location=torch.device('cpu'))
    if model is None:
        if args.type_bert == 'base':
            encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            encoder = BertModel.from_pretrained('bert-large-uncased')
        if type_model == 'full':
            model = FullRanker(encoder, device, args).to(device)
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
                num_mention_vecs = args.num_mention_vecs
                num_entity_vecs = 1
            else:
                num_mention_vecs = args.num_mention_vecs
                num_entity_vecs = args.num_entity_vecs
            model = UnifiedRetriever(encoder, device, num_mention_vecs,
                                    num_entity_vecs,
                                    args.mention_use_codes, args.entity_use_codes,
                                    attention_type, None, False, args = args).to(device)
    if dp:
        # model.load_state_dict(package['sd'])
        from collections import OrderedDict
        state_dict = package['sd']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(package['sd'])

    if eval_mode:
        model.eval()
    return model


def main(args):
    # Set random seed
    set_seed(args)
    # For resuming training, taking run id as an argument
    if args.resume_training:
        run = wandb.init(project = "hard-nce-el", resume = "must", id = args.run_id, config = args, settings=wandb.Settings(_service_wait=300))
    else:
        run = wandb.init(project="hard-nce-el", config = args, settings=wandb.Settings(_service_wait=300))
    # The directory to save model
    args.save_dir = '{}/{}/{}/'.format(args.save_dir, args.type_model, run.id)
    # If directory does not exist, make directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # Define logger
    logger = Logger(args.save_dir + '.log', True)
    logger.log(str(args))
    loader_val = None
    loader_test = None
    if not args.anncur and not args.resume_training:
        write_to_file(
            os.path.join(args.save_dir, "training_params.txt"), str(args)
        )
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Define BERT model
    if args.type_bert == 'base':
        encoder = BertModel.from_pretrained('bert-base-uncased')
    else:
        encoder = BertModel.from_pretrained('bert-large-uncased')
    # Assign model type
    ## Full: cross-encoder
    ## Dual: bi-encoder - requires only one mention and one candidate vector
    ## Extend_multi: comparing multiple entities with linear layer head
    ## Extend_multi_dot: comparing multiple entities with linear layer head
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
            num_mention_vecs = args.num_mention_vecs
            num_mention_vecs = args.num_mention_vecs
            num_entity_vecs = 1
        elif args.type_model == 'mlp_with_som':
            attention_type = 'mlp_with_som'
            num_mention_vecs = args.num_mention_vecs
            num_entity_vecs = args.num_entity_vecs
        elif args.type_model == 'mlp':
            attention_type = 'mlp'
            num_mention_vecs = 1
            num_entity_vecs = 1
        else:
            num_mention_vecs = args.num_mention_vecs
            num_entity_vecs = args.num_entity_vecs
        # freeze_bert option: disable backprop for bi-encoder
        if args.freeze_bert:
            model = FrozenRetriever(encoder, device, num_mention_vecs,
                                 num_entity_vecs,
                                 args.mention_use_codes, args.entity_use_codes,
                                 attention_type, None, False,args, num_heads = args.num_heads, num_layers = args.num_layers)
        # Unified retriever class: Class for non-cross-encoder type model
        else:
            model = UnifiedRetriever(encoder, device, num_mention_vecs,
                                 num_entity_vecs,
                                 args.mention_use_codes, args.entity_use_codes,
                                 attention_type, None, False,args, num_heads = args.num_heads, num_layers = args.num_layers)
    # load bi-encoder from anncur retriever (https://github.com/iesl/anncur)
    if args.anncur and not args.resume_training:
        package = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
        new_state_dict = package['state_dict']
        modified_state_dict = OrderedDict()
        # Change dict keys for loading model parameter
        for k, v in new_state_dict.items():
            if k.startswith('model.input_encoder'):
                name = k.replace('model.input_encoder.bert_model', 'mention_encoder')
            elif k.startswith('model.label_encoder'):
                name = k.replace('model.label_encoder.bert_model', 'entity_encoder')
            modified_state_dict[name] = v
        model.load_state_dict(modified_state_dict, strict = False)
        # Loading pre-trained transformer encoder (deprecated)
        if args.model_top is not None:
            package = torch.load(args.model_top) if device.type == 'cuda' \
            else torch.load(args.model_top, map_location=torch.device('cpu'))
            new_state_dict = package['model_state_dict']
            modified_state_dict = OrderedDict()
            for k, v in new_state_dict.items():
                name = k.replace('model.', '')
                modified_state_dict[name] = v
            model.extend_multi.load_state_dict(modified_state_dict, strict = False)
    # Load saved model
    if args.resume_training:
        count = 0
        print(args.save_dir)
        for each_file_name in os.listdir(args.save_dir):
            if each_file_name.startswith('epoch'):
                if count == 1: raise('More than two bin files found in the model directory')
                cpt=torch.load(os.path.join(args.save_dir,each_file_name)) if device.type == 'cuda' \
                else torch.load(os.path.join(args.save_dir,each_file_name) , map_location=torch.device('cpu'))
                count+=1
                print(os.path.join(args.save_dir,each_file_name))
        
        try:
            model.load_state_dict(cpt['sd'])
        except: 
            new_state_dict = cpt['sd']
            modified_state_dict = OrderedDict()
            # Change dict keys for loading model parameter
            for k, v in new_state_dict.items():
                if k.startswith('transformerencoder'):
                    k = k.replace('transformerencoder', 'extend_multi_transformerencoder')
                elif k.startswith('linearhead'):
                    k = k.replace('linearhead', 'extend_multi_linearhead')
                elif k.startswith('token_type_embeddings'):
                    k = k.replace('token_type_embeddings', 'extend_multi_token_type_embeddings')
                modified_state_dict[k] = v
            model.load_state_dict(modified_state_dict)            
    # Variables for parallel computing
    dp = torch.cuda.device_count() > 1
    if dp:
        logger.log('Data parallel across %d GPUs: %s' %
                   (len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)

    if args.type_bert == 'base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    use_full_dataset = (args.type_model == 'full') # Load dataset for cross-encoder
    macro_eval_mode = (args.eval_method == 'macro') # Load dataset for different eval type
    data = load_zeshel_data(args.data, args.cands_dir, macro_eval_mode, args.debug, nearest = args.nearest)
    loader_test = None
    # simpleoptim: disable learning rate schduler
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, len(data[1][0]))
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, len(data[1][0]))
    if args.resume_training and not args.training_finished:
        try:
            optimizer.load_state_dict(cpt['opt_sd'])
        except: 
            no_decay = ['bias', 'LayerNorm.weight']
            transformer = ['extend_multi', 'mlp']
            identity_init = ['transformerencoderlayer.weight']
            optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in transformer) and any(nd in n for nd in no_decay) and not any(nd in n for nd in identity_init)],
            'lr': args.lr, 'weight_decay': 0.0,
            'names': [n for n, p in model.named_parameters()
                        if any(nd in n for nd in transformer) and any(nd in n for nd in no_decay) and not any(nd in n for nd in identity_init)]},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
            'weight_decay': 0.0, 'lr': args.bert_lr,
            'names': [n for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)]},
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and any(nd in n for nd in transformer) and not any(nd in n for nd in identity_init) ],
            'lr': args.lr, 'weight_decay': args.weight_decay,
            'names': [n for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and any(nd in n for nd in transformer) and not any(nd in n for nd in identity_init)]},
            {'params': [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)],
            'weight_decay': args.weight_decay, 'lr': args.bert_lr,
            'names': [n for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and not any(nd in n for nd in transformer)]},
            {'params': [p for n, p in model.named_parameters()
                        if any(nd in n for nd in identity_init)],
            'weight_decay': args.weight_decay, 'lr': args.weight_lr,
            'names': [n for n, p in model.named_parameters()
                        if any(nd in n for nd in identity_init)]}
            ]
            optimizer = AdamW(optimizer_grouped_parameters,
                      eps=args.adam_epsilon)
            optimizer.load_state_dict(cpt['opt_sd'])
            print(optimizer)    
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


    logger.log('\n[TRAIN]')
    logger.log('  # train samples: %d' % len(data[1][0]))
    logger.log('  # epochs:        %d' % args.epochs)
    logger.log('  batch size:      %d' % args.B)
    logger.log('  grad accum steps %d' % args.gradient_accumulation_steps)
    logger.log('    (effective batch size w/ accumulation: %d)' %
               (args.B * args.gradient_accumulation_steps))
    logger.log('  # train steps:   %d' % num_train_steps)
    logger.log('  # warmup steps:  %d' % num_warmup_steps)
    logger.log('  learning rate:   %g' % args.lr)
    logger.log('  # parameters:    %d' % count_parameters(model))
    start_time = datetime.now()
    # When training_finished == True, skip training loop and carry out evaluation
    training_finished = args.training_finished
    step_num = 0
    trigger_times = 0 # Overfitting
    tr_loss, logging_loss = 0.0, 0.0
    # For distillation, define teacher and student loss for keep tracking
    if args.distill_training or args.regularization:
        student_tr_loss = 0.
        teacher_tr_loss = 0.
        student_logging_loss = 0.
        teacher_logging_loss = 0.
    model.zero_grad() # Optimizer zero grad
    best_val_perf = float('-inf')
    # Load epoch and loss from checkpoint
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
        best_val_perf = cpt['perf']
        print("cpt perf:", cpt['perf'])
        print("best val perf:", best_val_perf)
    start_epoch = 1
    # Begin from the epoch where training finished
    if args.resume_training:
        start_epoch = cpt['epoch'] + 1 
    # Deprecated
    if args.freeze_bert:
        loader_train, loader_val, loader_test, \
        num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                    args.C, args.B,
                                                    args.num_workers,
                                                    args.inputmark,
                                                    args.C_eval,
                                                    use_full_dataset,
                                                    macro_eval_mode, args = args)
        loader_train = preprocess_data(loader_train, model, args.debug)
        loader_val = preprocess_data(loader_val, model, args.debug)
        loader_test = preprocess_data(loader_test, model, args.debug)
    for epoch in range(start_epoch, args.epochs + 1):
        if training_finished: break
        logger.log('\nEpoch %d' % epoch)

        if not args.freeze_bert:
            data = load_zeshel_data(args.data, args.cands_dir, macro_eval_mode, args.debug, nearest=args.nearest)

            loader_train, loader_val, loader_test, \
            num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                    args.C, args.B,
                                                    args.num_workers,
                                                    args.inputmark,
                                                    args.C_eval,
                                                    use_full_dataset,
                                                    macro_eval_mode, args = args)
        # For self negative mining strategies, getting score distribution from the model
        if args.type_cands in self_negative_methods:
            with torch.no_grad():
                try:
                    torch.multiprocessing.set_start_method('spawn')
                except:
                    pass
                scores_tensor = torch.tensor([]).to(device)
                recall_evaluate = False
                model.evaluate_on = False
                # if args.type_model == 'extend_multi' or args.type_model == 'extend_multi_dot':
                #     recall_evaluate = True
                for i, batch in tqdm(enumerate(loader_train), total=len(loader_train)):
                    result = model.forward(**batch, recall_eval=recall_evaluate, beam_ratio=args.beam_ratio, sampling = True, args = args)
                    try:
                        scores = result[2]
                    except:
                        scores = result['scores']
                    scores_tensor = torch.cat([scores_tensor, scores], dim=0)
            data = load_zeshel_data(args.data, args.cands_dir, macro_eval_mode, args.debug, scores = scores_tensor, nearest=args.nearest)
            # Getting new train dataloader
            loader_train, _, _, \
            _, _ = get_loaders(data, tokenizer, args.L,
                        args.C, args.B,
                        args.num_workers,
                        args.inputmark,
                        args.C_eval,
                        use_full_dataset,
                        macro_eval_mode, args=args, self_negs_again=True)
            if args.freeze_bert:
                loader_train = preprocess_data(loader_train, model, args.debug)
        # Test the model from the beginning
        if args.test_at_first:
            if args.eval_method == 'micro':
                val_result = micro_eval(model, loader_val, num_val_samples, args)
            elif args.eval_method == 'macro':
                val_result = macro_eval(model, loader_val, num_val_samples, args)

        print('Begin Training')
        # Training loop
        for batch_idx, batch in tqdm(enumerate(loader_train), total = len(loader_train)):  # Shuffled every epoch
            if args.debug and batch_idx > 10: break
            model.train()
            # Forward pass
            result = model.forward(**batch, args = args)

            if type(result) is tuple:
                loss = result[0]
                scores = result[2]
            elif type(result) is dict:
                loss = result['loss']
                scores = result['scores']
            if args.distill_training or args.regularization:
                student_loss = result[3].mean()
                teacher_loss = result[4].mean()
                student_tr_loss += student_loss.item()
                teacher_tr_loss += teacher_loss.item()

            loss = loss.mean()  # Needed in case of dataparallel
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            tr_loss += loss.item()
            # except:
            #     for i in range(4):
            #         print(batch_idx, batch[i].shape)
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                   args.clip)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                # Logging training steps
                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    # writer.add_scalar('avg_loss', avg_loss, step_num)
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'.format(
                        step_num, num_train_steps, epoch,
                        batch_idx + 1, len(loader_train), avg_loss))
                    wandb.log({"average loss": avg_loss, \
                    "learning_rate": optimizer.param_groups[0]['lr'], \
                    "epoch": epoch})
                    if args.distill_training or args.regularization:
                        avg_teacher_loss = (teacher_tr_loss-teacher_logging_loss)/args.logging_steps
                        avg_student_loss = (student_tr_loss-student_logging_loss)/args.logging_steps
                        wandb.log({"average teacher loss": avg_teacher_loss, \
                        "average student loss": avg_student_loss,\
                        "learning_rate": optimizer.param_groups[0]['lr'], \
                        "epoch": epoch})  
                        student_logging_loss = student_tr_loss
                        teacher_logging_loss = teacher_tr_loss                      
                    logging_loss = tr_loss


        if args.eval_method == "skip":
            pass
        else:
            # Evaluation after one training loop
            if args.eval_method == 'micro':
                val_result = micro_eval(model, loader_val, num_val_samples, args)
            elif args.eval_method == 'macro':
                val_result = macro_eval(model, loader_val, num_val_samples, args)
            logger.log('Done with epoch {:3d} | train loss {:8.4f} | '
                   'val acc unormalized  {:8.4f} ({}/{})|'
                   'val acc normalized  {:8.4f} ({}/{}) '.format(
            epoch,
            tr_loss / step_num,
            val_result['acc_unorm'],
            val_result['num_correct'],
            num_val_samples,
            val_result['acc_norm'],
            val_result['num_correct'],
            val_result['num_total_norm'],
            newline=False))
            wandb.log({"unnormalized acc": val_result['acc_unorm'], "normalized acc": val_result['acc_norm']
            ,"val_loss": val_result['val_loss'], "epoch": epoch\
            ,"micro_unnormalized_acc": sum(val_result['num_correct'])/sum(val_result['num_total_unorm'])\
            ,"micro_normalized_acc": sum(val_result['num_correct'])/sum(val_result['num_total_norm'])})

        # When accuracy is higher than best performance, save the model as 'pytorch_model.bin'
        if val_result['acc_unorm'] >= best_val_perf:
            triggert_times = 0 
            logger.log('      <----------New best val perf: %g -> %g' %
                       (best_val_perf, val_result['acc_unorm']))
            best_val_perf = val_result['acc_unorm']
            print("saved best val perf:", best_val_perf)
            torch.save({'opt': args, 'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss}, os.path.join(args.save_dir, "pytorch_model.bin"))
        torch.save({'opt': args, 'sd': model.module.state_dict() if dp else model.state_dict(),
        'perf': best_val_perf, 'epoch': epoch,
        'opt_sd': optimizer.state_dict(),
        'scheduler_sd': scheduler.state_dict(),
        'tr_loss': tr_loss, 'step_num': step_num,
        'logging_loss': logging_loss}, os.path.join(args.save_dir, "epoch_{}.bin".format(epoch)))
        # Only save the most recent file (for saving disk storage)
        if epoch >= 2:
            try:
                os.remove(os.path.join(args.save_dir, "epoch_{}.bin".format(epoch - 1)))
            except:
                pass
        else:
            trigger_times += 1

        wandb.log({"epoch": epoch, "best acc": best_val_perf})
        # For bi-encoder, check out validation recall for entire entity sets
        if args.type_model=="dual":
            logger.log('retriever performance test')
            logger.log('loading datasets')
            entity_path = "./data/entities"
            samples_train = torch.load(entity_path+"/samples_train.pt")
            samples_heldout_train_seen = torch.load(entity_path+"/samples_heldout_train_seen.pt")
            samples_heldout_train_unseen = torch.load(entity_path+"/samples_heldout_train_unseen.pt")
            samples_val = torch.load(entity_path+"/samples_val.pt")
            samples_test = torch.load(entity_path+"/samples_test.pt")
            train_doc = torch.load(entity_path+"/train_doc.pt")
            heldout_train_doc = torch.load(entity_path+"/heldout_train_doc.pt")
            val_doc = torch.load(entity_path+"/val_doc.pt")
            test_doc = torch.load(entity_path+"/test_doc.pt")
            logger.log('loading train entities')
            all_train_entity_token_ids = torch.load(entity_path+"/all_train_entity_token_ids.pt")
            all_train_masks = torch.load(entity_path+"/all_train_masks.pt")

            logger.log('loading val and test entities')
            all_val_entity_token_ids = torch.load(entity_path+"/all_val_entity_token_ids.pt")
            all_val_masks = torch.load(entity_path+"/all_val_masks.pt")
            all_test_entity_token_ids = torch.load(entity_path+"/all_test_entity_token_ids.pt")
            all_test_masks = torch.load(entity_path+"/all_test_masks.pt")

            data = Data(train_doc, val_doc, test_doc, tokenizer,
            all_train_entity_token_ids, all_train_masks,
            all_val_entity_token_ids, all_val_masks,
            all_test_entity_token_ids, all_test_masks, args.max_len,
            samples_train, samples_val, samples_test)
            _, val_en_loader, _, _, \
            val_men_loader, _ = data.get_loaders(args.mention_bsz, args.entity_bsz, False)
            model.attention_type = "hard_attention"
            model.num_mention_vecs = 1
            model.num_entity_vecs = 1
            model.evaluate_on = True
            all_val_cands_embeds = get_all_entity_hiddens(val_en_loader, model,
                                                args.store_en_hiddens,
                                                    args.en_hidden_path, debug = args.debug)
            eval_result = evaluate(val_men_loader, model, all_val_cands_embeds,
                            args.k, device, len(val_en_loader),
                            args.store_en_hiddens, args.en_hidden_path)

            logger.log(
                'validation recall {:8.4f}'
                '|validation accuracy {:8.4f}'.format(
                eval_result[0],
                eval_result[1]
                ))
            wandb.log({"retriever/val_recall": eval_result[0], "retriever/val_accuracy": eval_result[1]})
        
        if trigger_times > 5: break # Stopping rule for overfitting
        if epoch >= args.epochs: training_finished = True # Training is over
        if args.training_one_epoch: # Only training for one epoch (because of timelimit)
            break
    # Evaluation when training finished
    if training_finished:
        # Retriever Evaluation
        if args.type_model=="dual":
            logger.log('retriever performance test')
            logger.log('loading datasets')
            entity_path = "./data/entities"
            samples_train = torch.load(entity_path+"/samples_train.pt")
            samples_heldout_train_seen = torch.load(entity_path+"/samples_heldout_train_seen.pt")
            samples_heldout_train_unseen = torch.load(entity_path+"/samples_heldout_train_unseen.pt")
            samples_val = torch.load(entity_path+"/samples_val.pt")
            samples_test = torch.load(entity_path+"/samples_test.pt")
            train_doc = torch.load(entity_path+"/train_doc.pt")
            heldout_train_doc = torch.load(entity_path+"/heldout_train_doc.pt")
            val_doc = torch.load(entity_path+"/val_doc.pt")
            test_doc = torch.load(entity_path+"/test_doc.pt")
            logger.log('loading train entities')
            all_train_entity_token_ids = torch.load(entity_path+"/all_train_entity_token_ids.pt")
            all_train_masks = torch.load(entity_path+"/all_train_masks.pt")

            logger.log('loading val and test entities')
            all_val_entity_token_ids = torch.load(entity_path+"/all_val_entity_token_ids.pt")
            all_val_masks = torch.load(entity_path+"/all_val_masks.pt")
            all_test_entity_token_ids = torch.load(entity_path+"/all_test_entity_token_ids.pt")
            all_test_masks = torch.load(entity_path+"/all_test_masks.pt")

            data = Data(train_doc, val_doc, test_doc, tokenizer,
            all_train_entity_token_ids, all_train_masks,
            all_val_entity_token_ids, all_val_masks,
            all_test_entity_token_ids, all_test_masks, args.max_len,
            samples_train, samples_val, samples_test)
            _, val_en_loader, _, _, \
            val_men_loader, _ = data.get_loaders(args.mention_bsz, args.entity_bsz, False)
            model.attention_type = "hard_attention"
            model.num_mention_vecs = 1
            model.num_entity_vecs = 1
            model.evaluate_on = True
            all_val_cands_embeds = get_all_entity_hiddens(val_en_loader, model,
                                                args.store_en_hiddens,
                                                    args.en_hidden_path, debug = args.debug)
            eval_result = evaluate(val_men_loader, model, all_val_cands_embeds,
                            args.k, device, len(val_en_loader),
                            args.store_en_hiddens, args.en_hidden_path)

            logger.log(
                'validation recall {:8.4f}'
                '|validation accuracy {:8.4f}'.format(
                eval_result[0],
                eval_result[1]
                ))
            wandb.log({"retriever/val_recall": eval_result[0], "retriever/val_accuracy": eval_result[1]})
        if args.type_model != "full":
            print('recall eval')
            args.C_eval = 1024
            args.val_random_shuffle =True
            loader_train, loader_val, loader_test, \
            num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                        args.C, args.B,
                                                        args.num_workers,
                                                        args.inputmark,
                                                        args.C_eval,
                                                        use_full_dataset,
                                                        macro_eval_mode, args = args)

            beam_list = [0.0625, 0.25, 0.5]
            for beam in beam_list:
                args.beam_ratio = beam
                recall_result = recall_eval(model, loader_val, num_val_samples, eval_mode=args.eval_method, args=args)
                print(beam, recall_result)
                wandb_log = {"valid(tournament, beam {})/unnormalized acc".format(str(beam)): recall_result['acc_unorm'], \
                "valid(tournament, beam {})/recall".format(str(beam)): recall_result['recall']}
                wandb.log(wandb_log)
        loader_train, loader_val, loader_test, \
        num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                        args.C, args.B,
                                                        args.num_workers,
                                                        args.inputmark,
                                                        args.C_eval,
                                                        use_full_dataset,
                                                        macro_eval_mode, args = args)
        cpt = torch.load(os.path.join(args.save_dir,"pytorch_model.bin")) if device.type == 'cuda' \
            else torch.load(os.path.join(args.save_dir,"pytorch_model.bin") , map_location=torch.device('cpu'))
        print("checkpoint from ", cpt['epoch'])
        print("best performance ", cpt['perf'])
        if dp:
            state_dict = cpt['sd']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # remove `module.`
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict)
            except:
                new_state_dict = cpt['sd']
                modified_state_dict = OrderedDict()
                # Change dict keys for loading model parameter
                for k, v in new_state_dict.items():
                    if 'extend_multi.' in k:
                        name = k.replace('extend_multi.', 'module.extend_multi_')
                    else:
                        name = 'module.' + k
                    modified_state_dict[name] = v
                model.load_state_dict(modified_state_dict)
        else:
            try:
                model.load_state_dict(cpt['sd'])
            except:
                new_state_dict = cpt['sd']
                modified_state_dict = OrderedDict()
                # Change dict keys for loading model parameter
                for k, v in new_state_dict.items():
                    if 'extend_multi.' in k:
                        k = k.replace('extend_multi.', 'extend_multi_')
                    modified_state_dict[k] = v
                model.load_state_dict(modified_state_dict)
        if args.store_reranker_score:
            micro_eval(model, loader_train, num_val_samples, args, mode = 'train')

        print('start evaluation on val set (to check whether correct model loaded)')
        if not args.case_based:
            if args.eval_method == 'micro':
                val_result = micro_eval(model, loader_val, num_val_samples, args)
            elif args.eval_method == 'macro':
                val_result = macro_eval(model, loader_val, num_val_samples, args)
        else:
            if args.eval_method == 'micro':
                val_result = micro_eval(model, loader_val, num_val_samples, args)
            elif args.eval_method == 'macro':
                val_result = macro_eval(model, loader_val, num_val_samples, args)
        print(val_result)
        print('val acc unormalized  {:8.4f} ({}/{})|'
                    'val acc normalized  {:8.4f} ({}/{}) '.format(
                val_result['acc_unorm'],
                val_result['num_correct'],
                num_val_samples,
                val_result['acc_norm'],
                val_result['num_correct'],
                val_result['num_total_norm'],
                newline=False))
        wandb_log = {"valid(cands {})/unnormalized acc".format(str(args.C_eval)): val_result['acc_unorm'], \
                "valid(cands {})/normalized acc".format(str(args.C_eval)): val_result['acc_norm'],\
                "valid(cands {})/micro_unnormalized_acc".format(str(args.C_eval)): sum(val_result['num_correct'])/sum(val_result['num_total_unorm']),\
                "valid(cands {})/micro_normalized_acc".format(str(args.C_eval)): sum(val_result['num_correct'])/sum(val_result['num_total_norm']),\
                "valid(cands {})/mrr".format(str(args.C_eval)): val_result['mrr']}
        for i in range(len(depth)):
            wandb_log["valid(cands {})/recall@{}".format(str(args.C_eval), depth[i])] = val_result['recall'][i]
        print(wandb_log)
        wandb.log(wandb_log)

        print('start evaluation on test set')
        if loader_test is None:
            _, _, loader_test, \
            _, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                            args.C, args.B,
                                                            args.num_workers,
                                                            args.inputmark,
                                                            args.C_eval,
                                                            use_full_dataset,
                                                            macro_eval_mode, args = args)
        if args.eval_method == 'micro':
            test_result = micro_eval(model, loader_test, num_test_samples, args)
        elif args.eval_method == 'macro':
            test_result = macro_eval(model, loader_test, num_test_samples, args)

        logger.log('\nDone training | training time {:s} | '
                'test acc unormalized {:8.4f} ({}/{})|'
                'test acc normalized {:8.4f} ({}/{})'.format(str(
        datetime.now() - start_time),
        test_result['acc_unorm'],
        test_result['num_correct'],
        num_test_samples,
        test_result['acc_norm'],
        test_result['num_correct'],
        test_result['num_total_norm']))

        wandb.log({"test_unnormalized acc": test_result['acc_unorm'], "test_normalized acc": test_result['acc_norm']})

        print('start evaluation on >64 candidates')
        C_eval_list = [128,256,512]
        for C_eval_elem in C_eval_list:
            print("C_eval", C_eval_elem)
            args.C_eval = C_eval_elem
            loader_train, loader_val, loader_test, \
            num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                        args.C, args.B,
                                                        args.num_workers,
                                                        args.inputmark,
                                                        args.C_eval,
                                                        use_full_dataset,
                                                        macro_eval_mode, args = args)
            if args.distill:
                train_result = micro_eval(model, loader_train, num_val_samples, args, mode = "train")
                print(train_result)
            if args.eval_method == 'micro':
                val_result = micro_eval(model, loader_val, num_val_samples, args)
            elif args.eval_method == 'macro':
                val_result = macro_eval(model, loader_val, num_val_samples, args)
            print('C_eval: {}'.format(C_eval_elem))
            print(val_result)
            print('val acc unormalized  {:8.4f} ({}/{})|'
                    'val acc normalized  {:8.4f} ({}/{})|'.format(
                val_result['acc_unorm'],
                val_result['num_correct'],
                num_val_samples,
                val_result['acc_norm'],
                val_result['num_correct'],
                val_result['num_total_norm'],
                newline=False))
            wandb_log = {"valid(cands {})/unnormalized acc".format(str(C_eval_elem)): val_result['acc_unorm'], \
                "valid(cands {})/normalized acc".format(str(C_eval_elem)): val_result['acc_norm'],\
                "valid(cands {})/micro_unnormalized_acc".format(str(C_eval_elem)): sum(val_result['num_correct'])/sum(val_result['num_total_unorm']),\
                "valid(cands {})/micro_normalized_acc".format(str(C_eval_elem)): sum(val_result['num_correct'])/sum(val_result['num_total_norm']),\
                "valid(cands {})/mrr".format(str(args.C_eval)): val_result['mrr']}
            for i in range(len(depth)):
                wandb_log["valid(cands {})/recall@{}".format(str(C_eval_elem), depth[i])] = val_result['recall'][i]
            print(wandb_log)
            wandb.log(wandb_log)


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

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='coefficient for multi-loss')
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
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num workers [%(default)d]')
    parser.add_argument('--k', type=int, default=64,
                        help='recall@k when evaluate')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--case_based', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--gold_first', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--test_at_first', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--order_changed', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--attend_to_gold', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--attend_to_itself', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--batch_first', action='store_false',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--nearest', action='store_true',
                        help='vertical interaction w/ nearest neighbors')
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
    parser.add_argument('--store_reranker_score', action = 'store_true',
                        help='debugging mode')
    parser.add_argument('--save_dir', type = str, default = '/shared/s3/lab07/jongsong/hard-nce-el_garage/models',
                        help='debugging mode')
    parser.add_argument('--training_one_epoch', action = 'store_true',
                        help='stop the training after one epoch')
    parser.add_argument('--type_cands', type=str,
                        default='fixed_negative',
                        choices=['fixed_negative',
                                'mixed_negative',
                                 'hard_negative',
                                 'self_negative',
                                'random_negative',
                                 'self_fixed_negative',
                                 'self_mixed_negative'],
                        help='fixed: top k, hard: distributionally sampled k')
    parser.add_argument('--mention_bsz', type=int, default=512,
                        help='the batch size')

    parser.add_argument('--run_id', type = str,
                        help='run id when resuming the run')
    parser.add_argument('--training_finished', action = 'store_true',
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
    parser.add_argument('--num_mention_vecs', type=int, default=1,
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
    parser.add_argument('--BCELoss', action='store_true',
                        help='getting score distribution for distillation purpose')
    parser.add_argument('--distill_training', action='store_true',
                        help='training smaller model from crossencoder scores')
    parser.add_argument('--energy_model', action='store_true',
                        help='regularize by bi-encoder objective')
    parser.add_argument('--regularization', action='store_true',
                        help='regularize by bi-encoder score distribution')
    parser.add_argument('--bert_lr', type=float, default=1e-5,
                        help='the batch size')
    parser.add_argument('--weight_lr', type=float, default=1e-2,
                        help='the batch size')
    parser.add_argument('--num_sampled', type=int, default=1024,
                        help='train set candidates to get distribution from')
    parser.add_argument('--max_len', type=int, default=128,
                        help='max length of the mention input '
                            'and the entity input')
    parser.add_argument('--final_k', type=int, default=64,
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
    if args.nearest: assert args.batch_first == False and args.type_cands == "fixed_negative"
    main(args)
