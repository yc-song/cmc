import torch
from retriever import UnifiedRetriever
from data_retriever import ZeshelDataset, transform_entities, load_data, \
    get_all_entity_hiddens, get_hard_negative, \
    Data
from util import Logger
from data_reranker import get_loaders, Logger,  load_zeshel_data
# from main_reranker import micro_eval, macro_eval/
from collections import OrderedDict

import argparse
import numpy as np
import os
import random
import wandb
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule, RobertaTokenizer, \
    RobertaModel
from torch.utils.data import DataLoader
from tqdm import tqdm

depth = [2,4,8,16,32,64]
def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def configure_optimizer(args, model, num_train_examples):
    # https://github.com/google-research/bert/blob/master/optimization.py#L25
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,
                      eps=args.adam_epsilon)

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


def evaluate(mention_loader, model, all_candidates_embeds, k, device,
             len_en_loader,
             too_large=False, en_hidden_path=None):
    model.eval()
    if not too_large:
        if hasattr(model, 'module'):
            model.module.evaluate_on = True
            model.module.candidates_embeds = all_candidates_embeds
        else:
            model.evaluate_on = True
            model.candidates_embeds = all_candidates_embeds
    nb_samples = 0


    r_k = [0 for d in depth]
    acc = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(mention_loader), total = len(mention_loader)):
            if not too_large:
                scores = model(batch[0], batch[1], None, None)
            else:
                scores = []
                for j in tqdm(range(len_en_loader), total = len_en_loader):
                    file_path = os.path.join(en_hidden_path,
                                             'en_hiddens_%s.pt' % j)
                    en_embeds = torch.load(file_path)
                    if hasattr(model, 'module'):
                        model.module.evaluate_on = True
                        model.module.candidates_embeds = en_embeds
                    else:
                        model.evaluate_on = True
                        model.candidates_embeds = en_embeds
                    score = model(batch[0], batch[1], None,
                                  None)['scores'].detach()
                    scores.append(score)
                scores = torch.cat(scores, dim=1)
            labels = batch[2].to(device)
            for i in range(len(depth)):
                top_k = scores.topk(depth[i], dim=1)[1]
                preds = top_k[:, 0]
                r_k[i]+=(top_k == labels.to(device)).sum().item()
            nb_samples += scores.size(0)
            acc += (preds == labels.squeeze(1).to(device)).sum().item()
    for i in range(len(r_k)):
        r_k[i] /= nb_samples
    acc /= nb_samples
    if hasattr(model, 'module'):
        model.module.evaluate_on = False
        model.module.candidates_embeds = None
    else:
        model.evaluate_on = False
        model.candidates_embeds = None
    return r_k, acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    set_seeds(args)
    if args.energy_model: assert args.type_model == "dual"
    if args.resume_training:
        run = wandb.init(project = "hard-nce-el", resume = "must", id = args.run_id, config = args)
    else:
        run = wandb.init(project="hard-nce-el", config = args)

    # configure logger
    args.save_dir = '{}/{}/{}/'.format(args.save_dir, args.type_model, run.id)
    args.en_hidden_path = './data/{}/{}/'.format(args.type_model, run.id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.en_hidden_path):
        os.makedirs(args.en_hidden_path)
    logger = Logger(args.save_dir + '.log', True)
    logger.log(str(args))
    # load data and initialize model and dataset
    entity_path = "./data/entities"
    try: 
        logger.log('loading datasets')
        samples_train = torch.load(entity_path+"/samples_train.pt")
        samples_heldout_train_seen = torch.load(entity_path+"/samples_heldout_train_seen.pt")
        samples_heldout_train_unseen = torch.load(entity_path+"/samples_heldout_train_unseen.pt")
        samples_val = torch.load(entity_path+"/samples_val.pt")
        samples_test = torch.load(entity_path+"/samples_test.pt")
        train_doc = torch.load(entity_path+"/train_doc.pt")
        heldout_train_doc = torch.load(entity_path+"/heldout_train_doc.pt")
        val_doc = torch.load(entity_path+"/val_doc.pt")
        test_doc = torch.load(entity_path+"/test_doc.pt")

    except:
        samples_train, samples_heldout_train_seen, \
        samples_heldout_train_unseen, samples_val, samples_test, \
        train_doc, heldout_train_doc, heldout_train_unseen_doc, \
        heldout_train_unseen_doc, val_doc, test_doc = load_data(
            args.data_dir)
        logger.log('loading datasets')
        torch.save(samples_train, entity_path+"/samples_train.pt")
        torch.save(samples_heldout_train_seen, entity_path+"/samples_heldout_train_seen.pt")
        torch.save(samples_heldout_train_unseen, entity_path+"/samples_heldout_train_unseen.pt")
        torch.save(samples_val, entity_path+"/samples_val.pt")
        torch.save(samples_test, entity_path+"/samples_test.pt")
        torch.save(train_doc, entity_path+"/train_doc.pt")
        torch.save(heldout_train_doc, entity_path+"/heldout_train_doc.pt")
        torch.save(val_doc, entity_path+"/val_doc.pt")
        torch.save(test_doc, entity_path+"/test_doc.pt")


        
    num_rands = int(args.num_cands * args.cands_ratio)
    num_hards = args.num_cands - num_rands
    # Wandb init
    # get model and tokenizer
    if args.pre_model == 'Bert':
        if args.type_bert == 'base':
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            encoder = BertModel.from_pretrained('bert-base-uncased')
        elif args.type_bert == 'large':
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            encoder = BertModel.from_pretrained('bert-large-uncased')
    elif args.pre_model == 'Roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        encoder = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError('wrong encoder type')
    # encoder=MLPEncoder(args.max_len)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.type_model == 'poly':
        attention_type = 'soft_attention'
    elif args.type_model == 'extend_multi':
        attention_type = 'extend_multi'
    elif args.type_model == 'extend_multi_dot':
        attention_type = 'extend_multi_dot'
    else:
        attention_type = 'hard_attention'
    if args.type_model == 'dual':
        num_mention_vecs = 1
        num_entity_vecs = 1
    elif args.type_model == 'multi_vector':
        num_mention_vecs = 1
        num_entity_vecs = args.num_entity_vecs
    elif args.type_model == 'extend_multi':
        num_mention_vecs = 1
        num_entity_vecs = 1
    elif args.type_model == 'extend_multi_dot':
        num_mention_vecs = 1
        num_entity_vecs = 1
    else:
        num_mention_vecs = args.num_mention_vecs
        num_entity_vecs = args.num_entity_vecs
    model = UnifiedRetriever(encoder, device, num_mention_vecs, num_entity_vecs,
                            args.mention_use_codes, args.entity_use_codes,
                            attention_type, None, False, num_heads = args.num_heads, num_layers = args.num_layers, args = args)
    if args.resume_training:
        cpt = torch.load(args.save_dir+"pytorch_model.bin") if device.type == 'cuda' \
            else torch.load(args.save_dir+"pytorch_model.bin", map_location=torch.device('cpu'))
        model.load_state_dict(cpt['sd'])
    dp = torch.cuda.device_count() > 1

    if args.anncur:
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
        model.load_state_dict(modified_state_dict)
    elif args.blink:
        package = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
        modified_state_dict = OrderedDict()
        # Change dict keys for loading model parameter
        position_ids = ["mention_encoder.embeddings.position_ids", "entity_encoder.embeddings.position_ids"]
        for k, v in package.items():
            name = None
            if k.startswith('cand_encoder'):
                name = k.replace('cand_encoder.bert_model', 'entity_encoder')
            elif k.startswith('context_encoder'):
                name = k.replace('context_encoder.bert_model', 'mention_encoder')
            if name is not None:
                modified_state_dict[name] = v
        model.load_state_dict(modified_state_dict, strict = False)
    elif args.cocondenser:
        state_dict = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name_context = 'mention_encoder.'+k
            name_candidate = 'entity_encoder.'+k
            new_state_dict[name_context] = v
            new_state_dict[name_candidate] = v
        model.load_state_dict(new_state_dict, strict = False)


    if args.retriever_path:
        if args.retriever_type == 'poly':
            retriever_attention_type = 'soft_attention'
        else:
            retriever_attention_type = 'hard_attention'
        if args.retriever_type == 'dual':
            retriever_num_mention_vecs = 1
            retriever_num_entity_vecs = 1
        elif args.retriever_type == 'multi_vector':
            retriever_num_mention_vecs = 1
            retriever_num_entity_vecs = args.num_entity_vecs
        else:
            retriever_num_mention_vecs = args.num_mention_vecs
            retriever_num_entity_vecs = args.num_entity_vecs
        retriever_model = UnifiedRetriever(encoder, device, retriever_num_mention_vecs, retriever_num_entity_vecs,
                                args.mention_use_codes, args.entity_use_codes,
                                retriever_attention_type, None, False)
        retriever_cpt = torch.load(args.retriever_path) if device.type == 'cuda' \
            else torch.load(args.retriever_path, map_location=torch.device('cpu'))
        retriever_model.load_state_dict(retriever_cpt['sd'])
        if dp:
            retriever_model = nn.DataParallel(retriever_model)
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    
    model.to(device)
    if args.retriever_path is not None:
        retriever_model.to(device)

    logger.log('transform train entities')
    # if args.debug:
    try: 
        logger.log('loading train entities')
        all_train_entity_token_ids = torch.load(entity_path+"/all_train_entity_token_ids.pt")
        all_train_masks = torch.load(entity_path+"/all_train_masks.pt")
    except:
        logger.log('failed to load train entities')
        all_train_entity_token_ids, all_train_masks = transform_entities(train_doc,
                                                                        args.max_len,
                                                                        tokenizer)
        logger.log("Saving train entities to file " )
        os.makedirs(entity_path, exist_ok = True)
        torch.save(all_train_entity_token_ids, entity_path+"/all_train_entity_token_ids.pt")
        torch.save(all_train_masks, entity_path+"/all_train_masks.pt")

        logger.log('transform valid and test entities')
    try: 

        logger.log('loading val and test entities')
        all_val_entity_token_ids = torch.load(entity_path+"/all_val_entity_token_ids.pt")
        all_val_masks = torch.load(entity_path+"/all_val_masks.pt")
        all_test_entity_token_ids = torch.load(entity_path+"/all_test_entity_token_ids.pt")
        all_test_masks = torch.load(entity_path+"/all_test_masks.pt")
    except:
        logger.log('failed to load train entities')
        all_val_entity_token_ids, all_val_masks = transform_entities(val_doc,
                                                                        args.max_len,
                                                                        tokenizer)
        all_test_entity_token_ids, all_test_masks = transform_entities(test_doc,
                                                                        args.max_len,
                                                                        tokenizer)
        logger.log("Saving val and test entities to file ")
        torch.save(all_val_entity_token_ids, entity_path+"/all_val_entity_token_ids.pt")
        torch.save(all_val_masks, entity_path+"/all_val_masks.pt")
        torch.save(all_test_entity_token_ids, entity_path+"/all_test_entity_token_ids.pt")
        torch.save(all_test_masks, entity_path+"/all_test_masks.pt")
    # # else:
    #     logger.log('loading train entities')
    #     all_train_entity_token_ids, all_train_masks = transform_entities(train_doc,train_doc
    #                                                                         args.max_len,
    #                                                                         tokenizer)
    #     logger.log('loading val and test entities')
    #     all_val_entity_token_ids, all_val_masks = transform_entities(val_doc,
    #                                                                         args.max_len,
    #                                                                         tokenizer)
    #     all_test_entity_token_ids, all_test_masks = transform_entities(test_doc,
    #                                                                         args.max_len,
    #                                                                         tokenizer)
    data = Data(train_doc, val_doc, test_doc, tokenizer,
                all_train_entity_token_ids, all_train_masks,
                all_val_entity_token_ids, all_val_masks,
                all_test_entity_token_ids, all_test_masks, args.max_len,
                samples_train, samples_val, samples_test)
    train_en_loader, val_en_loader, test_en_loader, train_men_loader, \
    val_men_loader, test_men_loader = data.get_loaders(args.mention_bsz,
                                                       args.entity_bsz, False)
    model.train()

    # configure optimizer
    num_train_samples = len(samples_train)
    if args.simpleoptim:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer_simple(args, model, num_train_samples)
    else:
        optimizer, scheduler, num_train_steps, num_warmup_steps \
            = configure_optimizer(args, model, num_train_samples)
    if args.resume_training:
        optimizer.load_state_dict(cpt['opt_sd'])
        scheduler.load_state_dict(cpt['scheduler_sd'])
    # train
    logger.log('***** train *****')
    logger.log('# train samples: {:d}'.format(num_train_samples))
    logger.log('# epochs: {:d}'.format(args.epochs))
    logger.log(' batch size: {:d}'.format(args.B))
    logger.log(' gradient accumulation steps {:d}'
               ''.format(args.gradient_accumulation_steps))
    logger.log(
        ' effective training batch size with accumulation: {:d}'
        ''.format(args.B * args.gradient_accumulation_steps))
    logger.log(' # training steps: {:d}'.format(num_train_steps))
    logger.log(' # warmup steps: {:d}'.format(num_warmup_steps))
    logger.log(' learning rate: {:g}'.format(args.lr))
    logger.log(' # parameters: {:d}'.format(count_parameters(model)))
    # start_time = datetime.now()
    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
    model.zero_grad()
    best_val_perf = float('-inf')
    start_epoch = 1
    if args.resume_training:
        start_epoch = cpt['epoch'] + 1
    print("epochs")
    for epoch in tqdm(range(start_epoch, args.epochs + 1)):
        logger.log('\nEpoch {:d}'.format(epoch))
        if args.type_cands == 'hard_adjusted_negative':
            distribution_sampling = True
            adjust_logits = True
            num_cands = args.num_cands
        elif args.type_cands == 'hard_negative':
            distribution_sampling = True
            adjust_logits = False
            num_cands = args.num_cands
        elif args.type_cands == 'mixed_negative':
            distribution_sampling = False
            adjust_logits = False
            num_cands = num_hards
        else:
            raise ValueError('type candidates wrong')
        if args.type_cands == 'mixed_negative' and num_hards == 0:
            candidates = None
        else:
            if args.retriever_path:
                all_train_cands_embeds = get_all_entity_hiddens(train_en_loader,
                                                                retriever_model,
                                                                args.store_en_hiddens,
                                                                args.en_hidden_path, debug = args.debug)
                candidates = get_hard_negative(train_men_loader, retriever_model, num_cands,
                                            len(train_en_loader), device,
                                            distribution_sampling,
                                            args.exclude_golds,
                                            args.store_en_hiddens,
                                            all_train_cands_embeds,
                                            args.en_hidden_path,
                                            adjust_logits, args.smoothing_value, debug = args.debug)
            else:
                all_train_cands_embeds = get_all_entity_hiddens(train_en_loader,
                                                                model,
                                                                args.store_en_hiddens,
                                                                args.en_hidden_path, args.debug)
                candidates = get_hard_negative(train_men_loader, model, num_cands,
                                            len(train_en_loader), device,
                                            distribution_sampling,
                                            args.exclude_golds,
                                            args.store_en_hiddens,
                                            all_train_cands_embeds,
                                            args.en_hidden_path,
                                            adjust_logits, args.smoothing_value, debug = args.debug)

        train_set = ZeshelDataset(tokenizer, samples_train, train_doc,
                                  args.max_len,
                                  candidates, device, num_rands,
                                  args.type_cands,
                                  all_train_entity_token_ids,
                                  all_train_masks, args)
        
        train_loader = DataLoader(train_set, args.B, shuffle=True,
                                  drop_last=False)
        if args.retriever_path:
            retriever_train_set = ZeshelDataset(tokenizer, samples_train, train_doc,
                                  args.max_len,
                                  candidates, device, num_rands,
                                  args.type_cands,
                                  all_train_entity_token_ids,
                                  all_train_masks, args, True)
        
            retriever_train_loader = DataLoader(train_set, args.B, shuffle=True,
                                  drop_last=False)
        logger.log("Start Training")

        for step, batch in tqdm(enumerate(train_loader), total = len(train_loader)):
            if args.debug and step > 10:
                break
            model.train()
            loss = model(*batch, args = args)['loss']

            if len(args.gpus) > 1:
                loss = loss.mean()
            loss.backward()
            tr_loss += loss.item()
            # print('loss is %f' % loss.item())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                step_num += 1

                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'
                               ''.format(step_num, num_train_steps,
                                         epoch, step + 1,
                                         len(train_loader), avg_loss))
                    logging_loss = tr_loss
                    wandb.log({"loss": avg_loss, "epoch": epoch})
        if args.energy_model:
            model.attention_type="dual"
            model.num_mention_vecs = 1
            model.num_entity_vecs = 1
        logger.log("get all entity hiddens")
        all_val_cands_embeds = get_all_entity_hiddens(val_en_loader, model,
                                                    args.store_en_hiddens,
                                                    args.en_hidden_path, debug = args.debug)
        logger.log("evaluate on val set")
        eval_result = evaluate(val_men_loader, model, all_val_cands_embeds,
                            args.k, device, len(val_en_loader),
                            args.store_en_hiddens, args.en_hidden_path)
    
        logger.log('Done with epoch {:3d} | '
                'validation recall {:8.4f}'
                '|validation accuracy {:8.4f}'.format(
            epoch,
            eval_result[0],
            eval_result[1]
        ))
        wandb.log({"val_recall": eval_result[0], "val_accuracy": eval_result[1], "epoch": epoch})
    
        # model.evaluate_on = False
        # logger.log('evaluation starts')
        # if args.eval_method == 'micro':
        #     val_result = micro_eval(model, loader_val, num_val_samples, debug = args.debug)
        # else:
        #     val_result = macro_eval(model, loader_val, num_val_samples)
        # logger.log('Done with epoch {:3d} | '
        #         'val acc unormalized  {:8.4f} ({}/{})|'
        #         'val acc normalized  {:8.4f} ({}/{}) '.format(
        #     epoch,
        #     val_result['acc_unorm'],
        #     val_result['num_correct'],
        #     num_val_samples,
        #     val_result['acc_norm'],
        #     val_result['num_correct'],
        #     val_result['num_total_norm'],
        #     newline=False))
        # wandb.log({"unnormalized accuracy": val_result['acc_unorm'] , "normalized acc": val_result['acc_norm'], "epoch": epoch})


        #  eval_train_result = evaluate(train_loader, model, args.k,device)[0]
        logger.log("save the model")
        torch.save({'opt': args,
                        'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss},
                       os.path.join(args.save_dir, "pytorch_model.bin"))


        # update dataset and dataloader
        # torch.cuda.empty_cache()

        # torch.cuda.empty_cache()
    # test model on test dataset

    package = torch.load(os.path.join(args.save_dir, "pytorch_model.bin")) if device.type == 'cuda' \
        else torch.load(os.path.join(args.save_dir, "pytorch_model.bin"), map_location=torch.device('cpu'))
    new_state_dict = package['sd']
    # encoder=MLPEncoder(args.max_len)
    if args.pre_model == 'Bert':
        encoder = BertModel.from_pretrained('bert-base-uncased')
    elif args.pre_model == 'Roberta':
        encoder = RobertaModel.from_pretrained('roberta-base')
    else:
        raise ValueError('wrong encoder type')
    model = UnifiedRetriever(encoder, device, num_mention_vecs, num_entity_vecs,
                             args.mention_use_codes, args.entity_use_codes,
                             attention_type, args=args)
    model.load_state_dict(new_state_dict)
    if dp:
        logger.log('Data parallel across {:d} GPUs {:s}'
                   ''.format(len(args.gpus.split(',')), args.gpus))
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print("***get_all_entity_hiddens***")


    # logger.log("evaluate on val set")

    # all_val_cands_embeds = get_all_entity_hiddens(val_en_loader, model,
    #                                             args.store_en_hiddens,
    #                                             args.en_hidden_path, debug = args.debug)
    # eval_result = evaluate(val_men_loader, model, all_val_cands_embeds,
    #                     args.k, device, len(val_en_loader),
    #                     args.store_en_hiddens, args.en_hidden_path)

    # logger.log('Training finished | train loss {:8.4f} | '
    #         'validation recall {:8.4f}'
    #         '|validation accuracy {:8.4f}'.format(
    #     tr_loss / step_num,
    #     eval_result[0],
    #     eval_result[1]
    # ))
    # wandb.log({"best_val_recall": eval_result[0], "best_val_accuracy": eval_result[1]})
    print("***test_result***")
    all_test_cands_embeds = get_all_entity_hiddens(test_en_loader, model,
                                                args.store_en_hiddens,
                                                args.en_hidden_path)
    test_result = evaluate(test_men_loader, model, all_test_cands_embeds,
                        args.k, device, len(test_en_loader),
                        args.store_en_hiddens, args.en_hidden_path)
    print(test_result)
    wandb.log({"test_accuracy": test_result[1]})
    for i in range(len(depth)):
        wandb.log({f"test_recall@{depth[i]}": test_result[0][i]})
    # if args.eval_method == 'micro':
    #     test_result = micro_eval(model, loader_test, num_test_samples)
    # else:
    #     test_result = macro_eval(model, loader_test, num_test_samples)


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
#    test_train_result = evaluate(train_loader, model,args)
#   logger.log('test train acc {:f}'.format(test_train_result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        help='model path')
    parser.add_argument('--resume_training', action='store_true',
                        help='resume training from checkpoint?')
    parser.add_argument('--run_id', type = str,
                        help='run id when resuming the run')
    parser.add_argument('--retriever_path', type = str,
                        help='retriever path for full and extend_multi')
    parser.add_argument('--retriever_type', type = str,
                        help='retriever path for full and extend_multi')
    parser.add_argument('--max_len', type=int, default=128,
                        help='max length of the mention input '
                             'and the entity input')
    parser.add_argument('--num_hards', type=int, default=10,
                        help='the number of the nearest neighbors we use to '
                             'construct hard negatives')
    parser.add_argument('--type_cands', type=str,
                        default='mixed_negative',
                        choices=['mixed_negative',
                                 'hard_negative',
                                 'hard_adjusted_negative'],
                        help='the type of negative we use during training')
    parser.add_argument('--data_dir', type=str,
                        help='the  data directory')
    parser.add_argument('--L', type=int, default=256,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--save_dir', type = str, default = './models',
                        help='debugging mode')
    parser.add_argument('--C', type=int, default=64,
                        help='max number of candidates [%(default)d]')
    parser.add_argument('--C_eval', type=int, default=64,
                        help='max number of candidates [%(default)d] when eval')
    parser.add_argument('--cands_dir', type=str,
                        help='candidates directory')
    parser.add_argument('--debug', action = 'store_true',
                        help='debugging mode')
    parser.add_argument('--eval_method', default='macro', type=str,
                        choices=['macro', 'micro'],
                        help='the evaluate method')
    parser.add_argument('--inputmark', action='store_true',
                        help='use mention mark at input?')
    parser.add_argument('--B', type=int, default=128,
                        help='the batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='the learning rate')
    parser.add_argument('--bert_lr', type=float, default=1e-5,
                        help='the batch size')
    parser.add_argument('--epochs', type=int, default=3,
                        help='the number of training epochs')
    parser.add_argument('--k', type=int, default=64,
                        help='recall@k when evaluate')
    parser.add_argument('--too_large', action = 'store_true',
                    help='too large to evaluate on batch')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear '
                             'learning rate warmup for [%(default)g]')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay [%(default)g]')
    parser.add_argument('--adam_epsilon', type=float, default=1e-6,
                        help='epsilon for Adam optimizer [%(default)g]')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='num gradient accumulation steps [%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers [%(default)d]')
    parser.add_argument('--distill', action='store_true',
                        help='getting score distribution for distillation purpose')
    parser.add_argument('--distill_training', action='store_true',
                        help='training smaller model from crossencoder scores')
    parser.add_argument('--simpleoptim', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help='num logging steps [%(default)d]')
    parser.add_argument('--gpus', default='0,1', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--eval_criterion', type=str, default='recall',
                        choices=['recall', 'accuracy'],
                        help='the criterion for selecting model')
    parser.add_argument('--pre_model', default='Bert',
                        choices=['Bert', 'Roberta'],
                        type=str, help='the encoder for train')
    parser.add_argument('--cands_ratio', default=1.0, type=float,
                        help='the ratio between random candidates and hard '
                             'candidates')
    parser.add_argument('--num_cands', default=128, type=int,
                        help='the total number of candidates')
    parser.add_argument('--smoothing_value', default=1, type=float,
                        help=' smoothing factor when sampling negatives '
                             'according to model distribution')
    parser.add_argument('--eval_batch_size', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--mention_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--anncur', action='store_true', help = "load anncur ckpt")
    parser.add_argument('--identity_bert', action = 'store_true',
                        help='assign same query and key matrix')
    parser.add_argument('--fixed_initial_weight', action='store_true',
                        help='fixed weight for identity weight')
    parser.add_argument('--entity_bsz', type=int, default=512,
                        help='the batch size')
    parser.add_argument('--exclude_golds', action='store_true',
                        help='exclude golds when sampling?')
    parser.add_argument('--type_model', type=str,
                        default='dual',
                        choices=['dual',
                                 'sum_max',
                                 'multi_vector',
                                 'poly',
                                 'extend_multi',
                                 'extend_multi_dot',
                                 'full'],
                        help='the type of model')
    parser.add_argument('--type_bert', type=str,
                        default='base',
                        choices=['base', 'large'],
                        help='the type of encoder')
    parser.add_argument('--token_type', action='store_false', help = "no token type id when specified")
    parser.add_argument('--energy_model', action='store_true',
                        help='regularize by bi-encoder objective')
    parser.add_argument('--model_top', type=str,
                        help='when loading top model path')
    parser.add_argument('--attend_to_gold', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--cocondenser', action='store_true', help = "load cocondenser ckpt")

    parser.add_argument('--blink', action = 'store_true',
                        help='load blink checkpoint')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='the number of multi-head attention in extend_multi ')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='the number of atten                ion layers in extend_multi ')
    parser.add_argument('--num_mention_vecs', type=int, default=8,
                        help='the number of mention vectors  ')
    parser.add_argument('--num_entity_vecs', type=int, default=8,
                        help='the number of entity vectors  ')
    parser.add_argument('--mention_use_codes', action='store_true',
                        help='use codes for mention embeddings?')
    parser.add_argument('--entity_use_codes', action='store_true',
                        help='use codes for entity embeddings?')
    parser.add_argument('--en_hidden_path', type=str,
                        help='all entity hidden states path')
    parser.add_argument('--store_en_hiddens', action='store_true',
                        help='store entity hiddens?')
    parser.add_argument('--batch_first', action='store_false',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--case_based', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    parser.add_argument('--gold_first', action='store_true',
                        help='simple optimizer (constant schedule, '
                             'no weight decay?')
    args = parser.parse_args()
    # Set environment variables before all else.
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus  # Sets torch.cuda behavior
    main(args)