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
            if debug and step > 30: break
            # try:
            result = model.forward(*batch, args = args)
            if type(result) is dict:
                preds = result['predictions']
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

        with torch.no_grad(): 
            for i, batch in tqdm(enumerate(val_loaders), total = len(val_loaders)):
                if i > 3 and debug: break
                scores = model.forward(*batch, recall_eval = True, beam_ratio = args.beam_ratio)[2]
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
        return {"recall": r_k, "acc_norm": acc_norm, "acc_unorm":acc_unorm, "num_total_norm": nb_samples, "num_correct": num_correct, "recall_correct": recall_correct, "num_total_norm": nb_samples}
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
            micro_results = micro_recall_eval(model, val_loader, num_unorm, k, all_candidates_embeds, args)
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
                                 attention_type, None, False, args = args).to(device)
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
    set_seed(args)
    if args.resume_training:
        run = wandb.init(project = "hard-nce-el", resume = "must", id = args.run_id, config = args)
    else:
        run = wandb.init(project="hard-nce-el", config = args)
    if not args.anncur or args.resume_training: args.model = './models/{}/{}/'.format(args.type_model, run.id)
    if not os.path.exists(args.model):
        os.makedirs(args.model)
    # writer = SummaryWriter()
    logger = Logger(args.model + '.log', True)
    logger.log(str(args))
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
        model = FullRanker(encoder, device)
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
                                 attention_type, None, False,args, num_heads = args.num_heads, num_layers = args.num_layers)
    if args.anncur and not args.resume_training:
        package = torch.load(args.model) if device.type == 'cuda' \
        else torch.load(args.model, map_location=torch.device('cpu'))
        new_state_dict = package['state_dict']
        modified_state_dict = OrderedDict()
        for k, v in new_state_dict.items():
            if k.startswith('model.input_encoder'):
                name = k.replace('model.input_encoder.bert_model', 'mention_encoder')
            elif k.startswith('model.label_encoder'):
                name = k.replace('model.label_encoder.bert_model', 'entity_encoder')
            modified_state_dict[name] = v
        model.load_state_dict(modified_state_dict, strict = False)
        if args.model_top is not None:
            package = torch.load(args.model_top) if device.type == 'cuda' \
            else torch.load(args.model_top, map_location=torch.device('cpu'))
            new_state_dict = package['model_state_dict']
            modified_state_dict = OrderedDict()
            for k, v in new_state_dict.items():
                name = k.replace('model', 'extend_multi')
                modified_state_dict[name] = v
            model.load_state_dict(modified_state_dict, strict = False)
        args.model = './models/{}/{}/'.format(args.type_model, run.id)
        if not os.path.exists(args.model):
            os.makedirs(args.model)

    if args.resume_training:
        count = 0
        for each_file_name in os.listdir(args.model):
            if each_file_name.startswith('epoch'):
                if count == 1: raise('More than two bin files found in the model directory')
                cpt=torch.load(os.path.join(args.model,each_file_name)) if device.type == 'cuda' \
                else torch.load(os.path.join(args.model,each_file_name) , map_location=torch.device('cpu'))
                count+=1
                print(os.path.join(args.model,each_file_name))
        model.load_state_dict(cpt['sd'])
    dp = torch.cuda.device_count() > 1
    if dp:
        logger.log('Data parallel across %d GPUs: %s' %
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
    if args.resume_training:
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

    step_num = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    best_val_perf = float('-inf')
    if args.resume_training:
        step_num = cpt['step_num']
        tr_loss, logging_loss = cpt['tr_loss'], cpt['logging_loss']
        best_val_perf = cpt['perf']
    start_epoch = 1
    if args.resume_training:
        start_epoch = cpt['epoch'] + 1 
    for epoch in range(start_epoch, args.epochs + 1):
        logger.log('\nEpoch %d' % epoch)
        loader_train, loader_val, loader_test, \
        num_val_samples, num_test_samples = get_loaders(data, tokenizer, args.L,
                                                    args.C, args.B,
                                                    args.num_workers,
                                                    args.inputmark,
                                                    args.C_eval,
                                                    use_full_dataset,
                                                    macro_eval_mode, args = args)
        if args.type_cands == "self_negative":
            torch.multiprocessing.set_start_method('spawn')
            scores = torch.tensor([]).to(device)
            model.eval()
            with torch.no_grad():
                print("spot 1")
                for i, batch in tqdm(enumerate(loader_train), total=len(loader_train)):
                    score = model.forward(*batch, recall_eval=True, beam_ratio=args.beam_ratio, args=args)[2]
                    scores = torch.cat((scores, score), dim = 0)
            print("spot 2")

            data = load_zeshel_data(args.data, args.cands_dir, macro_eval_mode, args.debug, scores = scores)

            loader_train, _, _, \
            _, _ = get_loaders(data, tokenizer, args.L,
                                                                args.C, args.B,
                                                                args.num_workers,
                                                                args.inputmark,
                                                                args.C_eval,
                                                                use_full_dataset,
                                                                macro_eval_mode, args=args, self_negs_again=True)
                    # if args.anncur:
        #     val_result = macro_eval(model, loader_val, num_val_samples, args)
        #     print(val_result)
        for batch_idx, batch in tqdm(enumerate(loader_train), total = len(loader_train)):  # Shuffled every epoch
            if args.debug and batch_idx > 10: break
            model.train()
            # try:
            result = model.forward(*batch, args = args)

            if type(result) is tuple:
                loss = result[0]
            elif type(result) is dict:
                loss = result['loss']
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

                # writer.add_scalar('lr', scheduler.get_last_lr()[0], step_num)
                if step_num % args.logging_steps == 0:
                    avg_loss = (tr_loss - logging_loss) / args.logging_steps
                    # writer.add_scalar('avg_loss', avg_loss, step_num)
                    logger.log('Step {:10d}/{:d} | Epoch {:3d} | '
                               'Batch {:5d}/{:5d} | '
                               'Average Loss {:8.4f}'.format(
                        step_num, num_train_steps, epoch,
                        batch_idx + 1, len(loader_train), avg_loss))
                    wandb.log({"average loss": avg_loss, "learning_rate": optimizer.param_groups[0]['lr'], "epoch": epoch})

                    logging_loss = tr_loss

                #  eval_result = micro_eval(args, model, loader_val)
                # writer.add_scalar('acc_val', eval_result['acc'], step_num)
        if args.eval_method == "skip":
            pass
        else:
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
            ,"val_loss": val_result['val_loss'], "epoch": epoch})
        try: 
            recall_result = recall_eval(model, loader_val, num_val_samples, eval_mode = args.eval_method, args = args)
            logger.log('Done with epoch {:3d} | recall {:8.4f} | '
                   'val acc unormalized  {:8.4f} ({}/{})|'
                   'val acc normalized  {:8.4f} ({}/{}) '.format(
            epoch,
            recall_result['recall'],
            recall_result['acc_unorm'],
            recall_result['num_correct'],
            num_val_samples,
            recall_result['acc_norm'],
            recall_result['num_correct'],
            recall_result['num_total_norm'],
            newline=False))
            wandb.log({"rereanker_recall": recall_result['recall'], "tournament_normalized_acc": recall_result['acc_norm']
            ,"tournament_unnormalized_acc": recall_result['acc_unorm'], "epoch": epoch})
        except:
            traceback.print_exc()

        if epoch >= 2:
            try:
                os.remove(os.path.join(args.model, "epoch_{}.bin".format(epoch - 1)))
            except:
                pass
        if val_result['acc_unorm'] >= best_val_perf:
            logger.log('      <----------New best val perf: %g -> %g' %
                       (best_val_perf, val_result['acc_unorm']))
            best_val_perf = val_result['acc_unorm']
            torch.save({'opt': args, 'sd': model.module.state_dict() if dp else model.state_dict(),
                    'perf': best_val_perf, 'epoch': epoch,
                    'opt_sd': optimizer.state_dict(),
                    'scheduler_sd': scheduler.state_dict(),
                    'tr_loss': tr_loss, 'step_num': step_num,
                    'logging_loss': logging_loss}, os.path.join(args.model, "epoch_{}.bin".format(epoch)))
            torch.save({'opt': args, 'sd': model.module.state_dict() if dp else model.state_dict(),
                        'perf': best_val_perf, 'epoch': epoch,
                        'opt_sd': optimizer.state_dict(),
                        'scheduler_sd': scheduler.state_dict(),
                        'tr_loss': tr_loss, 'step_num': step_num,
                        'logging_loss': logging_loss}, os.path.join(args.model, "pytorch_model.bin"))
        else:
            
            torch.save({'opt': args, 'sd': model.module.state_dict() if dp else model.state_dict(),
                    'perf': best_val_perf, 'epoch': epoch,
                    'opt_sd': optimizer.state_dict(),
                    'scheduler_sd': scheduler.state_dict(),
                    'tr_loss': tr_loss, 'step_num': step_num,
                    'logging_loss': logging_loss}, os.path.join(args.model, "epoch_{}.bin".format(epoch)))


        if args.training_one_epoch:
            break
    # cpt = torch.load(os.path.join/(args.model,"pytorch_model.bin")) if device.type == 'cuda' \
                # else torch.load(os.path.join(args.model,"pytorch_model.bin") , map_location=torch.device('cpu'))
    # model.load_state_dict(cpt['sd'])
    # model = load_model(os.path.join(args.model, "pytorch_model.bin"), device, eval_mode=True, dp=dp,
                    #    type_model=args.type_model)


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
    parser.add_argument('--num_workers', type=int, default=2,
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
                                 'extend_multi_dot'],
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
                                 'self_negative'],
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
