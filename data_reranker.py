import json
import math
import os
import statistics as stat
import sys
import torch
import util
import numpy as np
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from util import Logger as BasicLogger
from data_retriever import distribution_sample
from tqdm import tqdm
import random



class BasicDataset(Dataset):

    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False, args = None):
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_len_mention = 0
        self.max_len_candidate = 0
        self.max_num_candidates = max_num_candidates
        self.is_training = is_training
        self.num_samples_original = len(samples[0])
        self.indicate_mention_boundaries = indicate_mention_boundaries
        self.MENTION_START = '[unused0]'
        self.MENTION_END = '[unused1]'
        self.ENTITY_TAG = '[unused2]'
        self.args = args
        self.samples = self.cull_samples(samples)
        # if self.args.case_based:
        #     self.samples_case_based = samples 
        # ToDo: cull_samples에서 return되는 타입 파악.
        # 아니면 그냥 예외처리 (16개만)
        # if self.args.nearest:
        #     self.mention_ids = dict()
        #     for item in self.samples[0]:
        #         self.mention_ids[item['mention_id']]=item

    def __len__(self):
        return len(self.samples[0])

    def __getitem__(self, index):
        raise NotImplementedError

    def prepare_candidates(self, mention, candidates, args, self_negs_again = False):
        # xs = candidates['candidates'][:self.max_num_candidates]
        def place_elements(lst, elements, indices):
            for i, x in zip(indices, elements):
                lst[i] = x
            return lst
        xs = candidates['candidates']
        label_idx = candidates['labels']
        y = mention['label_document_id']
        scores = candidates['scores']
        if self.is_training:
            # At training time we can include target if not already included.
            if label_idx == '-1':
                xs = xs[-1:] + xs[:-1]
                scores = scores[-1:] + scores[:-1]
                label_idx = 0
            elif not y in xs[:self.max_num_candidates]:
                if len(scores) != 0:
                    scores = [scores[label_idx]]+scores[:label_idx]+scores[label_idx+1:]
                else: scores = [0]
                xs = [y] + [x for x in xs if x != y] 
                label_idx = 0
            if not self_negs_again:
                # At first, training set == entire set
                if args.type_cands == 'self_negative' or args.type_cands == "self_fixed_negative"\
                or args.type_cands == "self_mixed_negative":
                    if not y in xs[:args.num_sampled]:
                        xs = [y] + [x for x in xs if x != y] 
                        label_idx = 0
                    else:
                        label_idx = xs[:args.num_sampled].index(y)
                    return xs[:args.num_sampled], label_idx
                elif args.type_cands == 'fixed_negative':
                    pass
                elif args.type_cands == 'random_negative':
                    xs = xs[:self.max_num_candidates]
                    scores = scores[:self.max_num_candidates]
                    xs_with_scores = [(i, j) for i, j in zip(xs, scores)]
                    random.shuffle(xs_with_scores)
                    for i, x in enumerate(xs_with_scores):
                        xs[i] = x[0]
                        scores[i] = x[1]
                    label_idx = xs.index(y)
                elif args.type_cands == 'mixed_negative':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_fixed = int(self.max_num_candidates * args.cands_ratio)
                    num_hards = self.max_num_candidates - num_fixed
                    label_idx = xs.index(y)

                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    scores = [scores[label_idx]]+scores[:label_idx]+scores[label_idx+1:]
                    label_idx = 0
                    xs = np.array(xs)
                    scores = np.array(scores)
                    fixed_xs = xs[:num_fixed]
                    fixed_scores = scores[:num_fixed]
                    # Random sampling by giving same probs to each candidate
                    hard_scores = torch.zeros(len(candidates['scores']))[num_fixed:] 
                    probs = hard_scores.softmax(dim=0)
                    probs = probs.unsqueeze(0)
                    hard_cands = torch.tensor([num_fixed]) + distribution_sample(probs, num_hards, device).squeeze(0)
                    hard_xs = xs[hard_cands.squeeze(0)]
                    hard_scores = scores[hard_cands.squeeze(0)]
                    xs = np.concatenate((fixed_xs, hard_xs))
                    scores = np.concatenate((fixed_scores, hard_scores))
                if args.type_cands == 'hard_negative': 
                    # Later, training set is obtained by hard negs mining
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_hards = len(xs) 
                    # scores = candidates['scores'].clone().detach()
                    hard_scores = torch.tensor(candidates['scores'], dtype = float)
                    probs = hard_scores.softmax(dim=0).unsqueeze(0)
                    hard_cands = distribution_sample(probs, self.max_num_candidates,
                                                        device)
                    xs = np.array(xs)
                    scores = np.array(scores)
                    xs = xs[hard_cands.squeeze(0).cpu()]
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    scores_xs = scores[hard_cands.squeeze(0).cpu()]
                    scores = [scores[label_idx]] + [score for score in scores_xs if score != scores[label_idx]]
                    label_idx = 0
            else: 
                if args.type_cands == 'self_fixed_negative':
                    scores = torch.tensor(candidates['scores'], dtype = float)
                    topk_indices = torch.topk(scores, self.max_num_candidates)[1]
                    xs = np.array(xs)[topk_indices.cpu()]
                    xs = [y] + [x for x in xs if x!= y]
                    label_idx = 0
                elif args.type_cands == "self_negative":
                    # Later, training set is obtained by hard negs mining
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_hards = len(xs)
                    # scores = torch.tensor(candidates['self_scores'], dtype = float).clone().detach()
                    probs = candidates['self_scores'].softmax(dim=0).unsqueeze(0)
                    hard_cands = distribution_sample(probs, self.max_num_candidates,
                                                        device)
                    xs = np.array(xs)
                    xs = xs[hard_cands.squeeze(0).cpu()]
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    label_idx = 0

                elif args.type_cands == 'self_mixed_negative':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_fixed = int(self.max_num_candidates * args.cands_ratio)
                    num_hards = self.max_num_candidates - num_fixed
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    xs = np.array(xs)
                    fixed_xs = xs[:num_fixed]
                    # Random sampling by giving same probs to each candidate
                    scores = torch.zeros(len(candidates['scores']))[num_fixed:] 
                    probs = scores.softmax(dim=0)
                    probs = probs.unsqueeze(0)
                    hard_cands = torch.tensor([num_fixed]) + distribution_sample(probs, num_hards, device).squeeze(0)
                    hard_xs = xs[hard_cands.squeeze(0)]
                    xs = np.concatenate((fixed_xs, hard_xs))
                    label_idx = 0

            if args.nearest:
                xs_nearest = candidates['nearest_candidates']
                m_nearest = candidates['nearest_mentions']
                return xs[:self.max_num_candidates], label_idx, xs_nearest[:self.max_num_candidates], m_nearest[:self.max_num_candidates]      
            return xs[:self.max_num_candidates], label_idx, scores[:self.max_num_candidates]
        else:
            assert y in xs
            return xs[:self.max_num_candidates], label_idx, None

            

    def cull_samples(self, samples):
        self.num_samples_original = len(samples[0])
        # if self.args.nearest:
        if self.is_training:
            mentions = [mc for mc in samples[0]]
            return mentions, samples[1]

        else:
            # print(samples[0][0]['label_document_id'])
            # print(samples[1][0][samples[0][0]['mention_id']]['candidates'])
            mentions= []
            try:
                for mc in samples[0]:
                    
                    if mc['label_document_id'] in samples[1][mc['mention_id']]['candidates'][:self.max_num_candidates]:
                        mentions.append(mc)
            except:
                print(mc['label_document_id'])
                print(samples[1][mc['mention_id']]['candidates'][:self.max_num_candidates])
            # mentions = [mc for mc in samples[0] if mc['label_document_id'] in
                        # samples[1][mc['mention_id']]['candidates'][
                        # :self.max_num_candidates]]
            return mentions, samples[1]
        # else: 
        #     mentions = [mc for mc in samples[0] if mc['label_document_id'] in
        #                     samples[1][mc['mention_id']]['candidates'][
        #                     :self.max_num_candidates]]
        #     return mentions, samples[1]

    def get_mention_window(self, mention):
        # Get "enough" context from space-tokenized text.
        tokens = self.documents[mention['context_document_id']]['text'].split()
        prefix = tokens[max(0, mention['start_index'] - self.max_len_mention):
                        mention['start_index']]
        suffix = tokens[mention['end_index'] + 1:
                        mention['end_index'] + self.max_len_mention + 1]
        extracted_mention = tokens[mention['start_index']:
                                   mention['end_index'] + 1]
        prefix = ' '.join(prefix)
        suffix = ' '.join(suffix)
        extracted_mention = ' '.join(extracted_mention)
        assert extracted_mention == mention['text']

        # Get window under new tokenization.
        mention_tokens = self.tokenizer.tokenize(extracted_mention)
        if self.indicate_mention_boundaries:
            mention_tokens = [self.MENTION_START] + mention_tokens + \
                             [self.MENTION_END]
        return util.get_window(self.tokenizer.tokenize(prefix),
                               mention_tokens,
                               self.tokenizer.tokenize(suffix),
                               self.max_len_mention)


    def get_candidate_prefix(self, candidate_document_id):
        # Get "enough" context from space-tokenized text.
        tokens = self.documents[candidate_document_id]['text'].split()
        prefix = tokens[:self.max_len_candidate]
        prefix = ' '.join(prefix)

        # Get prefix under new tokenization.
        return self.tokenizer.tokenize(prefix)[:self.max_len_candidate], self.documents[candidate_document_id]['text'] 

class UnifiedDataset(BasicDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False, args=None, self_negs_again=False):
        super(UnifiedDataset, self).__init__(documents, samples, tokenizer,
                                             max_len, max_num_candidates,
                                             is_training,
                                             indicate_mention_boundaries, args)
        self.max_len = max_len // 2
        self.max_len_mention = self.max_len - 2  # cls and sep
        self.max_len_candidate = self.max_len - 2  # cls and sep
        self.args = args
        self.is_training = is_training
        self.self_negs_again = self_negs_again
        self.case_based = args.case_based
        if self.case_based:
            self.lookup_dict = {item['mention_id']: item for item in self.samples_case_based[0]}

    def __getitem__(self, index):
        mention = self.samples[0][index]
        candidates = self.samples[1][mention['mention_id']]
        mention_window = self.get_mention_window(mention)[0]
        mention_token  = self.documents[mention['context_document_id']]['text']
        mention_encoded_dict = self.tokenizer.encode_plus(mention_window,
                                                          add_special_tokens=True,
                                                          max_length=self.max_len,
                                                          pad_to_max_length=True,
                                                          truncation=True)
        mention_token_ids = torch.tensor(mention_encoded_dict['input_ids'])
        mention_masks = torch.tensor(mention_encoded_dict['attention_mask'])


        candidate_document_ids, label_ids, candidates_scores = self.prepare_candidates(mention, candidates, self.args, self.self_negs_again)

        candidates_token_ids = torch.zeros((len(candidate_document_ids),
                                            self.max_len))
        candidates_masks = torch.zeros((len(candidate_document_ids),
                                        self.max_len))
        candidates_tokens =  [None]*len(candidate_document_ids)
        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_window, candidate_token = self.get_candidate_prefix(candidate_document_id)
            candidate_dict = self.tokenizer.encode_plus(candidate_window,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len,
                                                        pad_to_max_length=True,
                                                        truncation=True)
            candidates_token_ids[i] = torch.tensor(candidate_dict['input_ids'])
            candidates_masks[i] = torch.tensor(candidate_dict['attention_mask'])
            candidates_tokens[i] = candidate_token




        if self.is_training and self.args.distill_training:
            return {"mention_token_ids": mention_token_ids,"mention_masks": mention_masks, "candidate_token_ids": candidates_token_ids, \
                "candidate_masks": candidates_masks, "label_idx": label_ids, "teacher_scores":torch.tensor(candidates_scores).clone().detach()}

        else:
            return {"mention_token_ids": mention_token_ids, "mention_masks": mention_masks, "candidate_token_ids": candidates_token_ids, \
                "candidate_masks": candidates_masks, "label_idx": label_ids}

class WikipediaDataset(BasicDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False, args=None, self_negs_again=False, max_context_len = 32):
        super(WikipediaDataset, self).__init__(documents, samples, tokenizer,
                                             max_len, max_num_candidates,
                                             is_training,
                                             indicate_mention_boundaries, args)
        self.max_len = max_len
        self.max_len_mention = max_context_len  # cls and sep
        self.max_len_candidate = self.max_len - self.max_len_mention # cls and sep
        self.args = args
        self.is_training = is_training
        self.self_negs_again = self_negs_again
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[unused0]','[unused1]','[unused2]']})
        self.samples = self.cull_samples(samples)

    def cull_samples(self, samples):
        self.num_samples_original = len(samples)
        # if self.args.nearest:
        if self.is_training:
            mentions = [i for i in range(len(samples))]
            return mentions, samples

        else:
            # print(samples[0][0]['label_document_id'])
            # print(samples[1][0][samples[0][0]['mention_id']]['candidates'])
            mentions= []
            returned_samples = []
            try:
                for i in range(len(samples)):
                    if 0<= samples[i]['label'] < self.max_num_candidates:
                        mentions.append(i)
                        returned_samples.append(samples[i])
            except: pass

            return mentions, returned_samples
    def prepare_candidates(self, candidates, args, self_negs_again = False):
        # xs = candidates['candidates'][:self.max_num_candidates]
        def place_elements(lst, elements, indices):
            for i, x in zip(indices, elements):
                lst[i] = x
            return lst
        xs = candidates['candidates']
        label_idx = candidates['label']
        y = xs[label_idx]
        scores = candidates['scores']
        if self.is_training:
            # At training time we can include target if not already included.
            if label_idx == '-1':
                xs = xs[-1:] + xs[:-1]
                scores = scores[-1:] + scores[:-1]
                label_idx = 0
            elif label_idx >= self.max_num_candidates:
                if len(scores) != 0:
                    scores = [scores[label_idx]]+scores[:label_idx]+scores[label_idx+1:]
                else: scores = [0]
                xs = [y] + [x for x in xs if x != y] 
                label_idx = 0
            if not self_negs_again:
                # At first, training set == entire set
                if args.type_cands == 'self_negative' or args.type_cands == "self_fixed_negative"\
                or args.type_cands == "self_mixed_negative":
                    if not y in xs[:args.num_sampled]:
                        xs = [y] + [x for x in xs if x != y] 
                        label_idx = 0
                    else:
                        label_idx = xs[:args.num_sampled].index(y)
                    return xs[:args.num_sampled], label_idx
                elif args.type_cands == 'fixed_negative':
                    pass
                elif args.type_cands == 'random_negative':
                    xs = xs[:self.max_num_candidates]
                    scores = scores[:self.max_num_candidates]
                    xs_with_scores = [(i, j) for i, j in zip(xs, scores)]
                    random.shuffle(xs_with_scores)
                    for i, x in enumerate(xs_with_scores):
                        xs[i] = x[0]
                        scores[i] = x[1]
                    label_idx = xs.index(y)
                elif args.type_cands == 'mixed_negative':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_fixed = int(self.max_num_candidates * args.cands_ratio)
                    num_hards = self.max_num_candidates - num_fixed
                    label_idx = xs.index(y)

                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    scores = [scores[label_idx]]+scores[:label_idx]+scores[label_idx+1:]
                    label_idx = 0
                    xs = np.array(xs)
                    scores = np.array(scores)
                    fixed_xs = xs[:num_fixed]
                    fixed_scores = scores[:num_fixed]
                    # Random sampling by giving same probs to each candidate
                    hard_scores = torch.zeros(len(candidates['scores']))[num_fixed:] 
                    probs = hard_scores.softmax(dim=0)
                    probs = probs.unsqueeze(0)
                    hard_cands = torch.tensor([num_fixed]) + distribution_sample(probs, num_hards, device).squeeze(0)
                    hard_xs = xs[hard_cands.squeeze(0)]
                    hard_scores = scores[hard_cands.squeeze(0)]
                    xs = np.concatenate((fixed_xs, hard_xs))
                    scores = np.concatenate((fixed_scores, hard_scores))
                if args.type_cands == 'hard_negative': 
                    # Later, training set is obtained by hard negs mining
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_hards = len(xs) 
                    # scores = candidates['scores'].clone().detach()
                    hard_scores = torch.tensor(candidates['scores'], dtype = float)
                    probs = hard_scores.softmax(dim=0).unsqueeze(0)
                    hard_cands = distribution_sample(probs, self.max_num_candidates,
                                                        device)
                    xs = np.array(xs)
                    scores = np.array(scores)
                    xs = xs[hard_cands.squeeze(0).cpu()]
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    scores_xs = scores[hard_cands.squeeze(0).cpu()]
                    scores = [scores[label_idx]] + [score for score in scores_xs if score != scores[label_idx]]
                    label_idx = 0
            else: 
                if args.type_cands == 'self_fixed_negative':
                    scores = torch.tensor(candidates['scores'], dtype = float)
                    topk_indices = torch.topk(scores, self.max_num_candidates)[1]
                    xs = np.array(xs)[topk_indices.cpu()]
                    xs = [y] + [x for x in xs if x!= y]
                    label_idx = 0
                elif args.type_cands == "self_negative":
                    # Later, training set is obtained by hard negs mining
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_hards = len(xs)
                    # scores = torch.tensor(candidates['self_scores'], dtype = float).clone().detach()
                    probs = candidates['self_scores'].softmax(dim=0).unsqueeze(0)
                    hard_cands = distribution_sample(probs, self.max_num_candidates,
                                                        device)
                    xs = np.array(xs)
                    xs = xs[hard_cands.squeeze(0).cpu()]
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    label_idx = 0

                elif args.type_cands == 'self_mixed_negative':
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    num_fixed = int(self.max_num_candidates * args.cands_ratio)
                    num_hards = self.max_num_candidates - num_fixed
                    xs = [y] + [x for x in xs if x != y]  # Target index always 0
                    xs = np.array(xs)
                    fixed_xs = xs[:num_fixed]
                    # Random sampling by giving same probs to each candidate
                    scores = torch.zeros(len(candidates['scores']))[num_fixed:] 
                    probs = scores.softmax(dim=0)
                    probs = probs.unsqueeze(0)
                    hard_cands = torch.tensor([num_fixed]) + distribution_sample(probs, num_hards, device).squeeze(0)
                    hard_xs = xs[hard_cands.squeeze(0)]
                    xs = np.concatenate((fixed_xs, hard_xs))
                    label_idx = 0

            if args.nearest:
                xs_nearest = candidates['nearest_candidates']
                m_nearest = candidates['nearest_mentions']
                return xs[:self.max_num_candidates], label_idx, xs_nearest[:self.max_num_candidates], m_nearest[:self.max_num_candidates]      
            return xs[:self.max_num_candidates], label_idx, scores[:self.max_num_candidates]
        else:
            assert y in xs
            return xs[:self.max_num_candidates], label_idx, None
    def get_candidate_prefix(self, candidate_document_id):
        # Get "enough" context from space-tokenized text.
        tokens_title = self.documents[candidate_document_id]['title'].split()
        tokens_text = self.documents[candidate_document_id]['text'].split()
        tokens = tokens_title + [self.ENTITY_TAG] + tokens_text
        prefix = tokens[:self.max_len_candidate]
        prefix = ' '.join(prefix)

        # Get prefix under new tokenization.
        return self.tokenizer.tokenize(prefix)[:self.max_len_candidate], self.documents[candidate_document_id]['text'] 

    def __getitem__(self, index):
        mention = self.samples[1][index]['context']
        candidates = self.samples[1][index]['candidates']
        mention_encoded_dict = self.tokenizer.encode_plus(mention,
                                                          add_special_tokens=False,
                                                          max_length=self.max_len_mention,
                                                          pad_to_max_length=True,
                                                          truncation=True)
        mention_token_ids = torch.tensor(mention_encoded_dict['input_ids'])

        mention_masks = torch.tensor(mention_encoded_dict['attention_mask'])


        candidate_document_ids, label_ids, candidates_scores = self.prepare_candidates(self.samples[1][index], self.args, self.self_negs_again)
        candidates_token_ids = torch.zeros((len(candidate_document_ids),
                                            self.max_len_candidate))
        candidates_masks = torch.zeros((len(candidate_document_ids),
                                        self.max_len_candidate))
        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_window, candidate_token = self.get_candidate_prefix(candidate_document_id)
            candidate_dict = self.tokenizer.encode_plus(candidate_window,
                                                        add_special_tokens=True,
                                                        max_length=self.max_len_candidate,
                                                        pad_to_max_length=True,
                                                        truncation=True)
            candidates_token_ids[i] = torch.tensor(candidate_dict['input_ids'])
            candidates_masks[i] = torch.tensor(candidate_dict['attention_mask'])
        # print(mention_token/_ids, mention_token_ids.shape, candidates_token_ids[0], candidates_token_ids.shape)

        if self.is_training and self.args.distill_training:
            return {"mention_token_ids": mention_token_ids,"mention_masks": mention_masks, "candidate_token_ids": candidates_token_ids, \
                "candidate_masks": candidates_masks, "label_idx": label_ids, "teacher_scores":torch.tensor(candidates_scores).clone().detach()}

        else:
            return {"mention_token_ids": mention_token_ids, "mention_masks": mention_masks, "candidate_token_ids": candidates_token_ids, \
                "candidate_masks": candidates_masks, "label_idx": label_ids}

class WikipediaFullDataset(WikipediaDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False, args = None, self_negs_again = False):
        super(WikipediaFullDataset, self).__init__(documents, samples, tokenizer,
                                          max_len, max_num_candidates,
                                          is_training,
                                          indicate_mention_boundaries, args = args)
    def __getitem__(self, index):
        mention_window = self.samples[1][index]['context'][1:-1]
        candidates = self.samples[1][index]['candidates']
        assert self.tokenizer.pad_token_id == 0
        candidate_document_ids, label_ids, _ = self.prepare_candidates(self.samples[1][index], self.args, self.self_negs_again)
        encoded_pairs = torch.zeros((len(candidate_document_ids), self.max_len))
        type_marks = torch.zeros((len(candidate_document_ids), self.max_len))
        input_lens = torch.zeros(len(candidate_document_ids))
        candidate_ids = [None]*len(candidate_document_ids)
        # candidate_ids = np.empty_like(candidate_document_ids)

        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_prefix, _ = self.get_candidate_prefix(candidate_document_id)
            encoded_dict = self.tokenizer.encode_plus(
                mention_window, candidate_prefix, return_attention_mask=False,
                pad_to_max_length=True, max_length=self.max_len,
                truncation=True)

            encoded_pairs[i] = torch.tensor(encoded_dict['input_ids'])
            candidate_ids[i] = candidate_document_id
            type_marks[i] = torch.tensor(encoded_dict['token_type_ids'])
            input_lens[i] = len(mention_window) + len(candidate_prefix) + 3
        # print( np.array([mention['mention_id']]), candidate_ids)
        # return encoded_pairs, type_marks, input_lens, np.array([mention['mention_id']]), candidate_ids
        return {"encoded_pairs": encoded_pairs, "type_marks": type_marks,\
        "input_lens": input_lens, "mention_id": mention_window,\
        "candidates_id": candidate_ids, "label_idx": label_ids}




class FullDataset(BasicDataset):
    def __init__(self, documents, samples, tokenizer, max_len,
                 max_num_candidates,
                 is_training, indicate_mention_boundaries=False, args = None, self_negs_again = False):
        super(FullDataset, self).__init__(documents, samples, tokenizer,
                                          max_len, max_num_candidates,
                                          is_training,
                                          indicate_mention_boundaries, args = args)
        self.max_len_mention = (max_len - 3) // 2  # [CLS], [SEP], [SEP]
        self.max_len_candidate = (max_len - 3) - self.max_len_mention
        self.args = args
        self.self_negs_again = self_negs_again
    def __getitem__(self, index):
        # For storing cross-encoder score, mention id and candidate id is loaded to dataloader
        mention = self.samples[0][index]
        candidates = self.samples[1][mention['mention_id']]
        mention_window, mention_start, mention_end\
            = self.get_mention_window(mention)
        mention_start += 1  # [CLS]
        mention_end += 1  # [CLS]
        assert self.tokenizer.pad_token_id == 0
        candidate_document_ids, label_ids, _ = self.prepare_candidates(mention, candidates, self.args, self.self_negs_again)
        encoded_pairs = torch.zeros((len(candidate_document_ids), self.max_len))
        type_marks = torch.zeros((len(candidate_document_ids), self.max_len))
        input_lens = torch.zeros(len(candidate_document_ids))
        candidate_ids = [None]*len(candidate_document_ids)
        # candidate_ids = np.empty_like(candidate_document_ids)
        for i, candidate_document_id in enumerate(candidate_document_ids):
            candidate_prefix, _ = self.get_candidate_prefix(candidate_document_id)
            encoded_dict = self.tokenizer.encode_plus(
                mention_window, candidate_prefix, return_attention_mask=False,
                pad_to_max_length=True, max_length=self.max_len,
                truncation=True)
            encoded_pairs[i] = torch.tensor(encoded_dict['input_ids'])
            candidate_ids[i] = candidate_document_id
            type_marks[i] = torch.tensor(encoded_dict['token_type_ids'])
            input_lens[i] = len(mention_window) + len(candidate_prefix) + 3
        # print( np.array([mention['mention_id']]), candidate_ids)
        # return encoded_pairs, type_marks, input_lens, np.array([mention['mention_id']]), candidate_ids
        return {"encoded_pairs": encoded_pairs, "type_marks": type_marks,\
        "input_lens": input_lens, "mention_id": mention['mention_id'],\
        "candidates_id": candidate_ids, "label_idx": label_ids}


def get_loaders(data, tokenizer, max_len, max_num_candidates, batch_size,
                num_workers, indicate_mention_boundaries,
                max_num_cands_val,
                use_full_dataset=True, macro_eval=True, args = None, self_negs_again = False, max_context_len = 32):
    eval_batch_size = batch_size
    if args.eval_batch_size is not None: eval_batch_size = args.eval_batch_size
    if args.dataset == 'zeshel':
        (documents, samples_train,
        samples_val, samples_test) = data
    elif args.dataset == 'wikipedia':
        (documents, samples_train,
        samples_val, samples_test, samples_test_2, samples_test_3) = data
    print('get train loaders')
    if use_full_dataset:
        if args.dataset == 'zeshel':

            dataset_train = FullDataset(documents, samples_train, tokenizer,
                                        max_len, max_num_candidates, True,
                                        indicate_mention_boundaries, args = args, self_negs_again=self_negs_again)
        elif args.dataset == 'wikipedia':
            dataset_train = WikipediaFullDataset(documents, samples_train, tokenizer,
                                        max_len, max_num_candidates, True,
                                        indicate_mention_boundaries, args = args, self_negs_again=self_negs_again)

    else:
        if args.dataset == 'zeshel':
            dataset_train = UnifiedDataset(documents, samples_train, tokenizer,
                                            max_len, max_num_candidates, True,
                                            indicate_mention_boundaries, args = args, self_negs_again=self_negs_again)
        elif args.dataset == 'wikipedia':
            dataset_train = WikipediaDataset(documents, samples_train, tokenizer,
                                            max_len, max_num_candidates, True,
                                            indicate_mention_boundaries, args = args, self_negs_again=self_negs_again, max_context_len = 32)
    if args.C>64:
        loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)
    else:
        if not self_negs_again:
            if args.type_cands == "self_negative" or args.type_cands == "self_fixed_negative":
                loader_train = DataLoader(dataset_train, batch_size=eval_batch_size,
                                    shuffle=True, num_workers=num_workers)            
            else:
                loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers)
        else:
            loader_train = DataLoader(dataset_train, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

    def help_loader(samples):
        if args.dataset == 'zeshel':
            num_samples = len(samples[0])
        else:
            num_samples = len(samples)
        if args.dataset == 'zeshel':    
            if use_full_dataset:
                dataset = FullDataset(documents, samples, tokenizer,
                                    max_len, max_num_cands_val, False,
                                    indicate_mention_boundaries, args)
            else:
                dataset = UnifiedDataset(documents, samples, tokenizer,
                                        max_len, max_num_cands_val, False,
                                        indicate_mention_boundaries, args)
        elif args.dataset == 'wikipedia':
            if use_full_dataset:
                dataset = WikipediaFullDataset(documents, samples, tokenizer,
                                    max_len, max_num_cands_val, False,
                                    indicate_mention_boundaries, args)
            else:
                dataset = WikipediaDataset(documents, samples, tokenizer,
                                        max_len, max_num_cands_val, False,
                                        indicate_mention_boundaries, args)
        loader = DataLoader(dataset, batch_size=eval_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
        return loader, num_samples
    
    if self_negs_again:
        loader_val = None
        loader_test = None
        val_num_samples = None
        test_num_samples = None
    else:
        if macro_eval:
            loader_val = []
            loader_test = []
            val_num_samples = []
            test_num_samples = []
            for sample in samples_val:
                loader, num_samples = help_loader(sample)
                loader_val.append(loader)
                val_num_samples.append(num_samples)
            for sample in samples_test:
                loader, num_samples = help_loader(sample)
                loader_test.append(loader)
                test_num_samples.append(num_samples)
        else:
            loader_val, val_num_samples = help_loader(samples_val)
            loader_test, test_num_samples = help_loader(samples_test)
            if args.dataset == 'wikipedia':
                loader_test_2, test_num_samples_2 = help_loader(samples_test_2)
                loader_test_3, test_num_samples_3 = help_loader(samples_test_3)
                return (loader_train, loader_val, loader_test, val_num_samples, test_num_samples,\
                 loader_test_2, loader_test_3, test_num_samples_2, test_num_samples_3)
    return (loader_train, loader_val, loader_test,
        val_num_samples, test_num_samples)


def load_zeshel_data(data_dir, cands_dir, macro_eval=True, debug = False, scores = None, nearest = False):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data')
    men_path = os.path.join(data_dir, 'mentions')

    def load_documents():
        documents = {}
        path = os.path.join(data_dir, 'documents')
        for fname in os.listdir(path):
            with open(os.path.join(path, fname)) as f:
                for i, line in enumerate(f):
                    fields = json.loads(line)
                    fields['corpus'] = os.path.splitext(fname)[0]
                    documents[fields['document_id']] = fields
        return documents

    documents = load_documents()

    def get_domains(part):
        domains = set()
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for line in f:
                field = json.loads(line)
                domains.add(field['corpus'])
        return list(domains)

    def load_mention_candidates_pairs(part, domain=None, in_domain=False, debug = False, scores = None, nearest = False):
        mentions = []
        with open(os.path.join(men_path, '%s.json' % part)) as f:
            for i, line in enumerate(f):
                if debug and i > 100: break
                field = json.loads(line)
                if in_domain:
                    if field['corpus'] == domain:
                        mentions.append(field)
                else:
                    mentions.append(field)
                    assert mentions[i]['context_document_id'] in documents
                    assert mentions[i]['label_document_id'] in documents
        candidates = {}
        print(os.path.join(cands_dir,
                               'candidates_%s.json' % part))
        with open(os.path.join(cands_dir,
                               'candidates_%s.json' % part)) as f:
            for i, line in enumerate(f):
                field = json.loads(line)
                if scores is not None:
                    if debug:
                        try:
                            field['self_scores'] = scores[i]
                        except: pass
                    else:
                        field['self_scores'] = scores[i]
                    
                candidates[field['mention_id']] = field
        return mentions, candidates
    

    print('start getting train pairs')
    samples_train = load_mention_candidates_pairs('train', debug = debug, scores = scores, nearest = nearest)
    print('start getting val and test pairs')
    if macro_eval:
        if scores is None:
            print('compute the domains')
            val_domains = get_domains('val')
            test_domains = get_domains('test')
            samples_val = []
            samples_test = []
            print('start getting val pairs')
            for val_domain in val_domains:
                pair = load_mention_candidates_pairs('val', val_domain, True, debug, nearest = nearest)
                samples_val.append(pair)
            print('start getting test pairs')
            for test_domain in test_domains:
                pair = load_mention_candidates_pairs('test', test_domain, True, debug, nearest = nearest)
                samples_test.append(pair)
        else:
            samples_val = None
            samples_test = None

    else:
        if scores is None:
            samples_val = load_mention_candidates_pairs('val')
            samples_test = load_mention_candidates_pairs('test')

            print('load all done')
        else:
            samples_val = None
            samples_test = None
    return documents, samples_train, samples_val, samples_test

def load_wikipedia_data(cands_dir):
    """

    :param data_dir: train_data_dir if args.train else eval_data_dir
    :return: mentions, entities,doc
    """
    print('begin loading data (AIDA)')
    def load_documents():
        documents = {}
        path = os.path.join(cands_dir, 'entities.json')
        with open(os.path.join(path)) as f:
            for i, line in tqdm(enumerate(f)):
                fields = json.loads(line)
                documents[fields['id']] = fields
        return documents
    documents = load_documents()

    def load_mention_candidates_pairs(part):
        assert part == 'AIDA-YAGO2_train' or part == 'AIDA-YAGO2_testa' or part == 'AIDA-YAGO2_testb' or part == 'clueweb_questions' or part == 'msnbc_questions'
        
        candidates = {}
        with open(os.path.join(cands_dir,
                               f'{part}.jsonl')) as f:
            for i, line in tqdm(enumerate(f)):
                field = json.loads(line)
                candidates[i] = field
        return candidates
    

    print('start getting train pairs')
    samples_train = load_mention_candidates_pairs('AIDA-YAGO2_train')
    print('start getting val and test pairs')
    samples_val = load_mention_candidates_pairs('AIDA-YAGO2_testa')
    samples_test = load_mention_candidates_pairs('AIDA-YAGO2_testb')
    samples_test_2 = load_mention_candidates_pairs('clueweb_questions')
    samples_test_3 = load_mention_candidates_pairs('msnbc_questions')

    print('load all done')

    return documents, samples_train, samples_val, samples_test, samples_test_2, samples_test_3

class Logger(BasicLogger):

    def __init__(self, log_path, on=True):
        super(Logger, self).__init__(log_path, on)

    def log_zeshel_data_stats(self, data):
        (documents, samples_train, samples_val, samples_test) = data

        def count_domains(dict_documents):
            domain_count = Counter()
            lens = []
            for doc in dict_documents.values():
                domain_count[doc['corpus']] += 1
                lens.append(len(doc['text'].split()))
            domain_count = sorted(domain_count.items(), key=lambda x: x[1],
                                  reverse=True)
            return domain_count, lens

        self.log('------------Zeshel data------------')
        domain_count_all, lens = count_domains(documents)
        self.log('%d documents in %d domains' % (len(documents),
                                                 len(domain_count_all)))
        for domain, count in domain_count_all:
            self.log('   {:20s}: {:5d}'.format(domain, count))
        self.log('Lengths (by space): mean %g, max %d, min %d, stdev %g' %
                 (stat.mean(lens), max(lens), min(lens), stat.stdev(lens)))

        def log_samples(samples, name, running_mention_ids,
                        train_label_document_ids=None):
            self.log('\n*** %s ***' % name)
            domain_count = Counter()
            mention_lens = []
            candidate_nums = []
            label_document_ids = Counter()
            label_id_seen_count = 0
            mention_id_overlap = False
            for mention in samples[0]:
                if mention['mention_id'] in running_mention_ids:
                    mention_id_overlap = True
                running_mention_ids[mention['mention_id']] = True
                domain_count[mention['corpus']] += 1
                mention_lens.append(mention['end_index'] -
                                    mention['start_index'] + 1)
                candidate_nums.append(len(samples[1][mention['mention_id']][
                                              'candidates']))
                # for doc_id in candidates['candidates']:
                #   assert mention['corpus'] == documents[doc_id]['corpus']
                label_document_ids[mention['label_document_id']] += 1

                if train_label_document_ids:
                    if mention['label_document_id'] in train_label_document_ids:
                        label_id_seen_count += 1

            domain_count = sorted(domain_count.items(), key=lambda x: x[1],
                                  reverse=True)
            self.log('%d mentions in %d domains' % (len(samples),
                                                    len(domain_count)))
            self.log('Mention ID overlap: %d' % mention_id_overlap)

            if train_label_document_ids:
                self.log('%d / %d linked to entities seen in train' %
                         (label_id_seen_count, len(samples)))

            if not name in ['heldout_train_seen', 'heldout_train_unseen']:
                for domain, count in domain_count:
                    self.log('   {:20s}: {:5d}'.format(domain, count))

            self.log('Mention lengths (by space): mean %g, max %d, min %d, '
                     'stdev %g' % (stat.mean(mention_lens), max(mention_lens),
                                   min(mention_lens), stat.stdev(mention_lens)))

            self.log('Candidate numbers: mean %g, max %d, min %d, stdev %g' %
                     (stat.mean(candidate_nums), max(candidate_nums),
                      min(candidate_nums), stat.stdev(candidate_nums)))

            return label_document_ids

        running_mention_ids = {}
        train_label_document_ids = log_samples(samples_train, 'train',
                                               running_mention_ids)
        log_samples(samples_val, 'val', running_mention_ids,
                    train_label_document_ids)
        log_samples(samples_test, 'test', running_mention_ids,
                    train_label_document_ids)

    def log_perfs(self, perfs, best_args):
        valid_perfs = [perf for perf in perfs if not np.isinf(perf)]
        best_perf = max(valid_perfs)
        self.log('%d perfs: %s' % (len(perfs), str(perfs)))
        self.log('best perf: %g' % best_perf)
        self.log('best args: %s' % str(best_args))
        self.log('')
        self.log('perf max: %g' % best_perf)
        self.log('perf min: %g' % min(valid_perfs))
        self.log('perf avg: %g' % stat.mean(valid_perfs))
        self.log('perf std: %g' % (stat.stdev(valid_perfs)
                                   if len(valid_perfs) > 1 else 0.0))
        self.log('(excluded %d out of %d runs that produced -inf)' %
                 (len(perfs) - len(valid_perfs), len(perfs)))

def preprocess_data(dataloader, model, debug):
    preprocessed_data = []
    print('preprocessing data')
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total = len(dataloader)):  # Shuffled every epoch
            if debug and batch_idx>10: break
            mention_embeds, mention_embeds_mask, candidates_embeds =\
                model.encode(*batch)
            preprocessed_data.append((mention_embeds.cpu(), mention_embeds_mask.cpu(), candidates_embeds.cpu()))
    return preprocessed_data