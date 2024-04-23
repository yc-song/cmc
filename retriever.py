from transformers import BertConfig
import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from transformers.models.bert.out_vector_modeling_bert import MyBertModel, DecoderLayerChunk
from torch.multiprocessing import Pool
import wandb
from scipy.stats import rankdata
import numpy as np
from mix_function import MeanLayer, MyLSTMBlock, ContextLayer, get_rep_by_avg, dot_attention, clean_input_ids, clean_input_embeddings
from time import time
import cProfile
import pstats
import faiss
profiler = cProfile.Profile()
NEAR_INF = 1e20


class PolyAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self, dim=1, attn='basic', residual=False, get_weights=True):
        super().__init__()
        self.attn = attn
        self.dim = dim
        self.get_weights = get_weights
        self.residual = residual

    def forward(self, xs, ys, mask_ys=None, values=None):
        """
        Compute attention.
        Attend over ys with query xs to obtain weights, then apply weights to
        values (ys if yalues is None)
        Args:
            xs: B x query_len x dim (queries)
            ys: B x key_len x dim (keys)
            mask_ys: B x key_len (mask)
            values: B x value_len x dim (values); if None, default to ys
        """
        # (2, 512, 768) * (2, 768, 128) = (2, 512, 128)
        l1 = torch.matmul(xs, ys.transpose(-1, -2))

        if self.attn == 'sqrt':
            d_k = ys.size(-1)
            l1 = l1 / math.sqrt(d_k)
        if mask_ys is not None:
            attn_mask = (mask_ys == 0).unsqueeze(-2)
            l1.masked_fill_(attn_mask, -NEAR_INF)
        l2 = F.softmax(l1, -1, dtype=torch.float).type_as(l1)
        if values is None:
            values = ys
        lhs_emb = torch.matmul(l2, values)

        # # add back the query
        if self.residual:
            lhs_emb = lhs_emb.add(xs)

        if self.get_weights:
            return lhs_emb.squeeze(self.dim - 1), l2
        else:
            return lhs_emb


class HardAttention(nn.Module):
    """
    Implements simple/classical hard attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, xs, ys):
        """

        :param xs: (B,T_x,d)
        :param ys: (B,C,T_y,d)
        :return: (B,C)
        """
        bsz, l_x, d = xs.size()
        bsz, C, l_y, d = ys.size()
        scores = (torch.matmul(xs, ys.reshape(bsz, -1, d).transpose(-1,
                                                                    -2)).reshape(
            bsz, l_x, C, l_y).max(-1)[0]).sum(1)
        return scores


class SoftAttention(nn.Module):
    """
    Implements simple/classical attention.
    """

    def __init__(self):
        super().__init__()
        self.attention = PolyAttention(dim=2, attn='basic',
                                       get_weights=False)

    def forward(self, xs, ys, values, mask_ys):
        """

        :param xs: (1,C,T_y,d)
        :param ys: (B,T_x,d)
        :param values: (B,T_x,d)
        :param mask_ys: (B,T_x)
        :return: (B,C)
        """
        bsz_x, C, l_y, d = xs.size()
        xs = xs.reshape(bsz_x, -1, d)
        bsz, l_x, d = ys.size()
        attended_embeds = self.attention(xs, ys,
                                         mask_ys=mask_ys,
                                         values=values)  # (B,CT_y,d)
        scores = (attended_embeds * xs).sum(-1).reshape(
            bsz, C, l_y).sum(-1)
        return scores

# Distillation loss function 
def distillation_loss(student_outputs, teacher_outputs, labels, alpha = 0.5, temperature = 1, valid = False, mrr_penalty = False):
    ## Inputs
    # student_outputs (torch.tensor): score distribution from the model trained
    # teacher_outputs (torch.tensor): score distribution from the teacher model
    # labels (torch.tensor): gold for given instance
    # alpha (float): weight b/w two loss terms
    # temperature (flost): softmac temperature
    ## Outputs
    # loss (torch.tensor): convex combination of two losses (student and teacher loss)
    # student_loss (torch.tensor): Cross-entropy from student model
    # teacher_loss (torch.tensor): KL-divergence b/w student and teacher model's score distribution

    assert 0<=alpha and alpha<=1 # Weight should exist at [0,1]
    # Teacher loss is obtained by KL divergence
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if teacher_outputs is not None:
        teacher_outputs = teacher_outputs[:,:student_outputs.size(1)].to(device)
        teacher_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs/temperature, dim=1),
                             nn.functional.softmax(teacher_outputs/temperature, dim=1))
    else: 
        teacher_loss = torch.tensor([0.]).to(device)
    # Student loss is calculated by cross-entropy loss
    student_loss = nn.CrossEntropyLoss()(student_outputs, labels)
    if teacher_outputs is not None and mrr_penalty:
        # Zero tensor for hinge loss
        zero_tensors = torch.zeros_like(labels)
        zero_tensors = np.expand_dims(zero_tensors.cpu().detach().numpy(), axis = 1)
        # Reranker ranking
        idx_array_reranker = rankdata(-student_outputs.cpu().detach().numpy(), axis = 1, method = 'min')
        rank_reranker = np.take_along_axis(idx_array_reranker, labels[:, None].cpu().detach().numpy(), axis=1)
        # Retriever ranking
        idx_array_retriever = rankdata(-teacher_outputs.cpu().detach().numpy(), axis = 1, method = 'min')
        rank_retriever = np.take_along_axis(idx_array_retriever, labels[:, None].cpu().detach().numpy(), axis=1)
        # penalty is defined as (max(0, 1/rank_retriever-1/rank_reranker))
        penalty = np.maximum(zero_tensors, 1/rank_retriever-1/rank_reranker)
        penalty = torch.from_numpy(penalty).squeeze(1).to(device)
        # penalty is multiplied to loss of each element
        # https://discuss.pytorch.org/t/weighted-cross-entropy-for-each-sample-in-batch/101358/10
        penalty_loss = torch.nn.CrossEntropyLoss(reduction='none')(student_outputs, labels)
        penalty_loss *= penalty
        penalty_loss = torch.mean(penalty_loss)
        student_loss += penalty_loss
    # Weighted sum of teacher and student loss
    loss = alpha*student_loss + (1-alpha)*teacher_loss
    return loss, student_loss, teacher_loss

class UnifiedRetriever(nn.Module):
    def __init__(self, encoder, device, num_codes_mention, num_codes_entity,
                 mention_use_codes, entity_use_codes, attention_type,
                 candidates_embeds=None, evaluate_on=False, args= None, num_heads = 2, num_layers = 2):
        super(UnifiedRetriever, self).__init__()
        self.mention_use_codes = mention_use_codes
        self.entity_use_codes = entity_use_codes
        self.attention_type = attention_type
        if args.type_model == "mixencoder":
            self.mention_encoder = None
            self.entity_encoder = None
        else:
            self.mention_encoder = encoder
            self.entity_encoder = copy.deepcopy(encoder)
        self.device = device
        self.loss_fct = CrossEntropyLoss()

        self.num_mention_vecs = num_codes_mention
        self.num_entity_vecs = num_codes_entity
        self.evaluate_on = evaluate_on
        self.embed_dim = None
        if self.mention_use_codes:
            self.embed_dim = BertConfig().hidden_size
            mention_codes = nn.Embedding(self.num_mention_vecs,
                                         self.embed_dim).weight.data.normal_(
                mean=0.0, std=self.mention_encoder.config.initializer_range)
            self.mention_codes = nn.Parameter(mention_codes)
            self.mention_codes_attention = PolyAttention(dim=2, attn='basic',
                                                         get_weights=False)
        if self.entity_use_codes:
            self.embed_dim = BertConfig().hidden_size
            entity_codes = nn.Embedding(self.num_entity_vecs,
                                        self.embed_dim).weight.data.normal_(
                mean=0.0, std=self.entity_encoder.config.initializer_range)
            self.entity_codes = nn.Parameter(entity_codes)
            self.entity_codes_attention = PolyAttention(dim=3, attn='basic',
                                                        get_weights=False)
        if self.attention_type == 'soft_attention':
            self.attention = SoftAttention()
        else:
            self.attention = HardAttention()
        self.candidates_embeds = candidates_embeds
        # self.extend_multi = extend_multi(num_heads, num_layers, args)
        if args.type_model == 'mlp':
            self.mlp = mlp(args)
        if args.type_model == 'mlp_with_som':
            self.mlp_with_som = mlp_with_som(args)
        # define transformer encoder
        if self.embed_dim is None: 
            if args.type_bert == 'base':
                self.embed_dim = 768
            else:
                self.embed_dim = 1024
        self.args = args
        if args.type_model == "extend_multi_dot" or args.type_model == "extend_multi":
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.extend_multi_transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
            if args.identity_bert:
                self.extend_multi_transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads, args = args).to(self.device)
            self.extend_multi_transformerencoder = torch.nn.TransformerEncoder(self.extend_multi_transformerencoderlayer, self.num_layers).to(self.device)
            self.extend_multi_linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
            self.extend_multi_token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)
        if args.type_model == "deformer":
            self.deformer = DeformerModel(encoder, args)
        if args.type_model == "mixencoder":
            self.mixencoder = MixEncoderModel(args)
        if self.args.case_based: 
            assert self.args.batch_first == False
            assert not self.args.attend_to_gold
    # Get embedding of mention and candidate
    def encode(self, mention_token_ids, mention_masks, candidate_token_ids,
               candidate_masks, entity_token_ids=None, entity_masks=None, too_large = False, entity_bsz=64):
        ## Input
        # mention_token_ids (torch.tensor): mention token ids by tokenizer
        # mention_masks (torch.tensor): 0 for meaningless tokens
        # candida_token_ids (torch.tensor): candidate token ids by tokenizer
        # mention_masks (torch.tensor): 0 for meaningless tokens
        ## Output
        # mention_embeds (torch.tensor): mention embedding
        # mention_embeds_masks (torch.tensor): 0 for meaningless tokens
        # candidates_embeds (torch.tensor): candidate embedding
        candidates_embeds = None
        mention_embeds = None
        mention_embeds_masks = None
        def candidates_encoding(candidate_token_ids, candidate_masks):
            if candidate_token_ids is not None:
                candidate_token_ids = candidate_token_ids.to(self.device).long()
                candidate_masks = candidate_masks.to(self.device).long()
                B, C, L = candidate_token_ids.size()
                candidate_token_ids = candidate_token_ids.view(-1, L)
                candidate_masks = candidate_masks.view(-1, L)
                # B X C X L --> BC X L
                candidates_hiddens = (self.entity_encoder(
                    input_ids=candidate_token_ids,
                    attention_mask=candidate_masks
                )[0]).reshape(B, C, L, -1)
                candidate_masks = candidate_masks.view(B, C, L)
                if self.entity_use_codes:
                    n, d = self.entity_codes.size()
                    candidates_embeds = self.entity_codes.unsqueeze(0).unsqueeze(
                        1).expand(B, C, n, d)
                    candidates_embeds = self.entity_codes_attention(
                        candidates_embeds, candidates_hiddens,
                        mask_ys=candidate_masks, values=candidates_hiddens)
                else:
                    candidates_embeds = candidates_hiddens[:,
                                        :, :self.num_entity_vecs,
                                        :]
                return candidates_embeds
        if not too_large:
            candidates_embeds = candidates_encoding(candidate_token_ids, candidate_masks)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            candidates_embeds = torch.tensor([])
            num_in_batch = 500
            for i in range(candidate_token_ids.size(1)//num_in_batch):
                candidate_embed = candidates_encoding(candidate_token_ids[:,num_in_batch*i:num_in_batch*(i+1),:], candidate_masks[:,num_in_batch*i:num_in_batch*(i+1),:]).cpu()
                candidates_embeds = torch.cat((candidates_embeds, candidate_embed), dim = 1)
            candidates_embeds = candidates_embeds.to(device)
        if mention_token_ids is not None:
            mention_token_ids = mention_token_ids.to(self.device).long()
            mention_masks = mention_masks.to(self.device).long()
            mention_hiddens = self.mention_encoder(input_ids=mention_token_ids,
                                                   attention_mask=mention_masks)[0]
            B = mention_token_ids.size(0)
            if self.mention_use_codes:
                # m codes m different embeds
                m, d = self.mention_codes.size()
                B, L = mention_token_ids.size()
                mention_codes_embeds = self.mention_codes.unsqueeze(0).expand(B,
                                                                              m,
                                                                              d)
                mention_embeds = self.mention_codes_attention(
                    mention_codes_embeds,
                    mention_hiddens,
                    mask_ys=mention_masks,
                    values=mention_hiddens)
            else:
                mention_embeds = mention_hiddens[:, :self.num_mention_vecs, :]
            mention_embeds_masks = mention_embeds.new_ones(B,
                                                           self.num_mention_vecs).byte()
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_token_ids = entity_token_ids.to(self.device).long()
            entity_masks = entity_masks.to(self.device).long()
            B = entity_token_ids.size(0)
            # B X C X L --> BC X L
            candidates_hiddens = self.entity_encoder(
                input_ids=entity_token_ids,
                attention_mask=entity_masks
            )[0]
            if self.entity_use_codes:
                n, d = self.entity_codes.size()
                candidates_embeds = self.entity_codes.unsqueeze(0).expand(B, n,
                                                                          d)
                candidates_embeds = self.entity_codes_attention(
                    candidates_embeds, candidates_hiddens,
                    mask_ys=candidate_masks, values=candidates_hiddens)
            else:
                candidates_embeds = candidates_hiddens[:, :self.num_entity_vecs,
                                    :]
        return mention_embeds, mention_embeds_masks, candidates_embeds

    def forward(self, mention_token_ids, mention_masks, candidate_token_ids,
                candidate_masks, label_idx = None, teacher_scores = None, candidate_probs=None, recall_eval = False, top_k = 64, \
                beam_ratio = 0.5, args = None, sampling = False, nearest_mention_token_ids = None, nearest_mention_masks = None,\
                nearest_label_token_ids = None, nearest_label_masks = None, loader_val_cased = None,\
                nearest_candidate_token_ids = None, nearest_candidate_masks = None,\
                 mention_id = None, candidate_id = None, evaluate = False):
        ## Input
        # mention_token_ids (torch.tensor): mention token ids by tokenizer
        # mention_masks (torch.tensor): 0 for meaningless tokens
        # candida_token_ids (torch.tensor): candidate token ids by tokenizer
        # mention_masks (torch.tensor): 0 for meaningless tokens
        # label_idx (torch.tensor): gold for given instance
        # teacher_scores (torch.tensor): score distribution from the teacher model
        ## Output
        # loss (torch.tensor): loss for training
        # predicts (torch.tensor): prediction for given instance
        # scores (torch.tensor): score distribution for given instance
        # student_loss (torch.tensor): Cross-entropy from student model
        # teacher_loss (torch.tensor): KL-divergence b/w student and teacher model's score distribution
        def create_faiss_index(entity_embeddings):
            # Assuming entity_embeddings is a 2D numpy array (num_entities x embedding_dim)
            entity_embeddings = entity_embeddings.squeeze(1)
            dimension = entity_embeddings.shape[-1]
            faiss_index = faiss.IndexFlatIP(dimension)  # Using L2 distance for similarity
            faiss_index.add(entity_embeddings)  # Adding the embeddings to the index
            faiss.write_index(faiss_index, './data/dual/faiss_index.index')
            return faiss_index

        def search_faiss_index(faiss_index, query_embeddings, k):
            # query_embeddings is a 2D numpy array (num_queries x embedding_dim)
            # k is the number of nearest neighbors to retrieve
            distances, indices = faiss_index.search(query_embeddings.cpu(), k)
            return distances, indices
        if self.evaluate_on or evaluate:  # This option is not used for extend_multi models
            if not self.candidates_embeds is None:
                mention_embeds, mention_embeds_masks = self.encode(
                    mention_token_ids, mention_masks, None, None)[:2]
                bsz, l_x, mention_dim = mention_embeds.size()
                if type(self.candidates_embeds) is not dict:
                    num_cands, l_y, cand_dim = self.candidates_embeds.size()
                if self.args.type_model == 'dual':
                    faiss_index = create_faiss_index(self.candidates_embeds.cpu())

            else: 

                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks)                
            if self.args.type_model == 'dual':
                for m in mention_embeds:
                    scores, indices = search_faiss_index(faiss_index, m.reshape(1,-1), self.args.num_cands)
                return {'scores':scores, 'indices': indices}
            if self.attention_type == 'soft_attention':
                scores = self.attention(self.candidates_embeds.unsqueeze(0).to(
                    self.device), mention_embeds, mention_embeds,
                    mention_embeds_masks)
            elif self.attention_type == 'extend_multi_dot':
                dot = True
                if candidate_id is not None:

                    candidate_id = [list(t) for t in zip(*candidate_id)]
                    candidates_embeds = torch.stack([torch.stack([self.candidates_embeds[item] for item in sublist]) for sublist in candidate_id])
                    # candidates_embeds = self.candidates_embeds[candidate_id]
                    # Take embeddings as input for extend_multi model
                    scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args)

                else:
                    scores = self.extend_multi(mention_embeds, self.candidates_embeds, dot, args)
            elif self.attention_type == 'deformer':
                mention_dict = {"input_ids":mention_token_ids.to(self.device), "attention_mask":mention_masks.to(self.device)}
                num_cands = candidate_token_ids.shape[1]
                candidates_embeds_masks = candidate_masks.reshape(-1, candidate_masks.shape[-1])
                
                mention_embeds, mention_embeds_masks= self.deformer.encode_lower(mention_dict)
                joint_embeddings = self.deformer.encode_higher(mention_embeds, self.candidates_embeds, mention_embeds_masks, candidates_embeds_masks)
                pooler = joint_embeddings[:,0,:]                
                scores = self.deformer.classifier(pooler).squeeze(-1)
                scores = scores.reshape(mention_token_ids.shape[0], num_cands)
            else:
                if self.args.model == 'dual':
                    faiss_index = create_faiss_index(self.candidates_embeds.cpu())
                    for m in mention_embeds:
                        scores, indices = search_faiss_index(faiss_index, m.reshape(1,-1), self.args.num_cands)
                    return {'scores':scores, 'indices': indices}
                else:
                    scores = (
                    torch.matmul(mention_embeds.reshape(-1, mention_dim),
                                 self.candidates_embeds.reshape(-1,
                                                                cand_dim).t().to(
                                     self.device)).reshape(bsz, l_x,
                                                           num_cands,
                                                           l_y).max(-1)[
                        0]).sum(1)
            predicts = scores.argmax(1)
            
            return {'scores':scores, 'predictions': predicts}
        else:  # train\
            B, C, L = candidate_token_ids.size() # e.g. 8, 256, 128
            # B x m x d
            #  get  embeds
            # print(candidates_embeds.shape) # (1, 65, 128, 768)

            if self.attention_type == 'extend_multi' or self.attention_type == 'extend_multi_dot':
                if self.attention_type == 'extend_multi_dot':
                    dot = True
                else: 
                    dot = False
                # if recall_eval is True, # of candidates is same as args.num_eval_cands
                # else, # of candidates is same as args.num_training_cands
                # Get embeddings
                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks)
                # Take embeddings as input for extend_multi model
                scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args)
            elif self.attention_type == 'mlp_with_som':
                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks)
                scores = self.mlp_with_som(mention_embeds, candidates_embeds)
            elif self.attention_type == 'deformer':
                mention_dict = {"input_ids":mention_token_ids.to(self.device), "attention_mask":mention_masks.to(self.device)}
                num_cands = candidate_token_ids.shape[1]
                candidate_token_ids = candidate_token_ids.int().reshape(-1, candidate_token_ids.shape[-1])
                candidate_masks = candidate_masks.reshape(-1, candidate_masks.shape[-1])
                entity_dict = {"input_ids":candidate_token_ids.to(self.device), "attention_mask":candidate_masks.to(self.device)}
                
                mention_embeds, mention_embeds_masks= self.deformer.encode_lower(mention_dict)
                candidates_embeds, candidates_embeds_masks= self.deformer.encode_lower(entity_dict)
                joint_embeddings = self.deformer.encode_higher(mention_embeds, candidates_embeds, mention_embeds_masks, candidates_embeds_masks)
                pooler = joint_embeddings[:,0,:]                
                scores = self.deformer.classifier(pooler).squeeze(-1)
                scores = scores.reshape(mention_token_ids.shape[0], num_cands)
            elif self.attention_type == 'mixencoder':
                candidate_token_ids = candidate_token_ids.int().to(self.device)
                candidate_masks = candidate_masks.int().to(self.device)
                candidate_token_type_ids = torch.zeros_like(candidate_masks).int().to(self.device)
                mention_token_type_ids = torch.zeros_like(mention_masks).int().to(self.device)
                candidate_embeddings = self.mixencoder.prepare_candidates(candidate_token_ids, candidate_token_type_ids, candidate_masks)  
                candidate_embeddings = candidate_embeddings.reshape(mention_token_ids.shape[0], -1, candidate_embeddings.shape[-2],  candidate_embeddings.shape[-1])
                # torch.Size([2, 512]) torch.Size([2, 10, 1, 768])

                scores = self.mixencoder.do_queries_match(input_ids = mention_token_ids.int().to(self.device), token_type_ids = mention_token_type_ids.int().to(self.device), attention_mask = mention_masks.int().to(self.device), \
                candidate_context_embeddings = candidate_embeddings)
            else:
                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids,
                                                candidate_masks)
                if self.attention_type == 'soft_attention':
                    scores = self.attention(candidates_embeds, mention_embeds,
                                            mention_embeds, mention_embeds_masks)
                elif self.attention_type == 'mlp':
                    scores = self.mlp(mention_embeds, candidates_embeds)
                else:
                    scores = self.attention(mention_embeds, candidates_embeds)
            predicts = scores.argmax(1)
            if label_idx is None:
                labels = torch.zeros(B).long().to(self.device)
            else:
                if label_idx.dim() == 2:
                    labels = label_idx[0].to(self.device)
                else:   
                    labels = label_idx.to(self.device)
            if candidate_probs is not None:  # logits adjustment
                candidate_probs = candidate_probs.to(self.device)
                scores[:, 1:] -= ((C - 1) * candidate_probs).log()
            if args.distill_training:
                loss, student_loss, teacher_loss = distillation_loss(scores, teacher_scores, labels, args.alpha, mrr_penalty = args.mrr_penalty)
                return {'loss':loss, 'predictions':predicts, 'scores':scores,\
                'student_loss':student_loss, 'teacher_loss':teacher_loss}
            elif args.energy_model:
                candidates_embeds = candidates_embeds.squeeze(-2)
                retriever_scores = torch.bmm(mention_embeds, candidates_embeds.transpose(2,1)).squeeze(1)
                retriever_loss = self.loss_fct(retriever_scores, labels)
                reranker_scores = scores
                reranker_loss = self.loss_fct(reranker_scores, labels)
                loss = retriever_loss + reranker_loss
                retriever_predicts = retriever_scores.argmax(1)
                return loss, predicts, scores, retriever_loss, reranker_loss, retriever_predicts, retriever_scores
            else:
                loss = self.loss_fct(scores, labels)
                return {'loss':loss, 'predictions':predicts, 'scores':scores}
    def extend_multi(self, xs, ys, dot = False, args = None, num_mention_vecs = 1, nearest_mention = None, nearest_gold = None,\
     nearest_mention_token_ids = None, nearest_mention_masks = None, nearest_candidate_token_ids = None, nearest_candidate_masks = None):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        # Deprecated
        if self.args.model_top is None and self.args.token_type :
            token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
            token_embedding_xs = self.token_type_embeddings(token_type_xs)
            token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
            token_embedding_ys = self.token_type_embeddings(token_type_ys)
            input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        else:
            input = torch.cat([xs, ys], dim = 1) # concatenate mention and entity embeddings
        # Take concatenated tensor as the input for transformer encoder
        attention_result = self.extend_multi_transformerencoder(input)
        # if self.args.skip_connection:
        #     if self.args.layernorm:
        #         layer_norm = nn.LayerNorm(input.size(-1)).to(self.device)
        #         attention_result = layer_norm(0.5*(input + self.extend_multi_transformerencoder(input)))
        #     else:
        #         attention_result = 0.5*(input + self.extend_multi_transformerencoder(input))
        # Get score from dot product
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,self.args.num_mention_vecs:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        # Get score distribution from linear head
        else:
            scores = self.linearhead(attention_result[:,self.args.num_mention_vecs:,:])
            scores = scores.squeeze(-1)

        return scores
    def forward_chunk(self, xs, ys, dot = False, args = None):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        xs = xs.repeat_interleave(int(ys.size(0)//xs.size(0)), dim=0)
        if args.token_type:
            token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
            token_embedding_xs = self.token_type_embeddings(token_type_xs)
            token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
            token_embedding_ys = self.token_type_embeddings(token_type_ys)
            input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        else:
            input = torch.cat([xs, ys], dim = 1)
        attention_result = self.extend_multi_transformerencoder(input)
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        else:
            scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
            scores = scores.squeeze(-1)
        return scores
class extend_multi(nn.Module):
    def __init__(self, num_heads, num_layers, args):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        if args.type_bert=='base':
            self.embed_dim = 768
        else: 
            self.embed_dim = 1024
        # define transformer encoder
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
        if args.identity_bert:
            self.transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads, args = args).to(self.device)
        self.args = args
        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers).to(self.device)
        self.linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)
        if self.args.case_based: 
            assert self.args.batch_first == False
            assert not self.args.attend_to_gold
    def process_chunk(self, xs, ys, dot):
        # Process the chunk using your deep learning model
        processed_chunk = self.forward(xs, ys, dot)
        return processed_chunk
    def forward(self, xs, ys, unifiedretriever = None, dot = False, args = None, num_mention_vecs = 1, nearest_mention = None, nearest_gold = None,\
     nearest_mention_token_ids = None, nearest_mention_masks = None, nearest_candidate_token_ids = None, nearest_candidate_masks = None):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        if self.args.model_top is None and self.args.token_type :
            token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
            token_embedding_xs = self.token_type_embeddings(token_type_xs)
            token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
            token_embedding_ys = self.token_type_embeddings(token_type_ys)
            input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        else:
            input = torch.cat([xs, ys], dim = 1) # concatenate mention and entity embeddings
        # Take concatenated tensor as the input for transformer encoder
        attention_result = self.transformerencoder(input)
        # Get score from dot product
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,self.args.num_mention_vecs:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        # Get score distribution from linear head
        else:
            scores = self.linearhead(attention_result[:,self.args.num_mention_vecs:,:])
            scores = scores.squeeze(-1)
        return scores
    def forward_chunk(self, xs, ys, dot = False, args = None):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        xs = xs.repeat_interleave(int(ys.size(0)//xs.size(0)), dim=0)
        if args.token_type:
            token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
            token_embedding_xs = self.token_type_embeddings(token_type_xs)
            token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
            token_embedding_ys = self.token_type_embeddings(token_type_ys)
            input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        else:
            input = torch.cat([xs, ys], dim = 1)
        attention_result = self.transformerencoder(input)
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        else:
            scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
            scores = scores.squeeze(-1)
        return scores
class mlp(nn.Module):
    def __init__(self, args):
        super(mlp, self).__init__()
        if args.type_bert == 'base':
            self.input_size = 768*2
        elif args.type_bert == 'large':
            self.input_size = 1024*2
        self.fc = nn.Linear(self.input_size, 1)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, xs, ys):
        xs = xs.expand(-1, ys.size(1), -1)
        ys = ys.squeeze(-2)
        input = torch.cat([xs, ys], dim = -1)
        output = self.softmax(self.fc(input).squeeze(-1))
        return output

class mlp_with_som(nn.Module):
    def __init__(self, args):
        super(mlp_with_som, self).__init__()
        self.dropout=nn.Dropout(0.1)
        self.layers = nn.ModuleList()
        if args.type_bert == 'base':
            self.input_size = 768*2
        elif args.type_bert == 'large':
            self.input_size = 1024*2
        self.act_fn = nn.Softplus()
        self.mlp_layers = [int(item) for item in args.mlp_layers.split(",")]
        for i, num in enumerate(self.mlp_layers):
            if i == 0:
                self.layers.append(nn.Linear(self.input_size, self.mlp_layers[i]))
            else:
                self.layers.append(nn.Linear(self.mlp_layers[i-1], self.mlp_layers[i]))
        self.layers.append(nn.Linear(self.mlp_layers[-1], 1))
    def mlp(self, input):
        input = torch.flatten(input, start_dim = -2)
        for i, layer in enumerate(self.layers[:-1]):
            input = self.act_fn(layer(self.dropout(input)))
        input = self.layers[-1](self.dropout(input))
        return input
    def forward(self, xs, ys):        
        # xs == context
        # ys == entity
        bsz, l_x, d = xs.size()
        bsz, C, l_y, d = ys.size()
        xs = xs.unsqueeze(1).expand(-1, C, -1, -1)
        xs = xs.reshape(-1, l_x, d)
        ys = ys.reshape(-1, l_y, d)
        output = torch.bmm(xs, ys.transpose(1,2))
        dot_scores, argmax_values = torch.max(output, dim=-1)
        mlp_input = torch.stack([xs, torch.gather(ys, dim = 1, index = argmax_values.unsqueeze(-1).expand(-1,-1,d))], dim = -2)
        scores = self.mlp(mlp_input).squeeze(-1)
        scores = torch.sum(scores, -1).unsqueeze(-1)
        scores = scores.reshape(bsz, C)
        return scores

class IdentityInitializedTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=None, dropout=0.1, activation="relu", args = None):
        super(IdentityInitializedTransformerEncoderLayer, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  
        self.args = args
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head, batch_first = self.args.batch_first)
        if self.args.fixed_initial_weight:
            self.weight = torch.tensor([0.]).to(self.device)
        else:
            self.weight = nn.Parameter(torch.tensor([-5.], requires_grad = True))
            
    def forward(self,src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        out1 = self.encoder_layer(src, src_mask, src_key_padding_mask) # For Lower Torch Version
        # out1 = self.encoder_layer(src, src_mask, src_key_padding_mask, is_causal) # For Higher Torch Version
        sigmoid_weight = torch.sigmoid(self.weight).to(self.device)
        out = sigmoid_weight*out1 + (1-sigmoid_weight)*src

        if self.args.layernorm:
            layer_norm = nn.LayerNorm(src.size(-1)).to(self.device)
            out = layer_norm(sigmoid_weight*out1 + (1-sigmoid_weight)*src)
        
        return out

class DeformerModel(torch.nn.Module):
    # encode a text using lower layers
    # Only us self.lm_q!!
    def __init__(self, encoder, args):
        super(DeformerModel, self).__init__()
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  
        self.num_layers = 4
        if args.type_bert=='base':
            self.embed_dim = 768
        else: 
            self.embed_dim = 1024
        self.lm_q = encoder.to(self.device)
        self.lower_encoder = self.lm_q.encoder.layer[:-self.num_layers]
        self.higher_encoder = self.lm_q.encoder.layer[-self.num_layers:]
        self.classifier = torch.nn.Linear(self.embed_dim, 1)
    def encode_lower(self, psg, num_vecs = 1): # passage를 받아서 representation을 만듦
        if psg is None:
            return None
        psg_out = self.lm_q(**psg, return_dict=True, output_hidden_states = True)
        p_hidden = psg_out.hidden_states[-self.num_layers]
        return p_hidden,psg['attention_mask']

    def encode_higher(self, a_embeddings, b_embeddings, a_attention_mask, b_attention_mask, train_flag=True, is_hard_msmarco=False):
        device = a_embeddings.device

        a_max_seq_len = a_attention_mask.shape[1]
        query_num = a_attention_mask.shape[0]

        candidate_num = int(b_embeddings.shape[0] / a_embeddings.shape[0])

        # repeat A
        # print(a_embeddings.shape, b_embeddings.shape, a_attention_mask.shape, b_attention_mask.shape)
        # raise()

        a_embeddings = a_embeddings.unsqueeze(1)
        a_embeddings = a_embeddings.repeat(1, candidate_num, 1, 1).reshape(query_num*candidate_num, a_max_seq_len,
                                                                           a_embeddings.shape[-1])
        a_attention_mask = a_attention_mask.unsqueeze(1)
        a_attention_mask = a_attention_mask.repeat(1, candidate_num, 1).reshape(query_num*candidate_num, a_max_seq_len)


        final_attention_mask = torch.cat((a_attention_mask, b_attention_mask.to(device)), dim=-1)
        final_hidden_states = torch.cat((a_embeddings, b_embeddings.to(device)), dim=1)

        # process attention mask
        input_shape = final_attention_mask.size()
        final_attention_mask = self.lm_q.get_extended_attention_mask(final_attention_mask, input_shape, device)

        for layer_module in self.higher_encoder:
            layer_outputs = layer_module(
                final_hidden_states,
                final_attention_mask
            )
            final_hidden_states = layer_outputs[0]
        return final_hidden_states

    # def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, train_flag = True):

    #     q_reps, q_attention_mask = self.encode_lower(query)
    #     p_reps, p_attention_mask = self.encode_lower(passage)
    #     joint_embeddings = self.encode_higher(a_embeddings=q_reps,
    #                                            b_embeddings=p_reps,
    #                                            a_attention_mask=q_attention_mask,
    #                                            b_attention_mask=p_attention_mask)

    #     # encoding a
    #     # a_lower_encoded_embeddings, _ = self.lower_encoding(input_ids=a_input_ids,
    #                                                         # attention_mask=a_attention_mask,
    #                                                         # token_type_ids=a_token_type_ids)

    #     # # encoding b
    #     # candidate_seq_len = b_input_ids.shape[-1]

    #     # b_input_ids = b_input_ids.reshape(-1, candidate_seq_len)
    #     # b_token_type_ids = b_token_type_ids.reshape(-1, candidate_seq_len)
    #     # b_attention_mask = b_attention_mask.reshape(-1, candidate_seq_len)

    #     # b_token_type_ids = b_token_type_ids + 1

    #     # b_input_ids, b_attention_mask, b_token_type_ids = clean_input_ids(b_input_ids, b_attention_mask,
    #     #                                                                   b_token_type_ids)

    #     # b_lower_encoded_embeddings, _ = self.lower_encoding(input_ids=b_input_ids,
    #     #                                                     attention_mask=b_attention_mask,
    #     #                                                     token_type_ids=b_token_type_ids)
    #     # #print("a_input_ids:", a_input_ids.shape)
    #     # #print("b_input_ids:", b_input_ids.shape)
    #     # # encoding together
    #     # joint_embeddings = self.joint_encoding(a_embeddings=a_lower_encoded_embeddings,
    #                                         #    b_embeddings=b_lower_encoded_embeddings,
    #                                         #    a_attention_mask=a_attention_mask,
    #                                         #    b_attention_mask=b_attention_mask,
    #                                         #    train_flag=train_flag, is_hard_msmarco=is_hard_msmarco)
    #     pooler = joint_embeddings[:,0,:]
    #     logits = self.classifier(pooler).squeeze(-1)

    #     #print("logits:", logits.shape)
    #     # if train_flag:
    #     logits = logits.reshape(q_reps.shape[0], -1)
    #     # if is_hard_msmarco:
    #     #     # for each query in batch, gold is the first candidate
    #     #     batch_size, candidate_num_per_query = logits.size()

    #     #     # Create a mask
    #     #     # Initialize the mask with zeros
    #     #     mask = torch.zeros(batch_size, candidate_num_per_query)

    #     #     # Set the first element of each row in the mask to 1
    #     #     mask[:, 0] = 1

    #     #     # Ensure mask is on the same device as score
    #     #     mask = mask.to(logits.device)
    #     # else:
    #     mask = torch.eye(logits.size(0)).to(logits.device)
    #     loss = torch.nn.functional.log_softmax(logits, dim=-1) * mask
    #     loss = (-loss.sum(dim=1)).mean()
    #     return EncoderOutput( 
    #         loss = loss,
    #         scores = logits,
    #         q_reps = q_reps,
    #         p_reps = p_reps
    #     )
    #     # else:
    #     #     logits = logits.reshape(a_input_ids.shape[0], -1)
    #     #     #print("reshaped logits:", logits.shape)
    #     #     return logits

class MixEncoderModel(nn.Module):
    TRANSFORMER_CLS = MyBertModel

    def __init__(self, args, encoder = None):
        super(MixEncoderModel, self).__init__()
        self.args = args
        self.num_labels = 1
        self.context_num = 1
        if self.args.type_bert == 'base':
            self.sentence_embedding_len = 768
            self.this_bert_config = BertConfig.from_pretrained('Luyu/co-condenser-marco')
            self.bert_model = MyBertModel.from_pretrained('Luyu/co-condenser-marco')
        elif self.args.type_bert == 'large':
            self.sentence_embedding_len = 1024
            self.this_bert_config = BertConfig.from_pretrained('bert-large-uncased')
            self.bert_model = MyBertModel.from_pretrained('bert-large-uncased')
        self.bert_model.resize_token_embeddings(30522)

        # used to compressing candidate to generate context vectors
        self.composition_layer = ContextLayer(self.this_bert_config.hidden_size, 1)

        # create models for decoder
        self.num_attention_heads = self.this_bert_config.num_attention_heads
        self.attention_head_size = int(self.this_bert_config.hidden_size / self.this_bert_config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.decoder = nn.ModuleDict({
            # used by candidate context embeddings to enrich themselves
            'candidate_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            'candidate_key': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                            range(self.this_bert_config.num_hidden_layers)]),
            'candidate_value': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                              range(self.this_bert_config.num_hidden_layers)]),
            # used to project representations into same space
            'layer_chunks': nn.ModuleList(
                [DecoderLayerChunk(self.this_bert_config) for _ in range(self.this_bert_config.num_hidden_layers)]),
            # used to compress candidate context embeddings into a vector
            'candidate_composition_layer': MeanLayer(),
            # used to generate query to compress Text_A thus get its representation
            'compress_query': nn.ModuleList([nn.Linear(self.this_bert_config.hidden_size, self.all_head_size) for _ in
                                             range(self.this_bert_config.num_hidden_layers)]),
            # used to update hidden states with self-att output as input
            'LSTM': MyLSTMBlock(self.this_bert_config.hidden_size),
        })

        # control which layer use decoder
        self.enrich_flag = [False] * self.this_bert_config.num_hidden_layers
        # self.enrich_flag[0] = True
        self.enrich_flag[-1] = True
        print("*" * 50)
        print(f"Enrich Layer: {self.enrich_flag}")
        print("*" * 50)

        self.init_decoder_params()

        # remove no use component
        for index, flag in enumerate(self.enrich_flag):
            if not flag:
                for key in ['layer_chunks', 'candidate_query', 'compress_query', 'candidate_key', 'candidate_value']:
                    self.decoder[key][index] = None

    # use parameters (layer chunk, q, q_1, k, v) of a pre-trained encoder to initialize the decoder
    def init_decoder_params(self):
        #  read parameters for layer chunks from encoder
        for index in range(self.this_bert_config.num_hidden_layers):
            this_static = self.decoder['layer_chunks'][index].state_dict()
            pretrained_static = self.bert_model.encoder.layer[index].state_dict()

            updating_layer_static = {}
            for k in this_static.keys():
                new_k = k.replace('attention.', 'attention.output.', 1)
                updating_layer_static[k] = pretrained_static[new_k]

            this_static.update(updating_layer_static)
            self.decoder['layer_chunks'][index].load_state_dict(this_static)

        #  read parameters for query from encoder
        pretrained_static = self.bert_model.state_dict()
        query_static = self.decoder['candidate_query'].state_dict()

        updating_query_static = {}
        for k in query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        query_static.update(updating_query_static)
        self.decoder['candidate_query'].load_state_dict(query_static)

        #  read parameters for compress_query from encoder
        compress_query_static = self.decoder['compress_query'].state_dict()

        updating_query_static = {}
        for k in compress_query_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.query.' + k.split(".")[1]
            updating_query_static[k] = pretrained_static[full_key]

        compress_query_static.update(updating_query_static)
        self.decoder['compress_query'].load_state_dict(compress_query_static)

        #  read parameters for key from encoder
        key_static = self.decoder['candidate_key'].state_dict()

        updating_key_static = {}
        for k in key_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.key.' + k.split(".")[1]
            updating_key_static[k] = pretrained_static[full_key]

        key_static.update(updating_key_static)
        self.decoder['candidate_key'].load_state_dict(key_static)

        #  read parameters for value from encoder
        value_static = self.decoder['candidate_value'].state_dict()

        updating_value_static = {}
        for k in value_static.keys():
            full_key = 'encoder.layer.' + k.split(".")[0] + '.attention.self.value.' + k.split(".")[1]
            updating_value_static[k] = pretrained_static[full_key]

        value_static.update(updating_value_static)
        self.decoder['candidate_value'].load_state_dict(value_static)

    def prepare_candidates(self, input_ids, token_type_ids, attention_mask):
        """ Pre-compute representations. Return shape: (-1, context_num, context_dim) """

        # reshape and encode candidates, (all_candidate_num, dim)
        candidate_seq_len = input_ids.shape[-1]
        input_ids = input_ids.reshape(-1, candidate_seq_len)
        token_type_ids = token_type_ids.reshape(-1, candidate_seq_len)
        attention_mask = attention_mask.reshape(-1, candidate_seq_len)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              enrich_candidate_by_question=[False]*self.this_bert_config.num_hidden_layers)
        last_hidden_state = out['last_hidden_state']

        # get (all_candidate_num, context_num, dim)
        candidate_embeddings = self.composition_layer(last_hidden_state, attention_mask=attention_mask)

        return candidate_embeddings

    def do_queries_match(self, input_ids, token_type_ids, attention_mask, candidate_context_embeddings, **kwargs):
        """ use pre-compute candidate to get scores."""
        # used to control ablation
        no_aggregator = kwargs.get('no_aggregator', False)
        no_enricher = kwargs.get('no_enricher', False)

        # (query_num (batch_size), candidate_num, context_num, embedding_len)
        candidate_num = candidate_context_embeddings.shape[1]
        # reshape to (query_num, candidate_context_num, embedding_len)
        this_candidate_context_embeddings = candidate_context_embeddings.reshape(candidate_context_embeddings.shape[0],
                                                                                 -1,
                                                                                 candidate_context_embeddings.shape[-1])

        lstm_initial_state = torch.zeros(this_candidate_context_embeddings.shape[0], candidate_num, this_candidate_context_embeddings.shape[-1],
                                         device=this_candidate_context_embeddings.device)

        # (query_num, candidate_context_num + candidate_num, dim)
        this_candidate_input_embeddings = torch.cat((this_candidate_context_embeddings, lstm_initial_state), dim=1)

        input_ids, attention_mask, token_type_ids = clean_input_ids(input_ids, attention_mask, token_type_ids)

        a_out = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                enrich_candidate_by_question=self.enrich_flag, candidate_embeddings=this_candidate_input_embeddings,
                                decoder=self.decoder, candidate_num=candidate_num)

        a_last_hidden_state, decoder_output = a_out['last_hidden_state'], a_out['decoder_output']

        # get final representations
        # control ablation
        if no_enricher:
            candidate_embeddings = self.decoder['candidate_composition_layer'](
                candidate_context_embeddings).squeeze(-2)
        else:
            # (query_num, candidate_num, context_num, dim)
            new_candidate_context_embeddings = decoder_output[:, :-candidate_num, :].reshape(decoder_output.shape[0],
                                                                                             candidate_num,
                                                                                             self.context_num,
                                                                                             decoder_output.shape[-1])

            # (query_num, candidate_num, dim)
            candidate_embeddings = self.decoder['candidate_composition_layer'](new_candidate_context_embeddings).squeeze(-2)

        if no_aggregator:
            # (query_num, 1, dim)
            query_embeddings = a_last_hidden_state[:, 0, :].unsqueeze(1)
            dot_product = torch.matmul(query_embeddings, candidate_embeddings.permute(0, 2, 1)).squeeze(-2)
        else:
            # (query_num, candidate_num, dim)
            query_embeddings = decoder_output[:, -candidate_num:, :]
            # (query_num, candidate_num)
            dot_product = torch.mul(query_embeddings, candidate_embeddings).sum(-1)

        return dot_product
        