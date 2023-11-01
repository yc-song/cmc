from transformers import BertConfig
import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss, BCELoss
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool
import wandb
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
def distillation_loss(student_outputs, teacher_outputs, labels, alpha = 0.5, temperature = 1, valid = False):
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
        self.num_heads = num_heads
        self.num_layers = num_layers
        # define transformer encoder
        if self.embed_dim is None: self.embed_dim = 768
        self.extend_multi_transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
        if args.identity_bert:
            self.extend_multi_transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads, args = args).to(self.device)
        self.args = args
        self.extend_multi_transformerencoder = torch.nn.TransformerEncoder(self.extend_multi_transformerencoderlayer, self.num_layers).to(self.device)
        self.extend_multi_linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
        self.extend_multi_token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)
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
            num_in_batch = 256
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
                nearest_candidate_token_ids = None, nearest_candidate_masks = None):
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

        if self.evaluate_on:  # This option is not used for extend_multi models
            mention_embeds, mention_embeds_masks = self.encode(
                mention_token_ids, mention_masks, None, None)[:2]
            bsz, l_x, mention_dim = mention_embeds.size()
            num_cands, l_y, cand_dim = self.candidates_embeds.size()
            if self.attention_type == 'soft_attention':
                scores = self.attention(self.candidates_embeds.unsqueeze(0).to(
                    self.device), mention_embeds, mention_embeds,
                    mention_embeds_masks)
            elif self.attention_type == "extend_multi":
                scores = None
                next_round_idxs = None
                num_stages = int(math.log10(num_cands)/math.log10(4))
                top_k = 64
                for stage in range(num_stages):
                    if stage == 0:
                        for i in range(int(num_cands/top_k)+1):
                            tmp_candidates_embed = self.candidates_embeds[top_k*i:min(top_k*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1)
                            if i == 0:
                                scores = self.extend_multi(mention_embeds, tmp_candidates_embed, num_mention_vecs = self.num_mention_vecs)
                                scores = torch.nn.functional.softmax(scores, dim = 1)
                                _, next_round_idxs = torch.topk(scores, int(top_k/4))
                            else:
                                tmp_scores = self.extend_multi(mention_embeds, tmp_candidates_embed, num_mention_vecs = self.num_mention_vecs)
                                tmp_scores = torch.nn.functional.softmax(tmp_scores, dim = 1)
                                scores = torch.cat([scores, tmp_scores], dim = 1)
                                _, tmp_next_round_idxs = torch.topk(tmp_scores, int(tmp_candidates_embed.size(1)/4))
                                tmp_next_round_idxs = tmp_next_round_idxs + torch.tensor([top_k*i], device = self.device)
                                next_round_idxs = torch.cat([next_round_idxs, tmp_next_round_idxs.squeeze(-1)], dim = 1)
                    else:
                        num_cands = next_round_idxs.size(1)
                        if num_cands <= 64: break
                        tmp_cls_cands = self.candidates_embeds[next_round_idxs.cpu()]
                        previous_round_idxs = next_round_idxs
                        for i in range(int(num_cands/top_k)+1):
                            tmp_candidates_embed = tmp_cls_cands[:,top_k*i:min(top_k*(i+1),num_cands), :]
                            if i == 0:
                                tmp_scores = self.extend_multi(mention_embeds, tmp_candidates_embed, num_mention_vecs = self.num_mention_vecs)
                                tmp_scores = torch.nn.functional.softmax(tmp_scores, dim = 1)
                                _, tmp_next_round_idxs = torch.topk(tmp_scores, int(tmp_candidates_embed.size(1)/4))
                                # next_round_idxs = torch.index_select(previous_round_idx, 1, next_round_idxs)
                                next_round_idxs = torch.gather(previous_round_idxs, 1, tmp_next_round_idxs)
                                # next_round_idxs = previous_round_idx[tmp_next_round_idxs]
                            else:
                                tmp_scores2 = self.extend_multi(mention_embeds, tmp_candidates_embed, num_mention_vecs = self.num_mention_vecs)
                                tmp_scores2 = torch.nn.functional.softmax(tmp_scores2, dim = 1)
                                tmp_scores = torch.cat([tmp_scores, tmp_scores2], dim = 1)
                                _, tmp_next_round_idxs = torch.topk(tmp_scores2, int(tmp_candidates_embed.size(1)/4))
                                tmp_next_round_idxs = tmp_next_round_idxs + torch.tensor([top_k*i], device = self.device)
                                tmp_next_round_idxs = torch.gather(previous_round_idxs, 1, tmp_next_round_idxs)
                                next_round_idxs = torch.cat([next_round_idxs, tmp_next_round_idxs], dim = 1)
                        if sampling:
                            for row in range(candidates_embeds["embeds"].size(0)):
                                scores[row, candidates_embeds["idxs"][row]] *= (round-1)/round
                                scores[row, candidates_embeds["idxs"][row]] += new_scores[row]/round
                        else:
                            for row in range(candidates_embeds["embeds"].size(0)):
                                scores[row, candidates_embeds["idxs"][row]] += new_scores[row]                                

            elif self.attention_type == "mlp_with_som":
                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids,
                                                candidate_masks)
                                                # candidate_token_ids[:,:args.num_training_cands,:],
                                                # candidate_masks[:,:args.num_training_cands,:])
                scores = self.mlp_with_som(mention_embeds, candidates_embeds)
            else:
                # candidate shape: (entity_bsz, 1, embed_dim)
                # score shape: (mention_bsz, entity_bsz)
                scores = (
                    torch.matmul(mention_embeds.reshape(-1, mention_dim),
                                 self.candidates_embeds.reshape(-1,
                                                                cand_dim).t().to(
                                     self.device)).reshape(bsz, l_x,
                                                           num_cands,
                                                           l_y).max(-1)[
                        0]).sum(1)

            return scores
        else:  # train\
            if args.energy_model:
                self.attention_type= "extend_multi_dot"
                self.num_mention_vecs = 1
                self.num_entity_vecs = 1
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
                # Deprecated
                if recall_eval:
                    round = 1
                    args.too_large = True
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks, too_large = args.too_large)
                    # Get the row and column index each 
                    i_indices = torch.arange(candidates_embeds.size(0)).view(-1, 1, 1, 1).float().expand(-1, candidates_embeds.size(1), -1, -1).to(self.device) # row index e.g. 0~3 for batch size 4
                    j_indices = torch.arange(candidates_embeds.size(1)).view(1, -1, 1, 1).float().expand(candidates_embeds.size(0), -1, -1, -1).to(self.device) # colum index i.e. its position inside batch
                    # Expand these indices to match the shape of the original tensor
                    candidates_embeds = torch.cat((candidates_embeds, i_indices, j_indices), dim=-1) # concatenating row and column index at the end.
                    candidates_reshaped = candidates_embeds.reshape(-1, top_k, 1, candidates_embeds.size(-1)) # For processing through the model which is optimized for # of candidates 64, reshape the input shape. 
                    scores_round_0 = self.forward_chunk(mention_embeds, candidates_reshaped[:,:,:,:-2], dot = dot, args = args)  # (B*16, top_k)
                    candidates_advanced_idxs = torch.topk(scores_round_0, int(top_k*beam_ratio))[1]
                    candidates_advanced = torch.gather(candidates_reshaped, 1, candidates_advanced_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, candidates_embeds.size(-1))) 
                    while torch.numel(candidates_advanced_idxs)//B > top_k:
                        candidates_advanced = candidates_advanced.reshape(-1, top_k, 1, candidates_advanced.size(-1))
                        scores = self.forward_chunk(mention_embeds, candidates_advanced[:,:,:,:-2], dot = dot, args = args)
                        candidates_advanced_idxs = torch.topk(scores, int(top_k*beam_ratio))[1]
                        candidates_advanced = torch.gather(candidates_advanced, 1, candidates_advanced_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, candidates_embeds.size(-1))) 
                    candidates_advanced = candidates_advanced.reshape(-1, top_k, 1, candidates_advanced.size(-1))
                    scores = self.forward_chunk(mention_embeds, candidates_advanced[:,:,:,:-2], dot = dot, args = args)
                    candidates_advanced_idxs = torch.argsort(scores, descending = True)
                    candidates_advanced = torch.gather(candidates_advanced, 1, candidates_advanced_idxs.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, candidates_embeds.size(-1))) 
                    return candidates_advanced[:,:,0,-1]
                else:
                    # Get embeddings
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids,
                                                candidate_masks)
                    # Deprecated
                    if args.case_based: 
                        # ToDo
                        # make extend_multi to take nearest mention and label as input, and process them as intended.
                        # also make self.encode to encode nearest mention and gold
                        nearest_mention_embeds, nearest_mention_embeds_masks, \
                        nearest_gold_embeds = self.encode(nearest_mention_token_ids.squeeze(0), nearest_mention_masks.squeeze(0),
                                                    nearest_label_token_ids,
                                                    nearest_label_masks)
                        nearest_gold_embeds = nearest_gold_embeds.squeeze(0)
                        scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args, nearest_mention = nearest_mention_embeds, nearest_gold = nearest_gold_embeds)
                    elif args.nearest:
                        # mention: (B, N, D) 
                        # entity: (B, C, N, D) -> (B, C*N, D)
                        nearest_candidate_token_ids = nearest_candidate_token_ids[:,:,:3,:]
                        nearest_candidate_masks = nearest_candidate_masks[:,:,:3,:]
                        nearest_mention_token_ids = nearest_mention_token_ids[:,:3,:]
                        nearest_mention_masks = nearest_mention_masks[:,:3,:]
                        # nearest_candidate_token_ids = nearest_candidate_token_ids.reshape(B, -1, L)
                        # nearest_candidate_masks = nearest_candidate_token_ids.reshape(B, -1, nearest_candidate_token_ids.size(-1))
                        # nearest_mention_embeds, nearest_mention_embeds_masks, \
                        # nearest_candidates_embeds = self.encode(nearest_mention_token_ids, nearest_mention_masks,
                        #                             nearest_candidate_token_ids,
                        #                             nearest_candidate_masks)
                        # nearest_candidates_embeds = nearest_candidates_embeds.reshape(B, C, -1, nearest_candidates_embeds.size(-1))
                        scores = self.extend_multi(mention_embeds, candidates_embeds, self, dot, args, \
                        nearest_mention_token_ids = nearest_mention_token_ids, nearest_mention_masks = nearest_mention_masks, \
                        nearest_candidate_token_ids = nearest_candidate_token_ids, nearest_candidate_masks = nearest_candidate_masks)

                    else:
                        # Take embeddings as input for extend_multi model
                        scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args)

            elif self.attention_type == 'mlp_with_som':
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids,
                                                candidate_masks)
                    scores = self.mlp_with_som(mention_embeds, candidates_embeds)
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
                labels = label_idx.to(self.device)
            if candidate_probs is not None:  # logits adjustment
                candidate_probs = candidate_probs.to(self.device)
                scores[:, 1:] -= ((C - 1) * candidate_probs).log()
            if args.distill_training:
                loss, student_loss, teacher_loss = distillation_loss(scores, teacher_scores, labels, args.alpha)
                return loss, predicts, scores, student_loss, teacher_loss
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
                return loss, predicts, scores
    def extend_multi(self, xs, ys, dot = False, args = None, num_mention_vecs = 1, nearest_mention = None, nearest_gold = None,\
     nearest_mention_token_ids = None, nearest_mention_masks = None, nearest_candidate_token_ids = None, nearest_candidate_masks = None):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        # Deprecated
        if self.args.case_based:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                xs = torch.cat([xs, nearest_mention], dim = 0)
                nearest_gold = nearest_gold.expand(-1, args.C_eval, -1)
                ys = torch.cat([ys, nearest_gold], dim = 0)
                input = torch.cat([xs, ys], dim = 1)
                attention_result = self.extend_multi_transformerencoder(input)
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.extend_multi.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])
            return scores_overall
        # Deprecated
        elif self.args.attend_to_gold:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                nearest_gold = torch.cat((ys[:i, 0:1, :], ys[i+1:, 0:1, :]))
                nearest_gold = nearest_gold.expand(-1, args.C_eval, -1)

                ys_new = torch.cat([ys[i:i+1,:,:], nearest_gold], dim = 0)
                xs_new = torch.cat([xs[i:i+1, :, :], xs[:i, :, :], xs[i+1:, :, :]])
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])

            return scores_overall
        # Deprecated
        elif self.args.order_changed:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                xs_new = torch.cat([xs[i:i+1, :, :], xs[:i, :, :], xs[i+1:, :, :]])
                ys_new = torch.cat([ys[i:i+1, :, :], ys[:i, :, :], ys[i+1:, :, :]])
                temp = ys_new[0,0,:].clone()
                ys_new[0,0,:] = ys_new[0,16,:]
                ys_new[0,16,:] = temp
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])                

            return scores_overall  
         
        elif self.args.nearest:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                # for B times
                # taking (N, 1, D) + (N, C, D) pair as the input
                # nearest_mention_token_ids: (N, L)
                # nearest_mention_token_ids: (C, N, L)
                # print("-1", nearest_mention_token_ids[i].shape, nearest_mention_masks[i].shape)
                # print("0", nearest_candidate_token_ids[i].shape, nearest_candidate_masks[i].shape)
                nearest_mention_embeds, nearest_mention_embeds_masks, \
                nearest_candidates_embeds = self.encode(nearest_mention_token_ids[i], nearest_mention_masks[i],
                                            nearest_candidate_token_ids[i],
                                            nearest_candidate_masks[i])                
                # print("1", xs[i].shape)
                # print("2", nearest_mention_embeds.shape)
                # print("3", ys[i].shape)
                # print("4", nearest_candidates_embeds.shape)

                xs_new = torch.cat((xs[i], nearest_mention_embeds.squeeze(-2)), dim = 0).unsqueeze(1) # (4, 1, 768)
                ys_new = torch.cat((ys[i].unsqueeze(-2), nearest_candidates_embeds.squeeze(2)), dim = 1).transpose(1,0) # (64, 4, 768)
                # print("5", xs_new.shape)
                # print("6", ys_new.shape)
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.extend_multi_transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,1:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])                

            return scores_overall      
        # Dprecated                  
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
        self.embed_dim = 768
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
        # Deprecated
        if self.args.case_based:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                xs = torch.cat([xs, nearest_mention], dim = 0)
                nearest_gold = nearest_gold.expand(-1, args.C_eval, -1)
                ys = torch.cat([ys, nearest_gold], dim = 0)
                input = torch.cat([xs, ys], dim = 1)
                attention_result = self.transformerencoder(input)
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])
            return scores_overall
        # Deprecated
        elif self.args.attend_to_gold:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                nearest_gold = torch.cat((ys[:i, 0:1, :], ys[i+1:, 0:1, :]))
                nearest_gold = nearest_gold.expand(-1, args.C_eval, -1)
                ys_new = torch.cat([ys[i:i+1,:,:], nearest_gold], dim = 0)
                xs_new = torch.cat([xs[i:i+1, :, :], xs[:i, :, :], xs[i+1:, :, :]])
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])

            return scores_overall
        elif self.args.attend_to_itself:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                xs_new = xs[i:i+1,:,:]
                xs_new = xs_new.expand(xs.size(0), -1, -1)
                ys_new = ys[i:i+1,:,:]
                ys_new = ys_new.expand(ys.size(0), -1, -1)
                print(xs_new, xs_new.shape, ys_new, ys_new.shape)

                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])

            return scores_overall
        # Deprecated
        elif self.args.order_changed:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                xs_new = torch.cat([xs[i:i+1, :, :], xs[:i, :, :], xs[i+1:, :, :]])
                ys_new = torch.cat([ys[i:i+1, :, :], ys[:i, :, :], ys[i+1:, :, :]])
                temp = ys_new[0,0,:].clone()
                ys_new[0,0,:] = ys_new[0,16,:]
                ys_new[0,16,:] = temp
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])                

            return scores_overall  
         
        elif self.args.nearest:
            scores_overall = torch.tensor([]).to(self.device)
            for i in range(xs.size(0)):
                # for B times
                # taking (N, 1, D) + (N, C, D) pair as the input
                # nearest_mention_token_ids: (N, L)
                # nearest_mention_token_ids: (C, N, L)
                print("-1", nearest_mention_token_ids[i].shape, nearest_mention_masks[i].shape)
                print("0", nearest_candidate_token_ids[i].shape, nearest_candidate_masks[i].shape)
                nearest_mention_embeds, nearest_mention_embeds_masks, \
                nearest_candidates_embeds = unifiedretriever.encode(nearest_mention_token_ids[i], nearest_mention_masks[i],
                                            nearest_candidate_token_ids[i],
                                            nearest_candidate_masks[i])                
                print("1", xs[i].shape)
                print("2", nearest_mention_embeds.shape)
                print("3", ys[i].shape)
                print("4", nearest_candidates_embeds.shape)

                xs_new = torch.cat((xs[i].unsqueeze(-2), nearest_mention_embeds.squeeze(-2)), dim = 0)
                nearest_candidates_embeds = nearest_candidates_embeds.transpose(1,0)
                ys_new = torch.cat((ys[i], nearest_candidates_embeds), dim = 0)
                print("5", xs_new.shape)
                print("6", ys_new.shape)
                input = torch.cat([xs_new, ys_new], dim = 1)
                attention_result = self.transformerencoder(input)
                
                if dot:
                    scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,args.num_mention_vecs:,:].transpose(2,1))
                    scores = scores.squeeze(-2)
                else:
                    scores = self.linearhead(attention_result[:,args.num_mention_vecs:,:])
                    scores = scores.squeeze(-1)
                scores_overall = torch.cat([scores_overall, scores[0:1]])                

            return scores_overall      
        # Dprecated                  
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
        return out
