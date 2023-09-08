from transformers import BertConfig
import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss
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

def distillation_loss(student_outputs, teacher_outputs, labels, alpha = 0.5):
    teacher_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs, dim=1),
                             nn.functional.softmax(teacher_outputs, dim=1))
    student_loss = nn.CrossEntropyLoss(student_outputs, labels)
    loss = alpha*student_loss + (1-alpha)*teacher_loss
    return loss

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
        self.extend_multi = extend_multi(num_heads, num_layers, args)
        if args.type_model == 'mlp':
            self.mlp = mlp(args)
        if args.type_model == 'mlp_with_som':
            self.mlp_with_som = mlp_with_som(args)

    def encode(self, mention_token_ids, mention_masks, candidate_token_ids,
               candidate_masks, entity_token_ids=None, entity_masks=None, too_large = False, entity_bsz=64):
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
            for i in range(candidate_token_ids.size(1)//64):
                candidate_embed = candidates_encoding(candidate_token_ids[:,64*i:64*(i+1),:], candidate_masks[:,64*i:64*(i+1),:]).cpu()
                candidates_embeds = torch.cat((candidates_embeds, candidate_embed), dim = 1)
            candidates_embeds = candidates_embeds.to(device)
        if mention_token_ids is not None:
            mention_token_ids = mention_token_ids.to(self.device).long()
            mention_masks = mention_masks.to(self.device).long()
            mention_hiddens = self.mention_encoder(input_ids=mention_token_ids,
                                                   attention_mask=mention_masks)[
                0]
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
                candidate_masks, teacher_scores = None, candidate_probs=None, recall_eval = False, top_k = 64, beam_ratio = None, args = None, sampling = False):
        if self.evaluate_on:  # evaluate or get candidates
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
                                                candidate_token_ids[:,:args.num_training_cands,:],
                                                candidate_masks[:,:args.num_training_cands,:])
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
            B, C, L = candidate_token_ids.size() # e.g. 8, 256, 128
            # B x m x d
            #  get  embeds
            # print(candidates_embeds.shape) # (1, 65, 128, 768)
            if self.attention_type == 'extend_multi' or self.attention_type == 'extend_multi_dot':
                if self.attention_type == 'extend_multi_dot':
                    dot = True
                else: 
                    dot = False
                ## if recall_eval is True, # of candidates is same as args.num_eval_cands
                ## else, # of candidates is same as args.num_training_cands
                if recall_eval:
                    round = 1
                    candidates_embeds = {}
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds["embeds"] = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks, too_large = args.too_large)
                    candidates_embeds["idxs"] = torch.arange(candidates_embeds["embeds"].size(1))\
                        .expand(candidates_embeds["embeds"].size(0), -1)
                    idxs = candidates_embeds["idxs"]
                    # scores = None
                    # num_chunks = C//top_k
                    # (16, 64, 1, 768)
                    processed_tensor = candidates_embeds["embeds"].reshape(-1, top_k, 1, candidates_embeds["embeds"].size(-1))
                    # idxs_chunks = candidates_embeds["idxs"].reshape(-1, top_k)
                    # print("***")
                    # print(processed_tensor[1], processed_tensor.shape)
                    scores = self.extend_multi.forward_chunk(mention_embeds, processed_tensor, dot, int(candidates_embeds["embeds"].size(1)/top_k))
                    # print(scores, scores.shape)
                    # scores = torch.nn.functional.softmax(scores, dim = 1)
                    scores = scores.view(B, -1, top_k) # (B, *, 64로 바꾼 다음에 나온 index에 대해 64씩을 더해줌)
                    idxs = torch.topk(scores, int(top_k*beam_ratio))[1]
                    cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
                    idxs += cumulative_idxs
                    idxs = idxs.view(B, int(C*beam_ratio))
                    batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
                    scores = scores.view(B, C)
                    candidates_embeds["embeds"] = candidates_embeds["embeds"][batch_idxs, idxs, :, :]
                    candidates_embeds["idxs"] = candidates_embeds["idxs"][batch_idxs, idxs.cpu()]
                    while idxs.size(1) >= top_k:
                        round += 1
                        processed_tensor = candidates_embeds["embeds"].reshape(-1, top_k, 1, candidates_embeds["embeds"].size(-1))
                        idxs_chunks = candidates_embeds["idxs"].reshape(-1, top_k)
                        # print(processed_tensor, processed_tensor.shape)
                        # print(idxs_chunks, idxs_chunks.shape)
                        new_scores = self.extend_multi.forward_chunk(mention_embeds, processed_tensor, dot, int(candidates_embeds["embeds"].size(1)/top_k))
                        # new_scores = torch.nn.functional.softmax(new_scores, dim = 1)
                        new_scores = new_scores.view(B, -1, top_k)
                        # print(new_scores.shape)
                        # print(new_scores, new_scores.shape)
                        idxs = torch.topk(new_scores, int(top_k*beam_ratio))[1]
                        # print(idxs, previous_idxs)
                        cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
                        idxs += cumulative_idxs
                        idxs = idxs.view(B, int(candidates_embeds["embeds"].size(1)*beam_ratio))
                        batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
                        new_scores = new_scores.view(B, candidates_embeds["embeds"].size(1))
                        if sampling:
                            for row in range(candidates_embeds["embeds"].size(0)):
                                scores[row, candidates_embeds["idxs"][row]] *= (round-1)/round
                                scores[row, candidates_embeds["idxs"][row]] += new_scores[row]/round
                        else:
                            for row in range(candidates_embeds["embeds"].size(0)):
                                scores[row, candidates_embeds["idxs"][row]] += new_scores[row]

                        candidates_embeds["embeds"] = candidates_embeds["embeds"][batch_idxs, idxs, :, :]
                        candidates_embeds["idxs"] = candidates_embeds["idxs"][batch_idxs, idxs.cpu()]

                        
                else:
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids[:,:args.num_training_cands,:],
                                                candidate_masks[:,:args.num_training_cands,:])
                    scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args, num_mention_vecs = self.num_mention_vecs)
            elif self.attention_type == 'mlp_with_som':
                    mention_embeds, mention_embeds_masks, \
                    candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids[:,:args.num_training_cands,:],
                                                candidate_masks[:,:args.num_training_cands,:])
                    scores = self.mlp_with_som(mention_embeds, candidates_embeds)
            else:
                mention_embeds, mention_embeds_masks, \
                candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                                candidate_token_ids[:,:args.num_training_cands,:],
                                                candidate_masks[:,:args.num_training_cands,:])
                if self.attention_type == 'soft_attention':
                    scores = self.attention(candidates_embeds, mention_embeds,
                                            mention_embeds, mention_embeds_masks)
                elif self.attention_type == 'mlp':
                    scores = self.mlp(mention_embeds, candidates_embeds)
                else:
                    scores = self.attention(mention_embeds, candidates_embeds)
            predicts = scores.argmax(1)
            labels = torch.zeros(B).long().to(self.device)
            if candidate_probs is not None:  # logits adjustment
                candidate_probs = candidate_probs.to(self.device)
                scores[:, 1:] -= ((C - 1) * candidate_probs).log()
            if args.distill_training:
                loss = distillation_loss(scores, teacher_scores, labels)
            else:
                loss = self.loss_fct(scores, labels)
            return loss, predicts, scores

class FrozenRetriever(UnifiedRetriever):
    def __init__(self, encoder, device, num_codes_mention, num_codes_entity,
                 mention_use_codes, entity_use_codes, attention_type,
                 candidates_embeds=None, evaluate_on=False, args= None, num_heads = 2, num_layers = 2):
        super(FrozenRetriever, self).__init__(encoder, device, num_codes_mention, num_codes_entity,
                 mention_use_codes, entity_use_codes, attention_type, candidates_embeds, evaluate_on, args, num_heads, num_layers)
        if args.freeze_bert: 
            for param in self.mention_encoder.parameters():
                param.requires_grad = False
            for param in self.entity_encoder.parameters():
                param.requires_grad = False
    def forward(self, mention_embeds, mention_embeds_masks, candidates_embeds, candidates_embeds_masks\
    ,candidate_probs=None, recall_eval = False, top_k = 64, beam_ratio = None, args = None):
        if self.evaluate_on:  # evaluate or get candidates
            bsz, l_x, mention_dim = mention_embeds.size()
            num_cands, l_y, cand_dim = candidates_embeds.size()
            if self.attention_type == 'soft_attention':
                scores = self.attention(candidates_embeds.unsqueeze(0).to(
                    self.device), mention_embeds, mention_embeds,
                    mention_embeds_masks)
            else:
                # candidate shape: (entity_bsz, 1, embed_dim)
                # score shape: (mention_bsz, entity_bsz)
                scores = (
                    torch.matmul(mention_embeds.reshape(-1, mention_dim),
                                 candidates_embeds.reshape(-1,
                                                                cand_dim).t().to(
                                     self.device)).reshape(bsz, l_x,
                                                           num_cands,
                                                           l_y).max(-1)[
                        0]).sum(1)

                
            return scores
        else:  # train
            candidates_embeds = candidates_embeds.squeeze(-2)
            B, C, L = candidates_embeds.size() # e.g. 8, 256, 128
            # B x m x d
            #  get  embeds
            # print(candidates_embeds.shape) # (1, 65, 128, 768)
            if self.attention_type == 'extend_multi' or self.attention_type == 'extend_multi_dot':
                if self.attention_type == 'extend_multi_dot':
                    dot = True
                else: 
                    dot = False
                ## if recall_eval is True, # of candidates is same as args.num_eval_cands
                ## else, # of candidates is same as args.num_training_cands
                if recall_eval:
                    round = 1
                    candidates_embeds_dict = {}
                    candidates_embeds_dict["embeds"] = candidates_embeds
                    candidates_embeds_dict["idxs"] = torch.arange(candidates_embeds_dict["embeds"].size(1))\
                        .expand(candidates_embeds_dict["embeds"].size(0), -1)
                    idxs = candidates_embeds_dict["idxs"]
                    # scores = None
                    # num_chunks = C//top_k
                    # (16, 64, 1, 768)
                    processed_tensor = candidates_embeds_dict["embeds"].reshape(-1, top_k, 1, candidates_embeds_dict["embeds"].size(-1))
                    # idxs_chunks = candidates_embeds["idxs"].reshape(-1, top_k)
                    # print("***")
                    # print(processed_tensor[1], processed_tensor.shape)
                    scores = self.extend_multi.forward_chunk(mention_embeds, processed_tensor, dot, int(candidates_embeds_dict["embeds"].size(1)/top_k))
                    # print(scores, scores.shape)
                    # scores = torch.nn.functional.softmax(scores, dim = 1)
                    scores = scores.view(B, -1, top_k) # (B, *, 64로 바꾼 다음에 나온 index에 대해 64씩을 더해줌)
                    idxs = torch.topk(scores, int(top_k*beam_ratio))[1]
                    cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
                    idxs += cumulative_idxs
                    idxs = idxs.view(B, int(C*beam_ratio))
                    batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
                    scores = scores.view(B, C)
                    candidates_embeds_dict["embeds"] = candidates_embeds_dict["embeds"][batch_idxs, idxs, :, :]
                    candidates_embeds_dict["idxs"] = candidates_embeds_dict["idxs"][batch_idxs, idxs.cpu()]
                    while idxs.size(1) >= top_k:
                        round +=1
                        processed_tensor = candidates_embeds_dict["embeds"].reshape(-1, top_k, 1, candidates_embeds_dict["embeds"].size(-1))
                        idxs_chunks = candidates_embeds_dict["idxs"].reshape(-1, top_k)
                        new_scores = self.extend_multi.forward_chunk(mention_embeds, processed_tensor, dot, int(candidates_embeds_dict["embeds"].size(1)/top_k))
                        # new_scores = torch.nn.functional.softmax(new_scores, dim = 1)
                        new_scores = new_scores.view(B, -1, top_k)
                        # print(new_scores.shape)
                        # print(new_scores, new_scores.shape)
                        idxs = torch.topk(new_scores, int(top_k*beam_ratio))[1]
                        # print(idxs, previous_idxs)
                        cumulative_idxs = torch.tensor([top_k*i for i in range(idxs.size(1))], device = self.device).unsqueeze(1)
                        idxs += cumulative_idxs
                        idxs = idxs.view(B, int(candidates_embeds_dict["embeds"].size(1)*beam_ratio))
                        batch_idxs = torch.arange(B).unsqueeze(1).expand_as(idxs)
                        new_scores = new_scores.view(B, candidates_embeds_dict["embeds"].size(1))
                        for row in range(candidates_embeds_dict["embeds"].size(0)):
                            # print(scores[row, candidates_embeds["idxs"][row]], new_scores[row])
                            scores[row, candidates_embeds["idxs"][row]] *= (round-1)/round
                            scores[row, candidates_embeds["idxs"][row]] += new_scores[row]/round
                        candidates_embeds_dict["embeds"] = candidates_embeds_dict["embeds"][batch_idxs, idxs, :, :]
                        candidates_embeds_dict["idxs"] = candidates_embeds_dict["idxs"][batch_idxs, idxs.cpu()]
                else:
                    scores = self.extend_multi(mention_embeds, candidates_embeds, dot, args, self.num_mention_vecs)
            else:
                if self.attention_type == 'soft_attention':
                    scores = self.attention(candidates_embeds, mention_embeds,
                                            mention_embeds, mention_embeds_masks)
                elif self.attention_type == 'mlp':
                    scores = self.mlp(mention_embeds, candidates_embeds)
                else:
                    scores = self.attention(mention_embeds, candidates_embeds)
            predicts = scores.argmax(1)
            labels = torch.zeros(B).long().to(self.device)
            if candidate_probs is not None:  # logits adjustment
                candidate_probs = candidate_probs.to(self.device)
                scores[:, 1:] -= ((C - 1) * candidate_probs).log()
            loss = self.loss_fct(scores, labels)
            return loss, predicts, scores

class extend_multi(nn.Module):
    def __init__(self, num_heads, num_layers, args):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed_dim = 768
        # if args.anncur:
            # self.transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads).to(self.device)
        # else:
        # ToDo: modify dim_feedforward argument of self.transformerencoderlayer
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
        if args.identity_bert:
            self.transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads).to(self.device)

        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers).to(self.device)
        self.linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
        self.token_type_embeddings = nn.Embedding(2, self.embed_dim).to(self.device)
    def process_chunk(self, xs, ys, dot):
        # Process the chunk using your deep learning model
        processed_chunk = self.forward(xs, ys, dot)
        return processed_chunk
    def forward(self, xs, ys, dot = False, args = None, num_mention_vecs = 1):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        if args.model_top is None and args.token_type :
            token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
            token_embedding_xs = self.token_type_embeddings(token_type_xs)
            token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
            token_embedding_ys = self.token_type_embeddings(token_type_ys)
            input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        else:
            input = torch.cat([xs, ys], dim = 1)
        attention_result = self.transformerencoder(input)
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,num_mention_vecs:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        else:
            scores = self.linearhead(attention_result[:,num_mention_vecs:,:])
            scores = scores.squeeze(-1)
        return scores
    def forward_chunk(self, xs, ys, dot = False, size_chunk = 1):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, embed_dim)
        xs = xs.repeat_interleave(size_chunk, dim=0)
        token_type_xs = torch.zeros(xs.size(0), xs.size(1)).int().to(self.device)
        token_embedding_xs = self.token_type_embeddings(token_type_xs)
        token_type_ys = torch.ones(ys.size(0), ys.size(1)).int().to(self.device)
        token_embedding_ys = self.token_type_embeddings(token_type_ys)
        input = torch.cat([xs + token_embedding_xs, ys + token_embedding_ys], dim = 1)
        attention_result = self.transformerencoder(input)
        if dot:
            scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,1:,:].transpose(2,1))
            scores = scores.squeeze(-2)
        else:
            scores = self.linearhead(attention_result[:,1:,:])
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
    def __init__(self, d_model, n_head, dim_feedforward=3072, dropout=0.1, activation="relu"):
        super(IdentityInitializedTransformerEncoderLayer, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head)
        self.weight = nn.Parameter(torch.tensor([-5.], requires_grad = True))
    def forward(self,src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        out1 = self.encoder_layer(src, src_mask, src_key_padding_mask) # For Lower Torch Version
        # out1 = self.encoder_layer(src, src_mask, src_key_padding_mask, is_causal) # For Higher Torch Version
        sigmoid_weight = torch.sigmoid(self.weight)
        wandb.log({"weight": self.weight.item()})
        out = sigmoid_weight*out1 + (1-sigmoid_weight)*src
        return out
