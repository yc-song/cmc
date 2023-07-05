from transformers import BertConfig
import torch
import torch.nn as nn
import copy
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import math

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


class UnifiedRetriever(nn.Module):
    def __init__(self, encoder, device, num_codes_mention, num_codes_entity,
                 mention_use_codes, entity_use_codes, attention_type,
                 candidates_embeds=None, evaluate_on=False, num_heads = 2, num_layers = 2):
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
        self.extend_multi = extend_multi(num_heads, num_layers)

    def encode(self, mention_token_ids, mention_masks, candidate_token_ids,
               candidate_masks, entity_token_ids=None, entity_masks=None):
        candidates_embeds = None
        mention_embeds = None
        mention_embeds_masks = None
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
                candidate_masks, candidate_probs=None):
        if self.evaluate_on:  # evaluate or get candidates
            mention_embeds, mention_embeds_masks = self.encode(
                mention_token_ids, mention_masks, None, None)[:2]
            bsz, l_x, mention_dim = mention_embeds.size()
            num_cands, l_y, cand_dim = self.candidates_embeds.size()
            if self.attention_type == 'soft_attention':
                scores = self.attention(self.candidates_embeds.unsqueeze(0).to(
                    self.device), mention_embeds, mention_embeds,
                    mention_embeds_masks)

            if self.attention_type == "extend_multi":
                scores = None
                next_round_idxs = None
                num_stages = int(math.log10(num_cands)/math.log10(4))
                top_k = 64
                for stage in range(num_stages):
                    if stage == 0:
                        for i in range(int(num_cands/top_k)+1):
                            tmp_candidates_embed = self.candidates_embeds[top_k*i:min(top_k*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1)
                            if i == 0:
                                scores = self.extend_multi(mention_embeds, tmp_candidates_embed)
                                scores = torch.nn.functional.softmax(scores, dim = 1)
                                _, next_round_idxs = torch.topk(scores, int(top_k/4))
                            else:
                                tmp_scores = self.extend_multi(mention_embeds, tmp_candidates_embed)
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
                                tmp_scores = self.extend_multi(mention_embeds, tmp_candidates_embed)
                                tmp_scores = torch.nn.functional.softmax(tmp_scores, dim = 1)
                                _, tmp_next_round_idxs = torch.topk(tmp_scores, int(tmp_candidates_embed.size(1)/4))
                                # next_round_idxs = torch.index_select(previous_round_idx, 1, next_round_idxs)
                                next_round_idxs = torch.gather(previous_round_idxs, 1, tmp_next_round_idxs)
                                # next_round_idxs = previous_round_idx[tmp_next_round_idxs]
                            else:
                                tmp_scores2 = self.extend_multi(mention_embeds, tmp_candidates_embed)
                                tmp_scores2 = torch.nn.functional.softmax(tmp_scores2, dim = 1)
                                tmp_scores = torch.cat([tmp_scores, tmp_scores2], dim = 1)
                                _, tmp_next_round_idxs = torch.topk(tmp_scores2, int(tmp_candidates_embed.size(1)/4))
                                tmp_next_round_idxs = tmp_next_round_idxs + torch.tensor([top_k*i], device = self.device)
                                tmp_next_round_idxs = torch.gather(previous_round_idxs, 1, tmp_next_round_idxs)
                                next_round_idxs = torch.cat([next_round_idxs, tmp_next_round_idxs], dim = 1)
                                
                        # Perform element-wise addition using broadcasting
                        for row in range(previous_round_idxs.size(0)):
                            scores[row, previous_round_idxs[row]] += tmp_scores[row]
                # num_stages = math.ceil(math.log10(num_cands+1)/math.log10(65)) # num of stages e.g. 66 cands: 2 stages
                # # print(mention_embeds.shape, self.candidates_embeds.shape)
                # next_round_idxs=None
                # all_scores = None
                # for stage in range(num_stages):
                #     if stage == 0:
                #         for i in range(int(num_cands/65)+1):
                #             if i == 0:
                #                 tmp_scores = self.extend_multi(mention_embeds, self.candidates_embeds[65*i:min(65*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1))
                #                 min_values, _ = torch.min(tmp_scores, dim = 1)
                #                 min_values = min_values.unsqueeze(-1)
                #                 max_values, max_idxs = torch.max(tmp_scores, dim = 1)
                #                 next_round_idxs = max_idxs.unsqueeze(-1)
                #                 max_values = max_values.unsqueeze(-1)
                #                 tmp_scores = (tmp_scores-min_values)/(max_values-min_values)

                #                 # print(scores.shape)
                #             else:
                #                 tmp_scores2 = self.extend_multi(mention_embeds, self.candidates_embeds[65*i:min(65*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1))
                #                 min_values, _ = torch.min(tmp_scores2, dim = 1)
                #                 min_values = min_values.unsqueeze(-1)
                #                 max_values, max_idxs = torch.max(tmp_scores2, dim = 1)
                #                 max_idxs = max_idxs + torch.tensor([65*i], device = self.device)
                #                 next_round_idxs = torch.cat([next_round_idxs, max_idxs.unsqueeze(-1)], dim = 1)
                #                 max_values = max_values.unsqueeze(-1)
                #                 tmp_scores2 = (tmp_scores2-min_values)/(max_values-min_values)
                #                 tmp_scores = torch.cat([tmp_scores, tmp_scores2], dim = 1)
                #         scores = tmp_scores
                    # else:
                    #     candidates_embeds = self.candidates_embeds[next_round_idxs.cpu()]
                    #     previous_round_idxs = next_round_idxs
                    #     for i in range(int(next_round_idxs.size(1)/65)+1):
                    #         if i == 0:
                    #             tmp_scores = self.extend_multi(mention_embeds, candidates_embeds[:,65*i:min(65*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1))
                    #             min_values, _ = torch.min(tmp_scores, dim = 1)
                    #             min_values = min_values.unsqueeze(-1)
                    #             max_values, max_idxs = torch.max(tmp_scores, dim = 1)
                    #             next_round_idxs = max_idxs.unsqueeze(-1)
                    #             max_values = max_values.unsqueeze(-1)
                    #             tmp_scores = (tmp_scores-min_values)/(max_values-min_values)

                    #             # print(scores.shape)
                    #         else:
                    #             tmp_scores2 = self.extend_multi(mention_embeds, candidates_embeds[:,65*i:min(65*(i+1),num_cands), :, :].expand(mention_embeds.size(0), -1, -1, -1))
                    #             min_values, _ = torch.min(tmp_scores2, dim = 1)
                    #             min_values = min_values.unsqueeze(-1)
                    #             max_values, max_idxs = torch.max(tmp_scores2, dim = 1)
                    #             max_idxs = max_idxs + torch.tensor([65*i], device = self.device) 
                    #             next_round_idxs = torch.cat([next_round_idxs, max_idxs.unsqueeze(-1)], dim = 1)
                    #             max_values = max_values.unsqueeze(-1)
                    #             tmp_scores2 = (tmp_scores2-min_values)/(max_values-min_values)
                    #             tmp_scores = torch.cat([tmp_scores, tmp_scores2], dim = 1)
                    #     expanded_score = torch.zeros_like(scores)
                    #     print("-2", scores[0][:100], scores.shape)
                    #     print("-1.75", tmp_scores[0], tmp_scores.shape)
                    #     expanded_score = expanded_score.scatter_(1, previous_round_idxs, tmp_scores)
                    #     print("-1.5", expanded_score[0][:500], expanded_score.shape)
                    #     torch.set_printoptions(threshold=10_000)
                    #     print("-1", previous_round_idxs[0], previous_round_idxs.shape)
                    #     print("1", scores[0][:500], scores.shape)
                    #     scores = scores + expanded_score
                    #     print("2", scores[0][:500], scores.shape)
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
        else:  # train
            B, C, L = candidate_token_ids.size()
            # B x m x d
            #  get  embeds
            mention_embeds, mention_embeds_masks, \
            candidates_embeds = self.encode(mention_token_ids, mention_masks,
                                            candidate_token_ids,
                                            candidate_masks)
            # print(candidates_embeds.shape) # (1, 65, 128, 768)
            if self.attention_type == 'extend_multi':
                scores = self.extend_multi(mention_embeds, candidates_embeds)
            else:
                if self.attention_type == 'soft_attention':
                    scores = self.attention(candidates_embeds, mention_embeds,
                                            mention_embeds, mention_embeds_masks)
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
    def __init__(self, num_heads, num_layers):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embed_dim = 768
        self.transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True).to(self.device)
        self.transformerencoder = torch.nn.TransformerEncoder(self.transformerencoderlayer, self.num_layers).to(self.device)
        self.linearhead = torch.nn.Linear(self.embed_dim, 1).to(self.device)
    def forward(self, xs, ys):
        xs = xs.to(self.device) # (batch size, 1, embed_dim)
        ys = ys.squeeze(dim = -2).to(self.device) #(batch_size, cands, 1, embed_dim)
        input = torch.cat([xs, ys], dim = 1)
        attention_result = self.transformerencoder(input)
        scores = self.linearhead(attention_result[:,1:,:])
        scores = scores.squeeze(-1)
        return scores
