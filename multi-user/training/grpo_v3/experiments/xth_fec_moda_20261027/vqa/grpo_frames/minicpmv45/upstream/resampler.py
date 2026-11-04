from functools import partial
from itertools import chain
from typing import Optional, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from transformers.integrations import is_deepspeed_zero3_enabled

def get_2d_sincos_pos_embed(embed_dim, image_size):
    """
    image_size: image_size or (image_height, image_width)
    return:
    pos_embed: [image_height, image_width, embed_dim]
    """
    if isinstance(image_size, int):
        grid_h_size, grid_w_size = image_size, image_size
    else:
        grid_h_size, grid_w_size = image_size[0], image_size[1]

    grid_h = np.arange(grid_h_size, dtype=np.float32)
    grid_w = np.arange(grid_w_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[0])  # (H, W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim // 2, grid[1])  # (H, W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=-1)  # (H, W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid_new(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (H, W)
    out: (H, W, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    out = np.einsum('hw,d->hwd', pos, omega)  # (H, W, D/2), outer product

    emb_sin = np.sin(out)  # (H, W, D/2)
    emb_cos = np.cos(out)  # (H, W, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=-1)  # (H, W, D)
    return emb

def get_1d_sincos_pos_embed_from_temporal_size(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
       given learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (batch_size, num_queries, embed_dim)
    """

    def __init__(
            self,
            num_queries,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            adaptive=False,
            max_size=(70, 70),
            max_temporal_size=72000,
            batch_infer=False 
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.adaptive = adaptive
        self.max_size = max_size
        self.max_temporal_size = max_temporal_size
        self.batch_infer = batch_infer

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)

        self.ln_post = norm_layer(embed_dim)
        self.proj = nn.Parameter((embed_dim ** -0.5) * torch.randn(embed_dim, embed_dim))

        self._set_2d_pos_cache(self.max_size)
        self._set_temporal_pos_cache(self.max_temporal_size)
        self.apply(self._init_weights)

    def _set_2d_pos_cache(self, max_size, device='cpu'):
        if is_deepspeed_zero3_enabled():
            device='cuda'
        pos_embed = torch.from_numpy(get_2d_sincos_pos_embed(self.embed_dim, max_size)).float().to(device)
        self.register_buffer("pos_embed", pos_embed, persistent=False)

    def _adjust_pos_cache(self, tgt_sizes, device):
        max_h = torch.max(tgt_sizes[:, 0])
        max_w = torch.max(tgt_sizes[:, 1])
        if max_h > self.max_size[0] or max_w > self.max_size[1]:
            self.max_size = [max(max_h, self.max_size[0]), max(max_w, self.max_size[1])]
            self._set_2d_pos_cache(self.max_size, device)
    
    def _set_temporal_pos_cache(self, max_temporal_size, device='cpu'):
        temporal_size = np.arange(max_temporal_size, dtype=np.float32)
        pos_embed = torch.from_numpy(get_1d_sincos_pos_embed_from_temporal_size(self.embed_dim, temporal_size)).float().to(device)
        self.register_buffer("temporal_pos_embed", pos_embed, persistent=False)
    
    def _adjust_temporal_pos_cache(self, max_temporal_size, device):
        if max_temporal_size > self.max_temporal_size:
            self.max_temporal_size = max_temporal_size
            self._set_temporal_pos_cache(self.max_temporal_size, device)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, tgt_sizes=None, temporal_ids=None):
        assert x.shape[0] == tgt_sizes.shape[0]
        bs = x.shape[0]

        device = x.device
        dtype = x.dtype

        patch_len = tgt_sizes[:, 0] * tgt_sizes[:, 1]

        self._adjust_pos_cache(tgt_sizes, device=device)

        temporal_pos_emb = False
        temporal_ids_flatten = None
        if temporal_ids is not None:
            # example: [[-1], [-1], [2, 6, 9]]
            temporal_ids_flatten = list(chain.from_iterable(temporal_ids))
            max_temporal_size = max(temporal_ids_flatten) + 1
            if max_temporal_size > -1:
                temporal_pos_emb = True
            if max_temporal_size > self.max_temporal_size:
                self._adjust_temporal_pos_cache(max_temporal_size, device)


        max_patch_len = torch.max(patch_len)
        key_padding_mask = torch.zeros((bs, max_patch_len), dtype=torch.bool, device=device)

        pos_embed = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            pos_embed.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True

        pos_embed = torch.nn.utils.rnn.pad_sequence(
            pos_embed, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D

        x = self.kv_proj(x)  # B * L * D
        x = self.ln_kv(x).permute(1, 0, 2)  # L * B * D

        q = self.ln_q(self.query)  # Q * D

        pos_embed_2d = []
        pos_embed_temporal = []
        for i in range(bs):
            tgt_h, tgt_w = tgt_sizes[i]
            if temporal_pos_emb:
                if temporal_ids_flatten[i] == -1:
                    pos_embed_temporal.append(torch.zeros(self.embed_dim, dtype=dtype, device=device))
                else:
                    pos_embed_temporal.append(self.temporal_pos_embed[temporal_ids_flatten[i]].to(dtype)) # D

            pos_embed_2d.append(self.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(dtype))  # patches * D
            key_padding_mask[i, patch_len[i]:] = True

        pos_embed_2d = torch.nn.utils.rnn.pad_sequence(
            pos_embed_2d, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # BLD => L * B * D
        
        v = x
        k = x + pos_embed_2d
        
        if self.batch_infer:
            out = self.batch_attn_forward(q, k, v, pos_embed_temporal, temporal_ids, key_padding_mask)
        else: # save gpu memory
            out = self.foreach_attn_forward(q, k, v, pos_embed_temporal, temporal_ids, key_padding_mask)
        
        #  out: Q * B * D
        x = out.permute(1, 0, 2)  # B * Q * D

        x = self.ln_post(x)
        x = x @ self.proj
        return x


    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)
    

    def batch_attn_forward(self, q, k, v, pos_embed_temporal, temporal_ids, key_padding_mask):
        bs = k.shape[1]

        if pos_embed_temporal:
            # temporal 维度折叠
            # 时序 embedding
            k += torch.stack(pos_embed_temporal, dim=0)
            bs = len(temporal_ids)
            merge_k = []
            merge_v = []
            merge_key_padding_mask = []

            start = 0
            for tp in temporal_ids:
                end = start + len(tp)
                # # L * (end-start) * D -> (end-start) * L * D -> 1 * L*(end-start) * D
                merge_k.append(k[:, start: end, :].permute(1, 0, 2).reshape(-1, self.embed_dim))
                merge_v.append(v[:, start: end, :].permute(1, 0, 2).reshape(-1, self.embed_dim))
                merge_key_padding_mask.append(key_padding_mask[start: end, :].reshape(-1, 1))

                start = end
                            
            k = torch.nn.utils.rnn.pad_sequence(merge_k, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # L*(end-start)
            v = torch.nn.utils.rnn.pad_sequence(merge_v, batch_first=True, padding_value=0.0).permute(1, 0, 2)  # L*(end-start)
            key_padding_mask = torch.nn.utils.rnn.pad_sequence(merge_key_padding_mask, batch_first=True, padding_value=True).squeeze(-1)

        out = self.attn(
            self._repeat(q, bs),  # Q * B * D
            k,  # L * B * D +  L * B * D
            v,
            key_padding_mask=key_padding_mask)[0]

        return out


    def foreach_attn_forward(self, q, k, v, pos_embed_temporal, temporal_ids, key_padding_mask):
        bs = k.shape[1]

        if pos_embed_temporal:
            k += torch.stack(pos_embed_temporal, dim=0)
            # bs = len(temporal_ids)
            out_list = []

            start = 0
            for tp in temporal_ids:
                end = start + len(tp)
                # 处理每个序列而不padding
                curr_k = k[:, start:end, :].reshape(-1, self.embed_dim)
                curr_v = v[:, start:end, :].reshape(-1, self.embed_dim)
                curr_key_padding_mask = key_padding_mask[start: end, :].reshape(-1)
                curr_out = self.attn(
                    q,
                    curr_k,
                    curr_v,
                    key_padding_mask=curr_key_padding_mask,
                )[0]

                out_list.append(curr_out)
                start = end

            # 合并所有序列的结果
            out = torch.stack(out_list, dim=1)

        else:
            out = self.attn(
                self._repeat(q, bs),  # Q * B * D
                k,  # L * B * D +  L * B * D
                v,
                key_padding_mask=key_padding_mask)[0]
            
        return out
