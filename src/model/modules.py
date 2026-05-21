import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Callable
from torch.nn import LayerNorm

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

import torch
import torch.nn as nn

class MAB(nn.Module):
    """"""
    def __init__(self, dim, num_heads, dropout=0.2, q_is_kv=True):
        super().__init__()
        self.q_is_kv = q_is_kv
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q = nn.RMSNorm(dim)
        if not q_is_kv:
            # self.norm_kv = self.norm_q
            self.norm_kv = nn.RMSNorm(dim)
            
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 4, dim), nn.Dropout(dropout)
        )
        self.norm2 = nn.RMSNorm(dim)

    def forward(self, x, y, return_attn=False):
        q = self.norm_q(x)
        if self.q_is_kv:
            k = self.norm_q(y)
            v = self.norm_q(y)
        else:
            k = self.norm_kv(y)
            v = self.norm_kv(y)
        attn_out, attn_weights = self.mha(query=q, key=k, value=v)
        out = x + attn_out
        out_norm = self.norm2(out)
        if return_attn:
            return out + self.ffn(out_norm), attn_weights
        else:
            return out + self.ffn(out_norm)

class SlideEncoder(nn.Module):
    """WSI encoder for contrastive learning, LLM injection, and patch analysis."""
    def __init__(self, dim=512, num_heads=8, num_layers=2, num_tokens=128):
        super().__init__()
        
        self.visual_tokens = nn.Parameter(torch.randn(1, 1 + num_tokens, dim))
        
        self.encoder_layers = nn.ModuleList([
            nn.ModuleDict({
                'cross': MAB(dim, num_heads, q_is_kv=False),
                'self': MAB(dim, num_heads, q_is_kv=True)
            })
            for _ in range(num_layers)
        ])
        self.patch_decoder = MAB(dim, num_heads, q_is_kv=False)
        

    def forward(self, image_features, return_attn=False):
        """Encode WSI patch features."""
        batch_size = image_features.size(0)
        tokens = self.visual_tokens.expand(batch_size, -1, -1)

        image_features_res = image_features
        
        attn_dict = {}
        for layer_idx, layer in enumerate(self.encoder_layers):
            if return_attn:
                tokens, cross_attn = layer['cross'](x=tokens, y=image_features, return_attn=True) # (B, 32, Dim)
                tokens, self_attn = layer['self'](x=tokens, y=tokens, return_attn=True)        # (B, 32, Dim)
                attn_dict[f'layer_{layer_idx}_cross_attn'] = cross_attn  # [B, num_heads, token_num, N]
                attn_dict[f'layer_{layer_idx}_self_attn'] = self_attn    # [B, num_heads, token_num, token_num]
            else:
                tokens = layer['cross'](x=tokens, y=image_features)
                tokens = layer['self'](x=tokens, y=tokens)
            
        if return_attn:
            patch_features, decoder_attn = self.patch_decoder(x=image_features, y=tokens, return_attn=True)
            attn_dict['decoder_cross_attn'] = decoder_attn  # [B, num_heads, N, token_num]
        else:
            patch_features = self.patch_decoder(x=image_features, y=tokens)

        patch_features = patch_features + image_features_res

        output = {
            "tokens": tokens,                     
            "patch_features": patch_features
        }
        if return_attn:
            output['attentions'] = attn_dict
        return output


class DynamicPathoResampler(nn.Module):
    def __init__(self, dim, num_latents=512, depth=6, num_heads=8):
        """Resampler for unordered WSI patch bags."""
        super().__init__()
        self.num_latents = num_latents
        self.dim = dim
    
        self.global_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        
        self.layers = nn.ModuleList([
            ResamplerLayer(dim, num_heads) for _ in range(depth)
        ])
        
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x):
        """Resample an unordered patch bag with shape [B, L, D]."""

        x = x.contiguous()

        B, L, D = x.shape
        
        
        # [B, L, D] -> [B, D]
        global_context = x.mean(dim=1) 
        
        # [B, D] -> [B, D]
        global_context = self.global_proj(global_context)
        
        # queries = self.latents.expand(B, -1, -1) + global_context.unsqueeze(1)
        queries = self.latents.repeat(B, 1, 1) + global_context.unsqueeze(1)
        queries = queries.contiguous()
        
        all_attn_maps = []
        for layer in self.layers:
            # queries: [B, N, D]
            queries, attn_map = layer(queries, x)
            all_attn_maps.append(attn_map)
            
        return self.norm_out(queries), all_attn_maps


class ResamplerLayer(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # C. Feed Forward
        self.norm_ffn = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, q, kv):
        # q: [B, N, D] (Latents)
        # kv: [B, L, D] (Image Features - Bag of Patches)

        q = q.contiguous()
        kv = kv.contiguous()
        
        q_norm = self.norm_self(q)
        kv_norm = q_norm.clone()
        q = q + self.self_attn(q_norm, kv_norm, kv_norm)[0]
        
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        # query=Latents, key=Patches, value=Patches
        res, weights = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        q = q + res.to(q.dtype)
        
        # 3. Feed Forward
        q = q + self.mlp(self.norm_ffn(q))
        
        return q, weights

        
class LocalGroundingHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm_text = nn.LayerNorm(dim)
        self.norm_vis = nn.LayerNorm(dim)

    def forward(self, text_embeds, visual_tokens):
        # text_embeds: [B, L, D]
        # visual_tokens: [B, M, D]
        q = self.norm_text(text_embeds)
        k = v = self.norm_vis(visual_tokens)
        
        out, text_to_vis_weights = self.cross_attn(q, k, v)
        return out, text_to_vis_weights


def get_grounding_map(text_to_vis_weights, resampler_attn_maps):
    """
    text_to_vis_weights: [B, L, M]
    resampler_attn_maps: list of [B, M, N]
    """
    A = resampler_attn_maps[-1]
    B = text_to_vis_weights
    
    grounding_map = torch.bmm(B, A)
    return grounding_map


class Attn_Net_Gated(nn.Module):
    """
    Attention Network with Sigmoid Gating (3 fc layers)
    args:
        L: input feature dimension
        D: hidden layer dimension
        dropout: whether to use dropout (p = 0.25)
        n_classes: number of classes 
    """
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):

        B, L, _ = x.size() # [B, N, D]
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # [B, N, 1]

        attn = A.view(B, -1, L) # [B, 1, N]
        attn = F.softmax(attn, dim=-1) # [B, 1, N]

        x = torch.bmm(attn, x).squeeze(1) # [B, D]

        return x, attn


class AttentionPoolingBlock(nn.Module):
    def __init__(
            self,
            context_dim: int,
            num_heads: int = 8,
            norm_layer: Callable = LayerNorm,
            need_weights: bool = False
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(context_dim, num_heads, batch_first=True)
        self.ln_q = norm_layer(context_dim)
        self.ln_k = norm_layer(context_dim)
        self.ln_v = norm_layer(context_dim)
        self.need_weights=need_weights
        self.num_heads = num_heads

    def forward(self, q, k, v, attn_mask=None, output_attn_weights=False, average_attn_weights=True):
        batch_size, seg_length, embed_dim = k.size()
        _, query_length, _ = q.size()

        if attn_mask is not None:
            if attn_mask.size() != (batch_size, seg_length):
                expected_shape = (batch_size, seg_length)
                actual_shape = attn_mask.size()
                raise ValueError(f"Expected attn_mask shape to be {expected_shape}, but got {actual_shape}")
            
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.n_head, query_length, seg_length)
            attn_mask = attn_mask.reshape(batch_size * self.n_head, query_length, seg_length)
            attn_mask = (1.0 - attn_mask) * torch.finfo(attn_mask.dtype).min

        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        if self.need_weights or output_attn_weights:
            out, attn_weights = self.attn(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=average_attn_weights)
            return out, attn_weights
        else:
            out = self.attn(q, k, v, attn_mask=attn_mask, need_weights=False)[0]
            return out


if __name__ == '__main__':
    pass 
