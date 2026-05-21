import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from .modules import DynamicPathoResampler, AttentionPoolingBlock, Attn_Net_Gated, SlideEncoder
from .utils.dist_utils import concat_all_gather, is_dist_avail_and_initialized, concat_all_gather_with_grad
from .loss import FlairLoss, SoftTopKLoss
from ...temp.src.models.pos_embed import get_2d_sincos_pos_embed
from ...temp.src.models.torchscale.model.LongNet import make_longnet_from_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class pathflip(nn.Module):
    def __init__(
        self,
        image_embed_dim=512,
        text_embed_dim=768,
        embed_dim=512,
        patch_size=256,
        num_query_latents=256,
        num_heads=8,
        text_encoder=None,
        freeze_text_encoder=True,
        text_pooling_type='mean',
        contrast_loss='infonce',
        temperature=0.1,
        use_fine_gained_loss=True,
        num_text_samples=8,
        num_flair_heads=8,
        init_logit_scale=10.0, 
        init_logit_bias=-10.0,
        text_encoder_use_lora=False,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
        use_soft_topk=False,
        top_k=128,
        topk_temperature=0.5,
    ):
        super(pathflip, self).__init__()

        self.patch_size = patch_size

        self.image_proj = nn.Sequential(
            nn.Linear(image_embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )
        self.slide_encoder = SlideEncoder(dim=embed_dim, num_heads=num_heads, num_layers=1, num_tokens=128)
        self.image_projection_global = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )
        self.image_projection_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )

        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        if text_encoder_use_lora:
            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)
        else:
            for param in self.text_encoder.parameters():
                param.requires_grad = not freeze_text_encoder

        self.text_pooling_type = text_pooling_type
        self.text_projection_global = nn.Sequential(
            nn.Linear(text_embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )
        self.text_projection_local = nn.Sequential(
            nn.Linear(text_embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )

        self.contrast_loss = contrast_loss
        if self.contrast_loss == 'infonce':
            self.temperature = temperature
        if self.contrast_loss == 'siglip':
            self.global_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            if init_logit_bias is not None:
                self.global_logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
            else:
                self.global_logit_bias = None

        self.use_fine_gained_loss = use_fine_gained_loss
        self.use_soft_topk = use_soft_topk
        if self.use_fine_gained_loss:
            if self.use_soft_topk:
                self.fine_gained_loss = SoftTopKLoss(
                    num_cap_per_img=num_text_samples,
                    top_k=top_k,
                    temperature=topk_temperature,
                )
            else:
                self.fine_gained_loss = FlairLoss(num_cap_per_img=num_text_samples)
                self.fine_gained_loss_head = AttentionPoolingBlock(embed_dim, num_flair_heads)
            self.local_logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
            if init_logit_bias is not None:
                self.local_logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
            else:
                self.local_logit_bias = None
    
    def forward(self, batch, return_attn=False):

        image_feats = batch['image']
        image_coords = batch['image_coords']
        device = image_feats.device
        input_ids_global = batch['input_ids_global']
        attn_mask_global = batch['attn_mask_global']
        input_ids_local = batch['input_ids_local']
        attn_mask_local = batch['attn_mask_local']
        B, L, D = image_feats.shape

        if is_dist_avail_and_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        ### image 
        image_feats = self.image_proj(image_feats.to(device)) # [B, L, D]
        outputs = self.slide_encoder(image_feats)
        image_feats_local = outputs['patch_features'] # [B, L, D]
        image_feats_global = outputs['tokens'][:, 0, :]

        image_feats_global = self.image_projection_global(image_feats_global) # [B, D]
        image_feats_local = self.image_projection_local(image_feats_local) # [B, N, D]      

        # text 
        ### global text
        text_output_global = self.text_encoder(input_ids_global, attn_mask_global)
        if self.text_pooling_type == 'mean':
            text_feats_global = text_output_global.last_hidden_state.mean(dim=1)
        elif self.text_pooling_type == 'cls_token':
            text_feats_global = text_output_global.last_hidden_state[:, 0]
        else:
            raise ValueError(f'Invalid pooling type: {self.text_pooling_type}')

        text_feats_global = self.text_projection_global(text_feats_global) # [B, D]

        ### Image-Text Global Contrastive Loss
        loss_itc = self.contrast_global(
            image_feats=image_feats_global, 
            text_feats=text_feats_global, 
            rank=rank, 
            world_size=world_size,
            bs=B, 
            device=device,
            need_norm=True,
            loss_type=self.contrast_loss,
        )
        
        ### Image-Text Fine-grained Loss
        loss_fine_gained = 0.
        query_attn = None
        if self.use_fine_gained_loss:
            ### local text
            input_ids_local = input_ids_local.reshape(-1, input_ids_local.shape[-1]) # [B*K, L]
            attn_mask_local = attn_mask_local.reshape(-1, attn_mask_local.shape[-1]) # [B*K, L]
            if self.text_pooling_type == 'mean':
                text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state.mean(dim=1) # [B*K, D]
            elif self.text_pooling_type == 'cls_token':
                text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state[:, 0]
            else:
                raise ValueError(f'Invalid pooling type: {self.text_pooling_type}')
            
            text_feats_local = self.text_projection_local(text_feats_local) # [B*K, D]

            if self.use_soft_topk:
                loss_fine_gained_out = self.fine_gained_loss(
                    image_features=image_feats_global,
                    text_features=text_feats_local,
                    logit_scale=self.local_logit_scale,
                    logit_bias=self.local_logit_bias,
                    image_tokens=image_feats_local,
                    rank=rank,
                    world_size=world_size,
                    output_weights=return_attn,
                )
            else:
                loss_fine_gained_out = self.fine_gained_loss(
                    image_features=image_feats_global,
                    text_features=text_feats_local,
                    logit_scale=self.local_logit_scale,
                    logit_bias=self.local_logit_bias,
                    image_tokens=image_feats_local,
                    visual_proj=self.fine_gained_loss_head,
                    rank=rank,
                    world_size=world_size,
                    output_attn_weights=return_attn,
                )

            if return_attn:
                loss_fine_gained, query_attn = loss_fine_gained_out
            else:
                loss_fine_gained = loss_fine_gained_out
 
        loss = loss_itc + loss_fine_gained

        if return_attn:
            if self.contrast_loss == 'siglip':
                if self.use_fine_gained_loss:
                    return loss, loss_itc, loss_fine_gained, query_attn, self.global_logit_scale, self.global_logit_bias, self.local_logit_scale, self.local_logit_bias
                else:
                    return loss, loss_itc, loss_fine_gained, query_attn, self.global_logit_scale, self.global_logit_bias
            elif self.contrast_loss == 'infonce':
                if self.use_fine_gained_loss:
                    return loss, loss_itc, loss_fine_gained, query_attn, self.local_logit_scale, self.local_logit_bias
                else:
                    return loss, loss_itc, loss_fine_gained, query_attn
            else:
                raise ValueError(f'Invalid loss type: {self.contrast_loss}')
        else:
            if self.contrast_loss == 'siglip':
                if self.use_fine_gained_loss:
                    return loss, loss_itc, loss_fine_gained, self.global_logit_scale, self.global_logit_bias, self.local_logit_scale, self.local_logit_bias
                else:
                    return loss, loss_itc, loss_fine_gained, self.global_logit_scale, self.global_logit_bias
            elif self.contrast_loss == 'infonce':
                if self.use_fine_gained_loss:
                    return loss, loss_itc, loss_fine_gained, self.local_logit_scale, self.local_logit_bias
                else:
                    return loss, loss_itc, loss_fine_gained
            else:
                raise ValueError(f'Invalid loss type: {self.contrast_loss}')

    def contrast_global(self, image_feats, text_feats, rank, world_size, bs, device, need_norm=True, loss_type='infonce'):
        if need_norm:
            image_feats = F.normalize(image_feats, dim=-1)
            text_feats = F.normalize(text_feats, dim=-1)

        if world_size>1:
            image_feats_all = concat_all_gather_with_grad(image_feats)  # [batch_size*num_gpu, embed_dim]
            text_feats_all = concat_all_gather_with_grad(text_feats)  # [batch_size*num_gpu, embed_dim]
        else:
            image_feats_all = image_feats
            text_feats_all = text_feats

        if loss_type == 'infonce':
            sim_i2t = torch.einsum("bd,nd->bn", image_feats, text_feats_all) # [batch_size, batch_size*num_gpu]
            sim_i2t = sim_i2t / self.temperature # [batch_size, batch_size*num_gpu]

            # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
            sim_t2i = torch.einsum("bd,nd->bn", text_feats, image_feats_all) # [batch_size, batch_size*num_gpu]
            sim_t2i = sim_t2i / self.temperature  # [batch_size, batch_size*num_gpu]

            # targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=torch.long).to(device)
            targets = torch.arange(rank * bs, rank * bs + bs, dtype=torch.long, device=device)

            loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2

            # return loss_itc, sim_i2t, sim_t2i

        elif loss_type == 'siglip':

            chunk_size = bs 
            total_bs = world_size * bs
            loss_acc = torch.tensor(0.0, device=device)
            pos_indices = torch.arange(rank * bs, rank * bs + bs, device=device)
            # ------------------------------------------
            # Pass 1: Image -> Text (Local Images vs Global Text Chunks)
            # ------------------------------------------
            # image_feats: [bs, D] (Local)
            # text_feats_all: [total_bs, D] (Global)
            for chunk_start in range(0, total_bs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_bs)
                text_chunk = text_feats_all[chunk_start:chunk_end] # [chunk_len, D]
                logits = image_feats @ text_chunk.t()
                logits = logits * self.global_logit_scale + self.global_logit_bias
                labels = torch.full((bs, chunk_end - chunk_start), -1.0, device=device)
                if mask.any():
                    labels[mask, chunk_pos_cols] = 1.0
                
                loss_acc += -F.logsigmoid(labels * logits).sum()
            # ------------------------------------------
            # Pass 2: Text -> Image (Local Text vs Global Image Chunks)
            # ------------------------------------------
            # text_feats: [bs, D] (Local)
            # image_feats_all: [total_bs, D] (Global)
            for chunk_start in range(0, total_bs, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_bs)
                image_chunk = image_feats_all[chunk_start:chunk_end]
                # logits: [bs, chunk_len]
                logits = text_feats @ image_chunk.t()
                logits = logits * self.global_logit_scale + self.global_logit_bias
                labels = torch.full((bs, chunk_end - chunk_start), -1.0, device=device)
                mask = (pos_indices >= chunk_start) & (pos_indices < chunk_end)
                if mask.any():
                    chunk_pos_cols = pos_indices[mask] - chunk_start
                    labels[mask, chunk_pos_cols] = 1.0
                loss_acc += -F.logsigmoid(labels * logits).sum()

        return loss_itc
    
    def forward_global(self, batch):
        # image
        image_feats_global = self.forward_image_global(batch)
        # text
        text_feats_global = self.forward_text_global(batch)
        return image_feats_global, text_feats_global

    def forward_image_local(self, batch):
        image_feats = batch['image']
        image_coords = batch['image_coords']
        B, L, D = image_feats.shape

        ### image 
        image_feats = self.image_proj(image_feats.to(device)) # [B, L, D]
        outputs = self.slide_encoder(image_feats)
        image_feats_local = outputs['patch_features'] # [B, L, D]

        return image_feats_local

    def forward_image_global(self, batch):
        image_feats = batch['image']
        image_coords = batch['image_coords']
        B, L, D = image_feats.shape
        # image
        image_feats = self.image_proj(image_feats)
        image_feats_global = self.slide_encoder(image_feats)['tokens'][:, 0, :]  # [B, D]
        image_feats_global = self.image_projection_global(image_feats_global)        

        return image_feats_global

    def forward_text_global(self, batch):

        device = next(self.parameters()).device
        input_ids_global = batch['input_ids_global'].to(device)
        attn_mask_global = batch['attn_mask_global'].to(device)

        # text
        text_output_global = self.text_encoder(input_ids_global, attn_mask_global)
        if self.text_pooling_type == 'mean':
            text_feats_global = text_output_global.last_hidden_state.mean(dim=1)
        elif self.text_pooling_type == 'cls_token':
            text_feats_global = text_output_global.last_hidden_state[:, 0]
        else:
            raise ValueError(f'Invalid pooling type: {self.text_pooling_type}')
        text_feats_global = self.text_projection_global(text_feats_global) # (B, D)
        
        return text_feats_global

    def forward_attn(self, batch):
        
        image_feats = batch['image']
        image_coords = batch['image_coords']
        device = image_feats.device
        # input_ids_global = batch['input_ids_global']
        # attn_mask_global = batch['attn_mask_global']
        input_ids_local = batch['input_ids_local']
        attn_mask_local = batch['attn_mask_local']
        B, L, D = image_feats.shape
        
        ### image 
        image_feats = self.image_proj(image_feats)
        outputs = self.slide_encoder(image_feats, return_attn=True)

        image_feats_local = outputs['patch_features'] # [B, L, D]
        image_feats_global = outputs['tokens'][:, 0, :]
        attn_dict = outputs['attentions']
        global_attn = attn_dict['layer_0_cross_attn'][:, 0, :]

        ### Image-Text Fine-grained Loss
        ### local text
        input_ids_local = input_ids_local.reshape(-1, input_ids_local.shape[-1]) # [B*K, L]
        attn_mask_local = attn_mask_local.reshape(-1, attn_mask_local.shape[-1]) # [B*K, L]
        if self.text_pooling_type == 'mean':
            text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state.mean(dim=1) # [B*K, D]
        elif self.text_pooling_type == 'cls_token':
            text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state[:, 0]
        else:
            raise ValueError(f'Invalid pooling type: {self.text_pooling_type}')
        
        text_feats_local = self.text_projection_local(text_feats_local).reshape(B, -1, D) # [B, K, D]
        _, text_conditioned_attn = self.fine_gained_loss_head(text_feats_local, image_feats_local, image_feats_local, output_attn_weights=True)  # [B, K, N]

        print(f'global_attn: {global_attn.shape}')
        print(f'text_conditioned_attn: {text_conditioned_attn.shape}')

        return global_attn, text_conditioned_attn, attn_dict
    
    def forward_query(self, batch):
        
        image_feats = batch['image']
        image_coords = batch['image_coords']
        device = image_feats.device

        input_ids_local = batch['input_ids_local']
        attn_mask_local = batch['attn_mask_local']
        B, L, D = image_feats.shape
        
        ### image 
        image_feats = self.image_proj(image_feats)
        outputs = self.slide_encoder(image_feats, return_attn=True)
        image_feats_local = outputs['patch_features'] # [B, L, D]

        ### local text
        input_ids_local = input_ids_local.reshape(-1, input_ids_local.shape[-1]) # [B*K, L]
        attn_mask_local = attn_mask_local.reshape(-1, attn_mask_local.shape[-1]) # [B*K, L]
        if self.text_pooling_type == 'mean':
            text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state.mean(dim=1) # [B*K, D]
        elif self.text_pooling_type == 'cls_token':
            text_feats_local = self.text_encoder(input_ids_local, attn_mask_local).last_hidden_state[:, 0]
        else:
            raise ValueError(f'Invalid pooling type: {self.text_pooling_type}')
        
        text_feats_local = self.text_projection_local(text_feats_local).reshape(B, -1, D) # [B, K, D]
        query_output_feats = self.fine_gained_loss_head(text_feats_local, image_feats_local, image_feats_local)  # [B, K, D]

        return text_feats_local, query_output_feats



def test_forward():
    outputs = model(batch, return_attn=True)
    loss, loss_itc, loss_fine_gained = outputs[0], outputs[1], outputs[2]
    query_attn = outputs[3]
    print(f'loss: {loss:.4f}, loss_itc: {loss_itc:.4f}, loss_fine_gained: {loss_fine_gained:.4f}, query_attn: {query_attn.shape}')

def test_forward_with_attn():
    global_attn, text_conditioned_attn, attn_dict = model.forward_attn(batch)
    print(f'global_attn: {global_attn.shape}, text_conditioned_attn: {text_conditioned_attn.shape}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16

    text_encoder = 'emilyalsentzer/Bio_ClinicalBERT'
    image = torch.randn(4, 512, 512).to(device, dtype=dtype)
    image_coords = torch.randint(0, 100000, (4, 512, 2)).to(torch.long).to(device)

    input_ids_global = torch.randint(0, 1000, (4, 128)).to(torch.long).to(device)
    attn_mask_global = torch.ones((4, 128), dtype=torch.long).to(device)

    input_ids_local = torch.randint(0, 1000, (4, 6, 128)).to(torch.long).to(device)
    attn_mask_local = torch.ones((4, 6, 128), dtype=torch.long).to(device)

    batch = {
        'image': image,
        'image_coords': image_coords,
        'input_ids_global': input_ids_global,
        'attn_mask_global': attn_mask_global,
        'input_ids_local': input_ids_local,
        'attn_mask_local': attn_mask_local,
    }
    model = pathflip(text_encoder=text_encoder, num_text_samples=2).to(device, dtype=dtype)

    # test_forward()
    test_forward_with_attn()
