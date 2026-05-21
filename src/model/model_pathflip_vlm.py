import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import get_peft_model, LoraConfig, TaskType


from .modules import Attn_Net_Gated, AttentionPoolingBlock, DynamicPathoResampler, SlideEncoder
from .utils.model_utils import prepare_inputs_labels_for_multimodal
from ...temp.src.models.pos_embed import get_2d_sincos_pos_embed
from ...temp.src.models.torchscale.model.LongNet import make_longnet_from_name


class pathflip_vlm(nn.Module):
    def __init__(
        self,
        image_embed_dim=512,
        text_embed_dim=768,
        embed_dim=512,
        patch_size=256,
        num_heads=8,
        text_encoder=None,
        num_fine_gained_heads=8,
        text_encoder_use_lora=False,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"],
        keep_k=128,
        tokenizer=None,
        llm_name_or_path=None,
        llm_lora=False,
        llm_lora_alpha=16,
        llm_lora_dropout=0.0,
        llm_lora_r=8,
        freeze_llm=False,
        freeze_image_encoder=False,
        freeze_text_encoder=True,
        use_soft_topk=False,
        topk_temperature=0.5,
    ):
        super(pathflip_vlm, self).__init__()

        self.patch_size = patch_size
        self.keep_k = keep_k
        self.IMAGE_TOKEN_INDEX = tokenizer.convert_tokens_to_ids('<image>')
        self.use_soft_topk = use_soft_topk
        self.topk_temperature = topk_temperature

        self.image_proj = nn.Sequential(
            nn.Linear(image_embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )
        self.slide_encoder = SlideEncoder(dim=embed_dim, num_heads=num_heads, num_layers=1, num_tokens=128)
        self.image_projection_local = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )
        if freeze_image_encoder:
            for param in self.image_proj.parameters():
                param.requires_grad = False
            for param in self.slide_encoder.parameters():
                param.requires_grad = False
            for param in self.image_projection_local.parameters():
                param.requires_grad = False

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

        self.text_projection_local = nn.Sequential(
            nn.Linear(text_embed_dim, embed_dim),
            nn.RMSNorm(embed_dim),
            nn.Dropout(0.1),
        )

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.text_projection_local.parameters():
                param.requires_grad = False

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name_or_path)
        self.llm.resize_token_embeddings(len(tokenizer))
        llm_hidden_dim = self.llm.config.hidden_size
        
        if llm_lora:
            self.llm = get_peft_model(
                self.llm,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=llm_lora_r,
                    lora_alpha=llm_lora_alpha,
                    lora_dropout=llm_lora_dropout,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
                        "gate_proj", "up_proj", "down_proj"          # MLP
                    ],
                    bias="none",
                )
            )
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        self.image_proj_llm_local = nn.Sequential(
            nn.Linear(embed_dim, llm_hidden_dim),
            nn.RMSNorm(llm_hidden_dim),
            nn.Dropout(0.1),
        )

        self.image_proj_llm_latten = nn.Sequential(
            nn.Linear(embed_dim, llm_hidden_dim),
            nn.RMSNorm(llm_hidden_dim),
            nn.Dropout(0.1),
        )

        if not self.use_soft_topk:
            self.fine_gained_head = AttentionPoolingBlock(embed_dim, num_heads=num_fine_gained_heads)
    
    def forward(self, batch, return_attn=False):

        image_feats = batch['image']
        image_coords = batch['image_coords']
        device = image_feats.device
        input_ids = batch['input_ids']
        attn_mask = batch['attn_mask']
        labels = batch['labels']
        input_ids_query = batch['input_ids_query']
        attn_mask_query = batch['attn_mask_query']

        B, L, D = image_feats.shape

        ### image
        image_feats = self.image_proj(image_feats.to(device)) # [B, L, D]
        outputs = self.slide_encoder(image_feats)
        image_local_feats = outputs['patch_features'] # [B, L, D]
        image_feats_latten = outputs['tokens'][:, 1:, :] # [B, N, D]
        image_feats_local = self.image_projection_local(image_local_feats) # [B, L, D] 

        query_text_feats = self.text_encoder(input_ids_query, attn_mask_query).last_hidden_state[:, 0] # [B, D]
        query_text_feats = self.text_projection_local(query_text_feats) # [B, D]

        if self.use_soft_topk:
            text_conditioned_score = self.compute_similarity_scores(query_text_feats, image_feats_local) # [B, L]
        else:
            _, text_conditioned_score = self.fine_gained_head(query_text_feats.unsqueeze(1), image_feats_local, image_feats_local, output_attn_weights=True) # [B, 1, L]
            text_conditioned_score = text_conditioned_score.squeeze(1) # [B, L]

        image_feats_local = self.differentiable_topk(x=image_feats_local, score=text_conditioned_score, keep_k=self.keep_k) # [B, K, D]
        image_feats_local = self.image_proj_llm_local(image_feats_local) # [B, K, llm_dim]   

        image_feats_latten = self.image_proj_llm_latten(image_feats_latten) # [B, N, llm_dim]

        image_feats_llm = torch.cat([image_feats_local, image_feats_latten], dim=1)

        llm_input = prepare_inputs_labels_for_multimodal(
            llm = self.llm,
            input_ids = input_ids,
            attention_mask = attn_mask,
            labels = labels,
            IMAGE_TOKEN_INDEX = self.IMAGE_TOKEN_INDEX,
            pixel_values = image_feats_llm,
        )
        llm_output = self.llm(**llm_input)

        return llm_output.loss
    
    @torch.no_grad()
    def generate(
        self, 
        image_feats, 
        image_coords, 
        input_ids, 
        attn_mask, 
        input_ids_query,
        attn_mask_query,
        tokenizer, 
        max_new_tokens=512, 
        temperature=0.7, 
        top_p=0.95, 
        top_k=None, 
        do_sample=True, 
        repetition_penalty=1.0, 
        eos_token_id=None
    ):
        """Generate text from image patch features and prompt tokens."""
        self.eval()
        device = image_feats.device
        B = image_feats.shape[0]
        
        ### image
        image_feats = self.image_proj(image_feats.to(device)) # [B, L, D]
        outputs = self.slide_encoder(image_feats)
        image_local_feats = outputs['patch_features'] # [B, L, D]
        image_feats_latten = outputs['tokens'][:, 1:, :] # [B, N, D]
        image_feats_local = self.image_projection_local(image_local_feats) # [B, L, D] 

        query_text_feats = self.text_encoder(input_ids_query, attn_mask_query).last_hidden_state[:, 0] # [B, D]
        query_text_feats = self.text_projection_local(query_text_feats) # [B, D]

        if self.use_soft_topk:
            text_conditioned_score = self.compute_similarity_scores(query_text_feats, image_feats_local) # [B, L]
        else:
            _, text_conditioned_score = self.fine_gained_head(query_text_feats.unsqueeze(1), image_feats_local, image_feats_local, output_attn_weights=True) # [B, 1, L]
            text_conditioned_score = text_conditioned_score.squeeze(1) # [B, L]

        image_feats_local = self.differentiable_topk(x=image_feats_local, score=text_conditioned_score, keep_k=self.keep_k) # [B, K, D]
        image_feats_local = self.image_proj_llm_local(image_feats_local) # [B, K, llm_dim]   

        image_feats_latten = self.image_proj_llm_latten(image_feats_latten) # [B, N, llm_dim]

        image_feats_llm = torch.cat([image_feats_local, image_feats_latten], dim=1)
        
        llm_input = prepare_inputs_labels_for_multimodal(
            llm = self.llm,
            input_ids = input_ids,
            attention_mask = attn_mask,
            labels = None,
            IMAGE_TOKEN_INDEX = self.IMAGE_TOKEN_INDEX,
            pixel_values = image_feats_llm,
        )
        
        generate_kwargs = {
            "attention_mask": llm_input["attention_mask"],
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty,
            "eos_token_id": eos_token_id if eos_token_id is not None else tokenizer.eos_token_id,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        if llm_input.get('inputs_embeds') is not None:
            generate_kwargs['inputs_embeds'] = llm_input['inputs_embeds']
        else:
            generate_kwargs['input_ids'] = llm_input['input_ids']
        
        return self.llm.generate(**generate_kwargs)
        
    def compute_similarity_scores(self, query_text_feats, image_feats_local):
        """
        patch
        SoftTopKattention head
        
        Args:
            query_text_feats:  [B, D]
            image_feats_local: patch [B, L, D]
        Returns:
            similarity_scores: [B, L]
        """
        text_normalized = F.normalize(query_text_feats, dim=-1).unsqueeze(1) # [B, 1, D]
        image_normalized = F.normalize(image_feats_local, dim=-1) # [B, L, D]
        
        similarity = torch.einsum('bid,bld->bil', text_normalized, image_normalized) # [B, 1, L]
        similarity_scores = similarity.squeeze(1) # [B, L]
        
        return similarity_scores
    
    def differentiable_topk(self, x: torch.Tensor, score: torch.Tensor, keep_k: int):
        """
        Differentiable global top-k for patch features.
        
        Args:
            x: patch [B, L, D] 
                B=batch size, L=number of patches, D=feature dimension
            score: patch importance scores [B, L]
            keep_k: atch K
        Return:
            x_pruned: selected features [B, K, D]
        """
        B, L, D = x.shape
        K = keep_k


        topk_idx = score_gumbel.topk(K, dim=-1).indices

        x_pruned = torch.gather(x, dim=1, index=index)  # [B, K, D]

        return x_pruned

    def coords_to_pos(self, coords, patch_size: int = 256, cls_token: bool = False):
        """
        This function is used to convert the coordinates to the positional indices

        Arguments:
        ----------
        coords: torch.Tensor
            The coordinates of the patches, of shape [B, L, 2]
        output: torch.Tensor
            The positional indices of the patches, of shape [B, L]
        """
        coords_ = torch.floor(coords / patch_size)
        pos = coords_[..., 0] * self.slide_ngrids + coords_[..., 1]
        if cls_token:
            pos = pos + 1  # add 1 for the cls token
        return pos.long()   


if __name__ == "__main__":
    pass
