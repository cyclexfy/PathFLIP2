from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ExponentialLR, SequentialLR
import torch.distributed as dist

from .model_pathflip import pathflip
from .utils.dist_utils import concat_all_gather, is_dist_avail_and_initialized, concat_all_gather_with_grad


class pathflip_pl(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = pathflip(
            image_embed_dim=args.image_embed_dim,
            text_embed_dim=args.text_embed_dim,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            num_query_latents=args.num_query_latents,
            text_encoder=args.text_encoder,
            freeze_text_encoder=args.freeze_text_encoder,
            text_pooling_type=args.text_pooling_type,
            num_text_samples=args.num_text_samples,
            num_flair_heads=args.num_flair_heads,
            contrast_loss=args.contrast_loss,
            temperature=args.temperature,
            use_fine_gained_loss=args.use_fine_gained_loss,
            init_logit_scale=args.init_logit_scale,
            init_logit_bias=args.init_logit_bias,
            text_encoder_use_lora=args.text_encoder_use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            use_soft_topk=args.use_soft_topk,
            top_k=args.top_k,
            topk_temperature=args.topk_temperature,
        )
        self.save_hyperparameters(args)

        self.eval_retrieval_on_epoch = args.eval_retrieval_on_epoch
        self.num_retrieval_samples_epoch = args.num_retrieval_samples_epoch
        self.val_image_feats_epoch = []
        self.val_text_feats_epoch = []
        
        for param in self.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    
    def forward(self, batch, return_attn=False):
        return self.model(batch, return_attn=return_attn)
    
    def training_step(self, batch, batch_idx):
        if self.args.contrast_loss == 'siglip':
            if self.args.use_fine_gained_loss:
                loss, loss_itc, loss_fine_gained, global_logit_scale, global_logit_bias, local_logit_scale, local_logit_bias = self(batch)
            else:
                loss, loss_itc, loss_fine_gained, global_logit_scale, global_logit_bias = self(batch)
        elif self.args.contrast_loss == 'infonce':
            if self.args.use_fine_gained_loss:
                loss, loss_itc, loss_fine_gained, local_logit_scale, local_logit_bias = self(batch)
            else:
                loss, loss_itc, loss_fine_gained = self(batch)

        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss_itc_step', loss_itc, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_itc_epoch', loss_itc, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        self.log('train_loss_fine_gained_step', loss_fine_gained, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_fine_gained_epoch', loss_fine_gained, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        if self.args.contrast_loss == 'siglip':
            self.log('global_logit_scale', global_logit_scale, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log('global_logit_bias', global_logit_bias, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            
        if self.args.use_fine_gained_loss:
            self.log('local_logit_scale', local_logit_scale, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
            self.log('local_logit_bias', local_logit_bias, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        pass
    
    def on_validation_epoch_start(self):
        self.val_image_feats_epoch = []
        self.val_text_feats_epoch = []

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            image_feat_global, text_feat_global = self.model.forward_global(batch)
        image_feat_global = image_feat_global / image_feat_global.norm(dim=1, keepdim=True)
        text_feat_global = text_feat_global / text_feat_global.norm(dim=1, keepdim=True)
        
        if self.eval_retrieval_on_epoch:
            self.val_image_feats_epoch.append(image_feat_global.detach().cpu())
            self.val_text_feats_epoch.append(text_feat_global.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.eval_retrieval_on_epoch or not self.val_image_feats_epoch:
            return

        image_feats = torch.cat(self.val_image_feats_epoch, dim=0).to(self.device)
        text_feats = torch.cat(self.val_text_feats_epoch, dim=0).to(self.device)

        if is_dist_avail_and_initialized():
            image_feats = concat_all_gather(image_feats)
            text_feats = concat_all_gather(text_feats)
            rank = dist.get_rank()
        else:
            rank = 0
        
        if rank == 0:
            max_samples = self.args.num_retrieval_samples_epoch
            if image_feats.size(0) > max_samples:
                image_feats = image_feats[:max_samples]
                text_feats = text_feats[:max_samples]

            sim_matrix = image_feats @ text_feats.t()
            img_to_txt_r1, img_to_txt_r5, img_to_txt_r10 = self.compute_retrieval_metrics(sim_matrix, 'image2text')
            txt_to_img_r1, txt_to_img_r5, txt_to_img_r10 = self.compute_retrieval_metrics(sim_matrix, 'text2image')

            self.log('val_i2t_r1_epoch', img_to_txt_r1, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            self.log('val_i2t_r5_epoch', img_to_txt_r5, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            self.log('val_i2t_r10_epoch', img_to_txt_r10, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            self.log('val_t2i_r1_epoch', txt_to_img_r1, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            self.log('val_t2i_r5_epoch', txt_to_img_r5, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
            self.log('val_t2i_r10_epoch', txt_to_img_r10, prog_bar=True, logger=True, rank_zero_only=True, sync_dist=False)
                
    def compute_retrieval_metrics(self, sim_matrix, direction='image2text'):
        """Compute retrieval metrics R@1, R@5, and R@10."""
        if direction == 'image2text':
            ranks = sim_matrix.argsort(dim=1, descending=True)
            target_ranks = torch.arange(ranks.size(0), device=ranks.device).unsqueeze(1)
        else:
            ranks = sim_matrix.argsort(dim=0, descending=True).t()
            target_ranks = torch.arange(ranks.size(0), device=ranks.device).unsqueeze(1)
        
        r1 = (ranks[:, :1] == target_ranks).sum().item() / ranks.size(0)
        r5 = (ranks[:, :5] == target_ranks).sum().item() / ranks.size(0)
        r10 = (ranks[:, :10] == target_ranks).sum().item() / ranks.size(0)
        
        return r1, r5, r10

    def configure_optimizers(self):
        ## todo

        self.trainer.fit_loop.setup_data()
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        total_training_steps = max(self.trainer.estimated_stepping_batches, 1)
        steps_per_epoch = max(total_training_steps // max(self.args.max_epochs, 1), 1)

        # warmup_steps = min(steps_per_epoch*3, self.args.warmup_steps)

        if hasattr(self.args, 'warmup_ratio') and self.args.warmup_ratio > 0:
            warmup_steps = int(total_training_steps * self.args.warmup_ratio)
        warmup_steps = max(1, min(warmup_steps, total_training_steps))

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=self.args.warmup_lr / self.args.init_lr,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        if self.args.scheduler == 'linear_warmup_cosine_lr':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(total_training_steps - warmup_steps, 1),
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps]
            )
        elif self.args.scheduler == 'linear_warmup_exp_lr':
            gamma_per_step = self.args.lr_decay_rate ** (1 / steps_per_epoch)
            scheduler = ExponentialLR(
                optimizer,
                gamma=gamma_per_step
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps]
            )
        elif self.args.scheduler == 'None':
            scheduler = None
        else:
            raise NotImplementedError()
        
        if scheduler is None:
            return optimizer
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

    def load_from_deepspeed(self, ckpt_path):
        source_checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        if 'state_dict' in source_checkpoint:
            source_state_dict = source_checkpoint['state_dict']
        else:
            source_state_dict = source_checkpoint
        
        source_state_dict_cleaned = {}
        for key, value in source_state_dict.items():
            new_key = key
            if key.startswith('module.'):
                new_key = key[7:]
            source_state_dict_cleaned[new_key] = value
        
        current_state_dict = self.state_dict()
        matched_params = 0
        unmatched_params = 0
        
        for param_name, param_value in source_state_dict_cleaned.items():
            if param_name in current_state_dict:
                if param_value.shape == current_state_dict[param_name].shape:
                    current_state_dict[param_name] = param_value
                    print(f"Matched: {param_name} - Shape: {param_value.shape}")
                    matched_params += 1
                else:
                    print(f"Shape mismatch: {param_name} - Source: {param_value.shape}, Target: {current_state_dict[param_name].shape}")
                    unmatched_params += 1
            else:
                print(f"Parameter not in target model: {param_name}")
                unmatched_params += 1
        
        self.load_state_dict(current_state_dict, strict=False)
        
        print(f"Successfully loaded {matched_params} parameters from DeepSpeed checkpoint")
        print(f"Skipped {unmatched_params} parameters due to mismatch or not found")
