from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ExponentialLR, SequentialLR

from .model_pathflip_vlm import pathflip_vlm

class pathflip_vlm_pl(LightningModule):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.model = pathflip_vlm(
            image_embed_dim=args.image_embed_dim,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            text_encoder=args.text_encoder_name_or_path,
            num_fine_gained_heads=args.num_fine_gained_heads,
            text_encoder_use_lora=args.text_encoder_use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            keep_k=args.top_k,
            use_soft_topk=args.use_soft_topk,
            tokenizer=tokenizer,
            llm_name_or_path=args.llm_name_or_path,
            llm_lora=args.llm_lora,
            llm_lora_alpha=args.llm_lora_alpha,
            llm_lora_dropout=args.llm_lora_dropout,
            llm_lora_r=args.llm_lora_r,
            freeze_llm=args.freeze_llm,
            freeze_image_encoder=args.freeze_image_encoder,
            freeze_text_encoder=args.freeze_text_encoder,
        )
        self.save_hyperparameters(args)
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        loss = self(batch)

        # self.log('global_step', float(self.global_step), prog_bar=False, logger=True)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss_epoch', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)        
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def on_train_epoch_end(self):
        pass

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        total_training_steps = max(self.trainer.estimated_stepping_batches, 1)
        steps_per_epoch = max(total_training_steps // max(self.args.max_epochs, 1), 1)

        # warmup_steps = min(steps_per_epoch*1, self.args.warmup_steps)

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
    
    def load_from_align_model(self, align_model_ckpt_path):

        source_checkpoint = torch.load(align_model_ckpt_path, map_location='cpu')
        if 'state_dict' in source_checkpoint:
            source_state_dict = source_checkpoint['state_dict']
        else:
            source_state_dict = source_checkpoint
        
        current_state_dict = self.state_dict()
        matched_params = 0
        unmatched_params = 0
        
        for param_name, param_value in source_state_dict.items():
            if param_name in current_state_dict:
                if param_value.shape == current_state_dict[param_name].shape:
                    current_state_dict[param_name] = param_value
                    # print(f"Matched: {param_name} - Shape: {param_value.shape}")
                    matched_params += 1
                else:
                    print(f"Shape mismatch: {param_name} - Source: {param_value.shape}, Target: {current_state_dict[param_name].shape}")
                    unmatched_params += 1
            else:
                print(f"Parameter not in target model: {param_name}")
                unmatched_params += 1
        
        self.load_state_dict(current_state_dict, strict=False)
        
        print(f"Successfully loaded {matched_params} parameters from align model")
        print(f"Skipped {unmatched_params} parameters due to mismatch or not found")
    
    def load_from_stage1(self, stage1_ckpt):
        ckpt = torch.load(stage1_ckpt)
        self.load_state_dict(ckpt, strict=False)
