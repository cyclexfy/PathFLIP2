import argparse

def get_args():
    
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--results_dir', type=str, default="outputs")
    parser.add_argument('--filename', type=str, default="pathflip_align")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    # devices
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    # parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--save_every_n_epochs', type=int, default=25)
    parser.add_argument('--log_every_n_steps', type=int, default=25) 
    # optimization
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='optimizer weight decay')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='optimizer warmup ratio')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1) 
    # checkpoint
    parser.add_argument('--init_checkpoint', type=str, default='')
    # data
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16) # 8
    parser.add_argument('--max_text_length', type=int, default=512) # 512
    parser.add_argument('--max_dataset_length', type=int, default=None)
    parser.add_argument('--train_data_path', type=str, default='datasets/SlideInstruct_train_stage1_caption_fine_grained.json')
    parser.add_argument('--val_data_path', type=str, default='datasets/SlideBench-Caption-TCGA-plus.json')
    parser.add_argument('--patch_sample', action='store_false', help='use patch sample or not', default=True)
    parser.add_argument('--num_patch_samples', type=int, default=4096)
    parser.add_argument('--num_text_samples', type=int, default=8) ## 
    parser.add_argument('--enriched_descrption', action='store_true', default=False)
    # model paramenters
    parser.add_argument('--itm', action='store_false', help='use global image-text matching or not', default=True)
    parser.add_argument('--text_encoder', type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--tokenizer', type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--freeze_text_encoder', action='store_true', default=False)
    parser.add_argument('--text_embed_dim', type=int, default=768, help='')
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512) # 256
    parser.add_argument('--text_pooling_type', type=str, default='cls_token', help='text pooling type, cls_token or mean')
    # Model structure parameters
    parser.add_argument('--slide_ngrids', type=int, default=1000, help='number of slide positional embedding grids')
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for WSI processing')
    parser.add_argument('--num_longnet_layers', type=int, default=4, help='number of LongNet encoder layers')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.25, help='dropout rate')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='drop path rate')
    parser.add_argument('--num_query_latents', type=int, default=256, help='number of query latents for resampler')
    parser.add_argument('--num_resampler_layers', type=int, default=4, help='number of resampler layers')
    parser.add_argument('--num_flair_heads', type=int, default=8, help='number of attention heads for fine-grained loss')
    # LoRA parameters for text encoder
    parser.add_argument('--text_encoder_use_lora', action='store_true', default=False, help='whether to use LoRA for text encoder fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA matrix rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA layer dropout rate')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"], help='target modules to apply LoRA adapters')
    # Loss
    parser.add_argument('--contrast_loss', type=str, default='infonce', help='contrastive loss type, infonce or siglip')
    parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of InfoNCE Imgae-Text Contrastive Loss')
    parser.add_argument('--init_logit_scale', type=float, default=10.0, help='init logit scale for SigLIP Imgae-Text Contrastive Loss')
    parser.add_argument('--init_logit_bias', type=float, default=-10.0, help='init logit bias for SigLIP Imgae-Text Contrastive Loss')

    parser.add_argument('--use_fine_gained_loss', action='store_false', help='use fine-grained loss or not', default=True)
    # Soft TopK
    parser.add_argument('--use_soft_topk', action='store_true', default=False, help='whether to use soft top-k loss')
    parser.add_argument('--top_k', type=int, default=128, help='number of top-k patches to select')
    parser.add_argument('--topk_temperature', type=float, default=0.5, help='temperature for soft top-k')
    # parser.add_argument('--text_sample_num', type=int, default=6) # 8

    # evaluation
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--rerank_cand_num', type=int, default=64)
    # parser.add_argument('--eval_retrieval_on_step', action='store_true', default=False)
    parser.add_argument('--eval_retrieval_on_epoch', action='store_true', default=False)
    # parser.add_argument('--num_retrieval_steps', type=int, default=2)
    parser.add_argument('--num_retrieval_samples_epoch', type=int, default=4096)

    args = parser.parse_args()
    return args
