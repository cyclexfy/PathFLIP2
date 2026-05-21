import argparse

def get_args():
    
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--results_dir', type=str, default="outputs")
    parser.add_argument('--filename', type=str, default="pathflip_vlm_stage1")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--strategy_name', type=str, default='deepspeed')
    # devices
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    # parser.add_argument('--devices', type=str, default='1')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=5)
    
    parser.add_argument('--save_by', type=str, default='epoch', choices=['epoch', 'step'], help='save checkpoint by epoch or step')
    parser.add_argument('--save_interval', type=int, default=1, help='checkpoint save interval')
    parser.add_argument('--save_last_k', type=int, default=-1, help='keep only the latest k checkpoints; -1 keeps all')
    parser.add_argument('--log_every_n_steps', type=int, default=50) 
    # train
    parser.add_argument('--freeze_llm', action='store_true', help='', default=False)
    parser.add_argument('--freeze_image_encoder', action='store_true', help='', default=False)
    parser.add_argument('--freeze_text_encoder', action='store_true', help='', default=False)
    # optimization
    parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
    parser.add_argument('--min_lr', type=float, default=5e-6, help='optimizer min learning rate')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.05, help='optimizer warmup ratio')
    parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
    parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16) 
    # checkpoint
    parser.add_argument('--stage1_ckpt_path', type=str, default='', help='')
    parser.add_argument('--init_checkpoint', type=str, default='')
    parser.add_argument('--align_model_ckpt_path', type=str, default='')
    # data
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_text_length', type=int, default=512) # 512
    parser.add_argument('--max_dataset_length', type=int, default=None)
    parser.add_argument('--caption_repeat', type=int, default=5)
    # todo change
    # parser.add_argument('--train_data_path', type=str, default='datasets/SlideInstruct_train_stage1_caption_valid.json')
    parser.add_argument('--train_data_path', nargs='*', default=['datasets/SlideInstruct_train_stage1_caption_valid.json'])
    parser.add_argument('--val_data_path', type=str, default=None)
    parser.add_argument('--conversation_type', type=str, default='qwen')
    parser.add_argument('--patch_sample', action='store_false', help='use patch sample or not', default=True)
    parser.add_argument('--num_patch_samples', type=int, default=4096)
    parser.add_argument('--patch_size', type=int, default=256, help='patch size for WSI processing')
    parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--num_fine_gained_heads', type=int, default=8, help='number of attention heads for fine-grained loss')
    # model paramenters
    parser.add_argument('--image_embed_dim', type=int, default=512)
    parser.add_argument('--embed_dim', type=int, default=512) # 256
    parser.add_argument('--llm_name_or_path', type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ### lora
    parser.add_argument('--llm_lora', action='store_true', help='use lora or not', default=False)
    parser.add_argument('--llm_lora_alpha', type=int, default=128)
    parser.add_argument('--llm_lora_dropout', type=float, default=0.05)
    parser.add_argument('--llm_lora_r', type=int, default=64)
    # LoRA parameters for text encoder
    parser.add_argument('--text_encoder_name_or_path', type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument('--text_encoder_use_lora', action='store_false', default=True, help='whether to use LoRA for text encoder fine-tuning')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA matrix rank')
    parser.add_argument('--lora_alpha', type=int, default=64, help='LoRA scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA layer dropout rate')
    parser.add_argument('--lora_target_modules', type=str, nargs='+', default=["query", "key", "value", "attention.output.dense", "intermediate.dense", "output.dense"], help='target modules to apply LoRA adapters')

    # Soft TopK
    parser.add_argument('--use_soft_topk', action='store_true', default=False, help='whether to use soft top-k loss')
    parser.add_argument('--top_k', type=int, default=128, help='number of top-k patches to select')
    # Loss

    # evaluation
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)

    args = parser.parse_args()
    return args
