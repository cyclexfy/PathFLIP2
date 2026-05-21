import os
import torch
import warnings
import sys
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from transformers import AutoTokenizer

from pytorch_lightning import Trainer, strategies
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from datetime import datetime
from transformers import AutoTokenizer

from .process_args_vlm import get_args
from src.model.pathflip_vlm_pl import pathflip_vlm_pl
from src.dataset.dataset_pathflip_vlm import datamodule_pathflip_vlm
from src.tools.process_args_vlm import get_args

def main(args):
    pl.seed_everything(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_name_or_path)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    text_encoder_tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name_or_path)

    # model
    if args.init_checkpoint:
        model = pathflip_vlm_pl.load_from_checkpoint(args.init_checkpoint, strict=False)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = pathflip_vlm_pl(args, tokenizer)
        if args.align_model_ckpt_path:
            model.load_from_align_model(args.align_model_ckpt_path)
        if args.stage1_ckpt_path:
            model.load_from_stage1(args.stage1_ckpt_path)

    print('total params:', sum(p.numel() for p in model.parameters()))

    datamodule = datamodule_pathflip_vlm(args=args, tokenizer=tokenizer, text_encoder_tokenizer=text_encoder_tokenizer)

    callbacks = []
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(args.results_dir, f"{args.filename}_{timestamp}")

    save_interval = args.save_interval
    save_by = args.save_by
    #     save_by = 'epoch'
    #     save_interval = args.save_every_n_epochs
    
    ckpt_kwargs = {
        'dirpath': save_dir,
        'save_last': True,
        'save_top_k': -1,
    }
    
    if save_by == 'epoch':
        ckpt_kwargs.update({
            'filename': '{epoch:02d}',
            'every_n_epochs': save_interval,
            'save_on_train_epoch_end': True,
        })
    else:
        ckpt_kwargs.update({
            'filename': '{epoch:02d}-{step:06d}',
            'every_n_train_steps': save_interval,
            'save_on_train_epoch_end': False,
        })
    
    callbacks.append(plc.ModelCheckpoint(**ckpt_kwargs))
    
    if len(args.devices.split(',')) > 1 or int(args.devices) > 1:
        if args.strategy_name == 'deepspeed':
            strategy = strategies.DeepSpeedStrategy(stage=2)
        else:
            strategy = strategies.DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"

    csv_logger = CSVLogger(save_dir=save_dir)
    tb_logger = TensorBoardLogger(save_dir=save_dir)

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        strategy=strategy,
        logger=[csv_logger, tb_logger],
        log_every_n_steps=args.log_every_n_steps,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    args = get_args()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)