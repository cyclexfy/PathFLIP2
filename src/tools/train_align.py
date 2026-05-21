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

from .process_args_align import get_args
from src.model.pathflip_pl import pathflip_pl
from src.dataset.dataset_pathflip import datamodule_pathflip


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = pathflip_pl.load_from_checkpoint(args.init_checkpoint, strict=False)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = pathflip_pl(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    datamodule = datamodule_pathflip(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        patch_sample=args.patch_sample,
        num_patch_samples=args.num_patch_samples,
        num_text_samples=args.num_text_samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        args=args
    )

    callbacks = []
    timestamp = datetime.now().strftime("%m%d_%H%M")
    save_dir = os.path.join(args.results_dir, args.filename, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    callbacks.append(
        plc.ModelCheckpoint(
            dirpath=save_dir,
            filename='{epoch:02d}', 
            every_n_epochs=args.save_every_n_epochs, 
            save_top_k=-1,
            save_on_train_epoch_end=True,
            save_last=True,
        )
    )
    
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
    trainer.validate(model, datamodule=datamodule)

    return

if __name__ == '__main__':
    args = get_args()
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)