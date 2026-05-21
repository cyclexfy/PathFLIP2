import math


class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        
        self._last_epoch = -1

    def step(self):
        """
        Update the learning rate in a PyTorch Lightning-compatible format.
        """
        self._step_count += 1
        if global_step < self.warmup_steps:
            warmup_lr_schedule(
                step=global_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_epoch = max(0, global_step - self.warmup_steps) // steps_per_epoch
            step_lr_schedule(
                epoch=step_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        

    def step(self):
        """
        Update the learning rate in a PyTorch Lightning-compatible format.
        """
        self._step_count += 1
        global_step = self._step_count - 1
        if global_step < self.warmup_steps:
            warmup_lr_schedule(
                step=global_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            steps_per_epoch = max(1, self.warmup_steps // max(1, self.max_epoch // 10))
            total_steps = self.max_epoch * steps_per_epoch
            remaining_steps = global_step - self.warmup_steps
            max_cosine_steps = max(1, total_steps - self.warmup_steps)
            
            cosine_epoch = min(self.max_epoch - 1, remaining_steps // steps_per_epoch)
            
            cosine_lr_schedule(
                epoch=cosine_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


class ConstantLRScheduler:
    def __init__(self, optimizer, init_lr, warmup_start_lr=-1, warmup_steps=0, **kwargs):
        self.optimizer = optimizer
        self.lr = init_lr
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr
        self.warmup_steps = warmup_steps
        
    
    def step(self):
        """
        Update the learning rate in a PyTorch Lightning-compatible format.
        """
        self._step_count += 1
        global_step = self._step_count - 1
        if global_step < self.warmup_steps:
            warmup_lr_schedule(
                step=global_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.lr,
            )
        else:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
