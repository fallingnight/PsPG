from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def build_scheduler(cfg, optimizer, iter_per_epoch):
    num_steps = int(cfg.OPTIM.MAX_EPOCH * iter_per_epoch)
    warmup_steps = int(cfg.OPTIM.WARMUP_EPOCHS * iter_per_epoch)
    decay_steps = int(cfg.OPTIM.DECAY_EPOCHS * iter_per_epoch)

    lr_scheduler = None
    if cfg.OPTIM.LR_SCHEDULER == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.0,
            lr_min=cfg.OPTIM.MIN_LR,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    elif cfg.OPTIM.LR_SCHEDULE == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=cfg.OPTIM.DECAY_RATE,
            warmup_lr_init=cfg.OPTIM.WARMUP_LR,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler
