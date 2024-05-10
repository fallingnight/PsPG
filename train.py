from utils.opt_parser import args

import os
import random
import sys
import numpy as np
import torch.utils.data
import torch.nn as nn

from model import build_pspg
from dataloader import build_dataset
from utils.cfg_builder import get_cfg
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_scheduler
from helper import (
    build_logger,
    write_logfile,
    save_checkpoint,
    load_checkpoint,
    load_model_only,
)
from timm.utils import ModelEmaV2
from trainer import train_epoch
from validator import run_val


def build_dataloaders(cfg):

    train_split = cfg.DATASET.TRAIN_SPLIT
    val_split = cfg.DATASET.VAL_SPLIT
    test_split = cfg.DATASET.TEST_SPLIT
    train_dataset = build_dataset(
        cfg, cfg.DATASET.NAME, train_split, cfg.DATASET.TRAIN_FILE_PATH
    )
    val_dataset = build_dataset(
        cfg, cfg.DATASET.NAME, val_split, cfg.DATASET.VAL_FILE_PATH
    )
    test_dataset = build_dataset(
        cfg, cfg.DATASET.NAME, test_split, cfg.DATASET.TEST_FILE_PATH
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.DATALOADER.TRAIN.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TRAIN.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.DATALOADER.VAL.BATCH_SIZE,
        shuffle=cfg.DATALOADER.VAL.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TEST.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    classnames = val_dataset.classnames
    return train_loader, val_loader, test_loader, classnames


def eval_best(cfg, test_loader, logger, model, model_name):

    test_info = f"Evaluating the best model..."
    print(test_info)
    write_logfile(test_info, logger)
    best_path = os.path.join(cfg.OUTPUT_DIR, model_name, "best.pth")
    model, best_epoch = load_model_only(best_path, model)
    run_val(best_epoch, logger, test_loader, model, cfg, model_name)


def main():
    # detailed config comments in cfg_builder.py
    cfg = get_cfg(args)
    seed = cfg.SEED
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    train_loader, val_loader, test_loader, classnames = build_dataloaders(cfg)
    model, model_name = build_pspg(cfg, classnames)
    model_ema = None

    # Note that the released models do not use ema, but we provide ema support
    if cfg.OPTIM.EMA:
        model_ema = ModelEmaV2(
            model,
            decay=cfg.OPTIM.EMA_DECAY,
        )
        print(f"Using EMA with decay = {cfg.OPTIM.EMA_DECAY:.4f}")
    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    logger = build_logger(cfg, model_name)
    command = " ".join(sys.argv)
    write_logfile(command, logger)
    if cfg.VERBOSE:
        write_logfile(cfg, logger)
        write_logfile(model, logger)
    print(f"accumulation step:{cfg.TRAIN.ACCUMULATION_STEPS}")
    start_epoch = 0
    if cfg.CHECKPOINT is not None:
        if not cfg.START_AFRESH:
            model, optimizer, lr_scheduler, start_epoch = load_checkpoint(
                cfg.CHECKPOINT, model, optimizer, lr_scheduler
            )
        else:
            model, _ = load_model_only(cfg.CHECKPOINT, model)

    # for we did not have an available server with multiple GPU, the feasibility of parallel training has not been tested
    device_count = torch.cuda.device_count()
    if device_count > 1:
        print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        device_ids = list(range(device_count))
        model = nn.DataParallel(model, device_ids=device_ids)

    best_AUC = run_val(start_epoch, logger, val_loader, model, cfg, model_name)
    save_dict = {
        "epoch": 0,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
    }
    save_checkpoint(save_dict, 0, True, cfg.OUTPUT_DIR, model_name)

    for epoch in range(start_epoch, cfg.OPTIM.MAX_EPOCH):
        train_loss, epoch_time, train_mAP = train_epoch(
            train_loader, val_loader, model, optimizer, cfg, model_ema
        )
        train_info = (
            f"Train: [{epoch+1}/{cfg.OPTIM.MAX_EPOCH}]\t"
            f"Total Time: {epoch_time:.2f}s\t"
            f"Loss(avg): {train_loss.avg:.4f}\t"
            f"mAP(avg):{train_mAP.avg:.4f}"
        )
        print(train_info)
        write_logfile(train_info, logger)
        lr_scheduler.step(epoch)

        auc_score = run_val(epoch + 1, logger, val_loader, model, cfg, model_name)
        if model_ema is not None:
            ema_info = f"EMA:[{epoch+1}/{cfg.OPTIM.MAX_EPOCH}]"
            print(ema_info)
            write_logfile(ema_info, logger)
            ema_auc = run_val(
                epoch + 1, logger, val_loader, model_ema.module, cfg, model_name
            )
            if ema_auc > auc_score:
                auc_score = ema_auc
        is_best_epoch = auc_score > best_AUC
        if is_best_epoch:
            best_AUC = auc_score
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": lr_scheduler.state_dict(),
        }
        if model_ema is not None:
            save_dict["model_ema"] = model_ema.module.state_dict()
        save_checkpoint(save_dict, epoch, is_best_epoch, cfg.OUTPUT_DIR, model_name)
    eval_best(cfg, test_loader, logger, model, model_name)


if __name__ == "__main__":
    main()
