from utils.opt_parser import args

import random
import numpy as np
import torch.utils.data

from model import build_pspg
from dataloader import build_dataset
from utils.opt_parser import parse_opt
from utils.cfg_builder import get_cfg
from helper import (
    load_model_only,
)
from validator import validate


def build_dataloaders(cfg):
    test_split = cfg.DATASET.TEST_SPLIT
    test_dataset = build_dataset(
        cfg, cfg.DATASET.NAME, test_split, cfg.DATASET.TEST_FILE_PATH
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
        shuffle=cfg.DATALOADER.TEST.SHUFFLE,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    classnames = test_dataset.classnames
    return test_loader, classnames


def main():
    cfg = get_cfg(args)
    seed = cfg.SEED
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
    test_loader, classnames = build_dataloaders(cfg)
    model, _ = build_pspg(cfg, classnames)

    epoch = 0
    if cfg.CHECKPOINT is not None:
        model, epoch = load_model_only(cfg.CHECKPOINT, model)
    else:
        print("use clip backbone")
        # raise FileNotFoundError("please enter the checkpoint path!")
    print(f"Epoch: {epoch}")
    print("Evaluating the checkpoint...")
    result_dict = validate(test_loader, model, cfg, args.model_name)
    macro_auc = result_dict["macro_auc"]
    micro_auc = result_dict["micro_auc"]
    top10_auc = result_dict["top10_auc"]
    map_score = result_dict["mAP_score"]
    mcc_score = result_dict["mcc_score"]
    acc_score = result_dict["acc_score"]
    macro_pcs = result_dict["macro_pcs"]
    micro_pcs = result_dict["micro_pcs"]
    macro_rc = result_dict["macro_rc"]
    micro_rc = result_dict["micro_rc"]
    marco_f1 = result_dict["marco_f1"]
    micro_f1 = result_dict["micro_f1"]

    val_info = (
        f"Macro_AUC: {macro_auc:.3f}\t"
        f"Micro_AUC: {micro_auc:.3f}\t"
        f"Top 10 AUC(mean): {top10_auc:.3f}\t"
        f"mAP: {map_score:.3f}\t"
        f"MCC: {mcc_score:.3f}\t"
        f"Accuracy: {acc_score:.3f}\t"
        f"Macro_Precision: {macro_pcs:.3f}\t"
        f"Micro_Precision: {micro_pcs:.3f}\t"
        f"Macro_Recall: {macro_rc:.3f}\t"
        f"Micro_Recall: {micro_rc:.3f}\t"
        f"Macro_F1: {marco_f1:.3f}\t"
        f"Micro_F1: {micro_f1:.3f}\t"
        f"with threshold: {cfg.TEST.THRESHOLD:.3f}"
    )
    print(val_info)


if __name__ == "__main__":
    main()
