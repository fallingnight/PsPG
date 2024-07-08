import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from validator import validate
from utils.losses import CLIPLoss, AsymmetricLoss, LossMixture, LossRanking
from model.text_encoder import preprocess_text
from helper import AverageMeter
from utils.metrics import mAP


def train_clip(model, images, tokenized_texts):
    criterion = CLIPLoss()
    with autocast():
        image_logits, _, _ = model(images, tokenized_texts)
    text_logits = image_logits.t()
    loss = criterion(image_logits, text_logits)
    return loss


def train_coop(model, cfg, images, target):
    criterion = AsymmetricLoss(
        gamma_neg=cfg.LOSS.ASL_GAMMA_NEG, gamma_pos=cfg.LOSS.ASL_GAMMA_POS
    )
    with autocast():
        _, lp_logits, _ = model(images)
    loss = 0.1 * criterion(lp_logits, target)
    return loss, lp_logits


def train_pspg(model, cfg, images, target):
    if cfg.PROMPT.ENABLE_PAIRLOSS:
        criterion = LossMixture(
            gamma_neg=cfg.LOSS.ASL_GAMMA_NEG,
            gamma_pos=cfg.LOSS.ASL_GAMMA_POS,
        )
    else:
        criterion = LossRanking(
            gamma_neg=cfg.LOSS.ASL_GAMMA_NEG,
            gamma_pos=cfg.LOSS.ASL_GAMMA_POS,
        )
    with autocast():
        _, lp_logits, pair_logits = model(images)
    loss = criterion(lp_logits, target, pair_logits)
    return loss, lp_logits


def set_model_train(model, cfg):
    if not isinstance(model, nn.DataParallel):
        if not cfg.TRAIN.FINETUNE_CLIP:
            model.image_encoder.eval()
            model.text_encoder.eval()

        if not cfg.MODEL.ENABLE_LP:
            model.prompt_learner.eval()
    else:
        if not cfg.TRAIN.FINETUNE_CLIP:
            model.module.image_encoder.eval()
            model.module.text_encoder.eval()

        if not cfg.MODEL.ENABLE_LP:
            model.module.prompt_learner.eval()


def train_epoch(
    train_loader,
    val_loader,
    model,
    optimizer,
    cfg,
    model_ema=None,
):
    losses = AverageMeter()
    train_mAP = AverageMeter()
    Softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    model.train()
    set_model_train(model, cfg)
    timestamp = time.time()
    epoch_time = time.time()
    accumulation_steps = cfg.TRAIN.ACCUMULATION_STEPS

    for i, (images, texts, target) in enumerate(train_loader):
        images = images.to(device)
        tokenized_texts = preprocess_text(texts, cfg.MODEL)
        tokenized_texts = {
            "input_ids": tokenized_texts["input_ids"].to(device),
            "attention_mask": tokenized_texts["attention_mask"].to(device),
        }
        target = target.to(device)
        loss = None
        output = None
        if cfg.TRAIN.FINETUNE_CLIP and not cfg.MODEL.ENABLE_LP:
            loss = train_clip(model, images, tokenized_texts)
        elif cfg.MODEL.PROMPT.NAME == "coop":
            loss, output = train_coop(model, cfg.MODEL, images, target)
        elif cfg.MODEL.PROMPT.NAME == "pspg":
            loss, output = train_pspg(model, cfg.MODEL, images, target)
        elif cfg.MODEL.PROMPT.NAME == "linear":
            loss, output = train_coop(model, cfg.MODEL, images, target)

        if output is not None:
            if output.dim() == 3:
                pred = Softmax(output.detach())[:, 0]
            else:
                pred = sigmoid(output.detach())
            mAP_value = mAP(target.cpu().numpy(), pred.cpu().numpy())
            train_mAP.update(mAP_value, images.size(0))
        losses.update(loss.item(), images.size(0))
        if accumulation_steps > 1:
            loss = loss / accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
        if i % cfg.TRAIN.PRINT_FREQ == 0:
            info = f"Train: [{i}/{len(train_loader)}]\tTime:{(time.time()-timestamp):.2f}s\tLoss(avg):{losses.avg:.4f}\t"
            if cfg.MODEL.ENABLE_LP:
                info = info + f"mAP(avg):{train_mAP.avg:.4f}"
            print(info)
            timestamp = time.time()
        if (
            cfg.TRAIN.VAL_FREQ_IN_EPOCH != -1
            and (i + 1) % cfg.TRAIN.VAL_FREQ_IN_EPOCH == 0
        ):
            result_dict = validate(val_loader, model, cfg)
            macro_auc = result_dict["macro_auc"]
            micro_auc = result_dict["micro_auc"]
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
                f"Test: [{i+1}/{len(train_loader)}]\t"
                f"Macro_AUC: {macro_auc:.2f}\t"
                f"Micro_AUC: {micro_auc:.2f}\t"
                f"mAP: {map_score:.2f}\t"
                f"MCC: {mcc_score:.2f}\t"
                f"Accuracy: {acc_score:.2f}\t"
                f"Macro_Precision: {macro_pcs:.2f}\t"
                f"Micro_Precision: {micro_pcs:.2f}\t"
                f"Macro_Recall: {macro_rc:.2f}\t"
                f"Micro_Recall: {micro_rc:.2f}\t"
                f"Macro_F1: {marco_f1:.2f}\t"
                f"Micro_F1: {micro_f1:.2f}\t"
                f"with threshold: {cfg.TEST.THRESHOLD:.2f}"
            )
            print(val_info)
            model.train()
            set_model_train(model, cfg)
    epoch_time = time.time() - epoch_time
    return losses, epoch_time, train_mAP
