import json
import os
import torch
import time
from torch.cuda.amp import autocast
from helper import write_logfile

from utils.metrics import (
    calc_roc,
    calc_map,
    fbetaMacro,
    fbetaMicro,
    Average_MCC,
    accuracy,
    precisionMacro,
    precisionMicro,
    recallMacro,
    recallMicro,
)


def threshold_to_binary(y_pred, threshold):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred


def generate_json_output(y_pred, y_label, y_true):
    num_samples = y_pred.shape[0]

    json_data = []

    for i in range(num_samples):
        sample_data = {
            "id": i + 1,
            "probabilities": y_pred[i, :].tolist(),
            "labels": y_label[i, :].tolist(),
            "ground_truth": y_true[i, :].tolist(),
        }

        json_data.append(sample_data)

    return json_data


def build_output_file(cfg, json_data, model_name=None):
    folder = os.path.join(cfg.OUTPUT_DIR, model_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = os.path.join(folder, model_name + "_output.json")
    with open(file_path, "w") as json_file:
        json.dump(json_data, json_file)


def validate(val_loader, model, cfg, model_name=None):
    preds = []
    targets = []
    Softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model.to(device)
    model.eval()
    timestamp = time.time()
    with torch.no_grad():
        for i, (images, _, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)
            with autocast():
                _, outputs, _ = model(images)
            if outputs.dim() == 3:
                outputs = Softmax(outputs).cpu()[:, 0]
            else:
                outputs = sigmoid(outputs).cpu()
            preds.append(outputs.cpu())
            targets.append(target.cpu())
            batch_time = time.time() - timestamp
            timestamp = time.time()

        y_pred = torch.cat(preds).numpy()
        ground_truth = torch.cat(targets)
        y_true = ground_truth.numpy()
        num_classes = ground_truth.shape[1]
        y_label = threshold_to_binary(y_pred, cfg.TEST.THRESHOLD)
        macro_auc, micro_auc, aucs = calc_roc(y_true, y_pred, num_classes)
        top_10_auc = sorted(aucs, reverse=True)[:10]
        top_10_mean_auc = sum(top_10_auc) / len(top_10_auc)
        map_score = calc_map(y_true, y_pred, num_classes)
        mcc_score = Average_MCC(y_true, y_label, num_classes)
        acc_score, _ = accuracy(y_true, y_label)
        macro_pcs, _ = precisionMacro(y_true, y_label)
        micro_pcs = precisionMicro(y_true, y_label)
        macro_rc, _ = recallMacro(y_true, y_label)
        micro_rc = recallMicro(y_true, y_label)
        marco_f1, _ = fbetaMacro(y_true, y_label)
        micro_f1 = fbetaMicro(y_true, y_label)
        if cfg.TEST.SAVE_PRED == True:
            json_data = generate_json_output(y_pred, y_label, y_true)
            build_output_file(cfg, json_data, model_name)
            print("JSON file generated successfully.")
    result_dict = {
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "top10_auc": top_10_mean_auc,
        "mAP_score": map_score,
        "mcc_score": mcc_score,
        "acc_score": acc_score,
        "macro_pcs": macro_pcs,
        "micro_pcs": micro_pcs,
        "macro_rc": macro_rc,
        "micro_rc": micro_rc,
        "marco_f1": marco_f1,
        "micro_f1": micro_f1,
    }
    torch.cuda.empty_cache()
    return result_dict


def run_val(epoch, logger, val_loader, model, cfg, model_name):
    result_dict = validate(val_loader, model, cfg, model_name)

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
        f"Test: [{epoch}/{cfg.OPTIM.MAX_EPOCH}]\t"
        f"Macro_AUC: {macro_auc:.3f}\t"
        f"Micro_AUC: {micro_auc:.3f}\t"
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
    write_logfile(val_info, logger)
    return result_dict["macro_auc"]
