from glob import glob
import json
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)

LABELS = []
THRESHOLD = 0.5


def get_labels(dataset_name):
    LABELS = {}
    LABELS["CXR"] = [
        "Pericardial Effusion, Calcification",
        "Cardiac Malposition",
        "Cardiomegaly",
        "No Finding",
        "Pneumothorax",
        "Cavity, Cyst",
        "Mediastinal Lesion",
        "Masses, Nodules",
        "Rib Fracture",
        "Pulmonary Arterial and/or Venous Hypertension",
        "Consolidation",
        "Interstitial Lung Disease",
        "Pleural Effusion",
        "Pleural Thickening, Adhesions, Calcification",
        "Scoliosis",
        "Clavicular Fracture",
        "Obstructive Atelectasis",
        "Obstructive Emphysema",
    ]

    LABELS["Chexpert"] = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    LABELS["Vindr"] = [
        "Aortic enlargement",
        "Atelectasis",
        "Calcification",
        "Cardiomegaly",
        "Clavicle fracture",
        "Consolidation",
        "Emphysema",
        "Enlarged PA",
        "Interstitial lung disease",
        "Infiltration",
        "Lung Opacity",
        "Lung cavity",
        "Lung cyst",
        "Mediastinal shift",
        "Nodule/Mass",
        "Pleural effusion",
        "Pleural thickening",
        "Pneumothorax",
        "Pulmonary fibrosis",
        "Rib fracture",
        "Other lesion",
        "COPD",
        "Lung tumor",
        "Pneumonia",
        "Tuberculosis",
        "Other disease",
        "No finding",
    ]

    LABELS["Padchest"] = [
        "normal",
        "COPD signs",
        "unchanged",
        "chronic changes",
        "cardiomegaly",
        "aortic elongation",
        "scoliosis",
        "vertebral degenerative changes",
        "interstitial pattern",
        "pleural effusion",
        "air trapping",
        "aortic atheromatosis",
        "costophrenic angle blunting",
        "pneumonia",
        "apical pleural thickening",
        "vascular hilar enlargement",
        "alveolar pattern",
        "infiltrates",
        "laminar atelectasis",
        "fibrotic band",
        "kyphosis",
        "increased density",
        "pacemaker",
        "callus rib fracture",
        "pseudonodule",
        "calcified granuloma",
        "volume loss",
        "nodule",
        "atelectasis",
        "hilar congestion",
        "hemidiaphragm elevation",
        "sternotomy",
        "suboptimal study",
        "NSG tube",
        "hiatal hernia",
        "heart insufficiency",
        "bronchiectasis",
        "vertebral anterior compression",
        "suture material",
        "bronchovascular markings",
        "hilar enlargement",
        "diaphragmatic eventration",
        "endotracheal tube",
        "nipple shadow",
        "central venous catheter via jugular vein",
        "consolidation",
        "metal",
        "emphysema",
        "gynecomastia",
        "calcified densities",
        "goiter",
        "dual chamber device",
        "osteosynthesis material",
        "flattened diaphragm",
        "aortic button enlargement",
        "tracheostomy tube",
        "supra aortic elongation",
        "central venous catheter via subclavian vein",
        "mammary prosthesis",
        "single chamber device",
        "pulmonary mass",
        "pleural thickening",
        "tracheal shift",
        "granuloma",
        "osteopenia",
        "descendent aortic elongation",
        "hypoexpansion",
        "bullas",
        "hyperinflated lung",
        "tuberculosis sequelae",
        "superior mediastinal enlargement",
        "sclerotic bone lesion",
        "lobar atelectasis",
        "pulmonary fibrosis",
        "mediastinic lipomatosis",
        "rib fracture",
        "hypoexpansion basal",
        "azygos lobe",
        "vascular redistribution",
        "mastectomy",
        "surgery neck",
        "central venous catheter",
        "minor fissure thickening",
        "ground glass pattern",
        "calcified adenopathy",
        "dai",
        "adenopathy",
        "pulmonary edema",
        "artificial heart valve",
        "reservoir central venous catheter",
        "mediastinal enlargement",
        "axial hyperostosis",
        "cavitation",
        "non axial articular degenerative changes",
        "pneumothorax",
        "pectum excavatum",
        "vertebral compression",
        "calcified pleural thickening",
        "humeral fracture",
        "multiple nodules",
        "exclude",
        "surgery breast",
        "costochondral junction hypertrophy",
        "clavicle fracture",
        "vertebral fracture",
        "lung metastasis",
        "osteoporosis",
        "mediastinal mass",
    ]

    LABELS["Chestxray"] = [
        "Fibrosis",
        "Consolidation",
        "Emphysema",
        "No Finding",
        "Infiltration",
        "Mass",
        "Effusion",
        "Pneumothorax",
        "Edema",
        "Hernia",
        "Atelectasis",
        "Pleural Thickening",
        "Pneumonia",
        "Cardiomegaly",
        "Nodule",
    ]

    return LABELS[dataset_name]


def multilabelConfussionMatrix(y_test, predictions):

    TP = np.zeros(y_test.shape[1])
    FP = np.zeros(y_test.shape[1])
    TN = np.zeros(y_test.shape[1])
    FN = np.zeros(y_test.shape[1])

    for j in range(y_test.shape[1]):
        TPaux = 0
        FPaux = 0
        TNaux = 0
        FNaux = 0
        for i in range(y_test.shape[0]):
            if int(y_test[i, j]) == 1:
                if int(y_test[i, j]) == 1 and int(predictions[i, j]) == 1:
                    TPaux += 1
                else:
                    FPaux += 1
            else:
                if int(y_test[i, j]) == 0 and int(predictions[i, j]) == 0:
                    TNaux += 1
                else:
                    FNaux += 1
        TP[j] = TPaux
        FP[j] = FPaux
        TN[j] = TNaux
        FN[j] = FNaux

    return TP, FP, TN, FN


def multilabelMicroConfussionMatrix(TP, FP, TN, FN):
    TPMicro = 0.0
    FPMicro = 0.0
    TNMicro = 0.0
    FNMicro = 0.0

    for i in range(len(TP)):
        TPMicro = TPMicro + TP[i]
        FPMicro = FPMicro + FP[i]
        TNMicro = TNMicro + TN[i]
        FNMicro = FNMicro + FN[i]

    return TPMicro, FPMicro, TNMicro, FNMicro


def accuracyMacro(y_test, predictions):

    accuracymacro = 0.0
    accurancy_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        acc_class = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i])
        accuracymacro += acc_class
        accurancy_list.append(acc_class)

    accuracymacro = np.mean(accurancy_list)

    return accuracymacro, accurancy_list


def precisionMacro(y_test, predictions):

    precisionmacro = 0.0
    precision_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        if not np.all(y_test[:, i] == 0):
            pcs_class = 0.0
            if TP[i] + FP[i] != 0:
                pcs_class = TP[i] / (TP[i] + FP[i])
                precisionmacro = precisionmacro + pcs_class
            precision_list.append(pcs_class)

    precisionmacro = np.mean(precision_list)

    return precisionmacro, precision_list


def precisionMicro(y_test, predictions):

    precisionmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FPMicro) != 0:
        precisionmicro = float(TPMicro / (TPMicro + FPMicro))

    return precisionmicro


def recallMacro(y_test, predictions):

    recallmacro = 0.0
    recall_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)

    for i in range(len(TP)):
        if not np.all(y_test[:, i] == 0):
            rc_class = 0.0
            if TP[i] + FN[i] != 0:
                rc_class = TP[i] / (TP[i] + FN[i])
                recallmacro = recallmacro + rc_class
            recall_list.append(rc_class)

    recallmacro = np.mean(recall_list)

    return recallmacro, recall_list


def recallMicro(y_test, predictions):
    recallmicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    if (TPMicro + FNMicro) != 0:
        recallmicro = float(TPMicro / (TPMicro + FNMicro))

    return recallmicro


def fbetaMacro(y_test, predictions, beta=1):
    fbetamacro = 0.0

    fbeta_list = []
    thres_list = []
    for i in range(len(LABELS)):
        if not np.all(y_test[:, i] == 0):
            fbeta_class = 0.0
            precision, recall, thresholds = precision_recall_curve(
                y_test[:, i], predictions[:, i]
            )
            f1 = []
            for pcs, rc in zip(precision, recall):
                if pcs + rc != 0:
                    f1.append(2 * (pcs * rc) / (pcs + rc))
                else:
                    f1.append(0)
            f1 = np.array(f1)
            best_threshold_index = np.argmax(f1)
            best_threshold = thresholds[best_threshold_index]
            fbeta_class = f1[best_threshold_index]
            thres_list.append(best_threshold)
            fbeta_list.append(fbeta_class)
        else:
            thres_list.append(THRESHOLD)

    fbetamacro = np.mean(fbeta_list)
    return fbetamacro, fbeta_list, thres_list


def fbetaMicro(y_test, predictions, beta=1):
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1 + pow(beta, 2)) * TPMicro)
    den = float((1 + pow(beta, 2)) * TPMicro + pow(beta, 2) * FNMicro + FPMicro)
    fbetamicro = float(num / den)

    return fbetamicro


def MCC_Macro(ground_truth, predictions):

    mcc_scores = [
        matthews_corrcoef(ground_truth[:, i], predictions[:, i])
        for i in range(len(LABELS))
    ]
    average_mcc = np.mean(mcc_scores)

    return average_mcc, mcc_scores


def calc_ci(data):
    lower_ci = np.percentile(data, 2.5)
    upper_ci = np.percentile(data, 97.5)
    mean_ci = np.mean(data)
    return lower_ci, upper_ci, mean_ci


def calc_roc(ground_truth, predictions):
    # Micro.
    fpr, tpr, _ = roc_curve(ground_truth.ravel(), predictions.ravel())
    micro_roc_auc = auc(fpr, tpr)
    # Macro.
    roc_per_class = []
    macro_roc_auc = 0

    for i in range(len(LABELS)):
        if not np.all(ground_truth[:, i] == 0):
            fpr, tpr, _ = roc_curve(ground_truth[:, i], predictions[:, i])
            new_auc = auc(fpr, tpr)
            roc_per_class.append(new_auc)

    macro_roc_auc = np.mean(roc_per_class)

    return macro_roc_auc, micro_roc_auc, roc_per_class


def calc_ap(ground_truth, predictions):
    precision, recall, thresholds = precision_recall_curve(ground_truth, predictions)
    return auc(recall, precision)


def calc_map(ground_truth, predictions):
    num_classes = len(LABELS)
    ap_list = []
    for i in range(num_classes):
        gt_class = ground_truth[:, i]
        pred_class = predictions[:, i]
        if not np.all(gt_class == 0):
            ap = calc_ap(gt_class, pred_class)
            ap_list.append(ap)

    map_score = np.mean(ap_list)

    return map_score, ap_list


def threshold_to_binary(y_pred, threshold):
    binary_pred = (y_pred > threshold).astype(int)
    return binary_pred


def output_stats(
    ground_truth,
    labels,
    predictions,
    filename,
    df,
    df_ci,
):
    ma_AUROC_list = []
    mi_AUROC_list = []
    mAP_list = []
    length = ground_truth.shape[0]
    for i in tqdm(range(1000)):
        randnum = random.randint(0, 1000)
        random.seed(randnum)
        gt_i = random.choices(ground_truth, k=length)
        random.seed(randnum)
        pred_i = random.choices(predictions, k=length)
        gt_i = np.array(gt_i)
        pred_i = np.array(pred_i)

        macro_roc_auc, micro_roc_auc, _ = calc_roc(gt_i, pred_i)
        ma_AUROC_list.append(macro_roc_auc)
        mi_AUROC_list.append(micro_roc_auc)

        map_score, _ = calc_map(gt_i, pred_i)
        mAP_list.append(map_score)

    lower_ma_auc, higher_ma_auc, macro_roc_auc = calc_ci(np.array(ma_AUROC_list))
    lower_mi_auc, higher_mi_auc, micro_roc_auc = calc_ci(np.array(mi_AUROC_list))
    lower_map, higher_map, map_score = calc_ci(np.array(mAP_list))

    data = {
        "File Name": [filename],
        "AUROC Macro": [macro_roc_auc],
        "AUROC Micro": [micro_roc_auc],
        "mAP": [map_score],
    }

    df = df._append(pd.DataFrame(data))

    data_ci = {
        "File Name": [filename],
        "AUROC Macro": [macro_roc_auc],
        "AUROC Macro 2.5": [lower_ma_auc],
        "AUROC Macro 97.5": [higher_ma_auc],
        "AUROC Micro": [micro_roc_auc],
        "AUROC Micro 2.5": [lower_mi_auc],
        "AUROC Micro 97.5": [higher_mi_auc],
        "mAP": [map_score],
        "mAP 2.5": [lower_map],
        "mAP 97.5": [higher_map],
    }
    df_ci = df_ci._append(pd.DataFrame(data_ci))

    print(f"AUROC Macro:{macro_roc_auc:.3f}({lower_ma_auc:.3f},{higher_ma_auc:.3f})")

    print(f"AUROC Micro:{micro_roc_auc:.3f}({lower_mi_auc:.3f},{higher_mi_auc:.3f})")

    print("-" * 80)

    print(f"mAP:{map_score:.3f}({lower_map:.3f},{higher_map:.3f})")

    print("-" * 80)

    return (
        df,
        df_ci,
    )


def indices_not_in(list1, list2):
    list1_lower = [item.lower() for item in list1]
    list2_lower = [item.lower() for item in list2]

    result = [(i, val) for i, val in enumerate(list1_lower) if val not in list2_lower]

    indices = [idx for idx, _ in result]

    return indices


def zsl_filter(np_array, filtered_indices):
    filtered = []
    for row, indices in zip(np_array, filtered_indices):
        filtered.append(row[indices])
    return np.array(filtered)


if __name__ == "__main__":
    LABELS = get_labels("Vindr")
    PREFIX = "pspg/vindr-2"
    zsl = False
    indices = None

    df = pd.DataFrame()
    df_ci = pd.DataFrame()

    if zsl:
        label_last = get_labels("Chexpert")
        indices = indices_not_in(LABELS, label_last)
        LABELS = LABELS[indices]

    for i in glob(os.path.join(PREFIX, "*.json")):
        filename = i.split("\\")[-1]
        print(i.split("\\")[-1])
        print("-" * 80)

        with open(i, "r") as f:
            # results = [json.loads(line) for line in f]
            results = json.load(f)
        predictions = np.array([i["probabilities"] for i in results])
        labels = np.array([i["labels"] for i in results])
        ground_truth = np.array([i["ground_truth"] for i in results])
        if zsl:
            predictions = zsl_filter(predictions, indices)
            labels = zsl_filter(labels, indices)
            ground_truth = zsl_filter(ground_truth, indices)
        (
            df,
            df_ci,
        ) = output_stats(
            ground_truth,
            labels,
            predictions,
            filename,
            df,
            df_ci,
        )
        print("=" * 80)
        info = ""
        if zsl:
            info = "_zsl"
        df.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + info + "_global2_stats.csv"),
            index=False,
        )
        df_ci.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + info + "_global2_ci_stats.csv"),
            index=False,
        )
