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
    dataframes_acc,
    dataframes_pcs,
    dataframes_rc,
    dataframes_f1,
    dataframes_auroc,
    dataframes_ap,
):
    gt_i = ground_truth
    pred_i = predictions
    gt_i = np.array(gt_i)
    pred_i = np.array(pred_i)

    macro_roc_auc, micro_roc_auc, classes_auc = calc_roc(gt_i, pred_i)

    map_score, classes_ap = calc_map(gt_i, pred_i)

    macro_f1, classes_f1, threslist = fbetaMacro(gt_i, pred_i)

    thresholded = []
    num_samples = pred_i.shape[0]
    for i in range(num_samples):
        thresholded.append(threshold_to_binary(pred_i[i, :], threslist))
    thresholded = np.array(thresholded)

    acc, classes_acc = accuracyMacro(ground_truth, thresholded)

    macro_pcs, classes_pcs = precisionMacro(ground_truth, thresholded)

    micro_pcs = precisionMicro(ground_truth, thresholded)

    macro_rc, classes_rc = recallMacro(ground_truth, thresholded)

    micro_rc = recallMicro(ground_truth, thresholded)

    micro_f1 = fbetaMicro(ground_truth, thresholded)

    mcc_score, classes_mcc = MCC_Macro(ground_truth, thresholded)

    data = {
        "File Name": [filename],
        "Accuracy": [acc],
        "Precision Macro": [macro_pcs],
        "Precision Micro": [micro_pcs],
        "Recall Macro": [macro_rc],
        "Recall Micro": [micro_rc],
        "F1 Macro": [macro_f1],
        "F1 Micro": [micro_f1],
        "AUROC Macro": [macro_roc_auc],
        "AUROC Micro": [micro_roc_auc],
        "mAP": [map_score],
        "MCC": [mcc_score],
    }
    data_acc = {"File Name": [filename]}

    data_pcs = {"File Name": [filename]}

    data_rc = {"File Name": [filename]}

    data_f1 = {"File Name": [filename]}

    data_auc = {"File Name": [filename]}

    data_ap = {"File Name": [filename]}

    df = df._append(pd.DataFrame(data))

    print(f"Accuracy:{acc:.3f}")
    for i in range(len(LABELS)):

        print(f"ACC_{LABELS[i]}: {np.round(classes_acc[i], 3)}")

        data_acc[f"ACC_{LABELS[i]}"] = [classes_acc[i]]

    print("-" * 80)

    print(f"Precision Macro:{macro_pcs:.3f}")

    print(f"Precision Micro:{micro_pcs:.3f}")
    for i in range(len(LABELS)):

        print(f"Precision_{LABELS[i]}: {np.round(classes_pcs[i], 3)}")

        data_pcs[f"Precision_{LABELS[i]}"] = [classes_pcs[i]]

    print("-" * 80)

    print(f"Recall Macro:{macro_rc:.3f}")

    print(f"Recall Micro:{micro_rc:.3f}")
    for i in range(len(LABELS)):

        print(f"Recall_{LABELS[i]}: {np.round(classes_rc[i], 3)}")

        data_rc[f"Recall_{LABELS[i]}"] = [classes_rc[i]]

    print("-" * 80)

    print(f"F1 Macro:{macro_f1:.3f}")

    print(f"F1 Micro:{micro_f1:.3f}")
    for i in range(len(LABELS)):

        print(f"F1_{LABELS[i]}: {np.round(classes_f1[i], 3)}")

        data_f1[f"F1_{LABELS[i]}"] = [classes_f1[i]]

    print("-" * 80)

    print(f"AUROC Macro:{macro_roc_auc:.3f}")

    print(f"AUROC Micro:{micro_roc_auc:.3f}")
    for i in range(len(LABELS)):

        print(f"AUROC_{LABELS[i]}: {np.round(classes_auc[i], 3)}")

        data_auc[f"AUROC_{LABELS[i]}"] = [classes_auc[i]]

    print("-" * 80)

    print(f"mAP:{map_score:.3f}")

    for i in range(len(LABELS)):
        print(f"AP_{LABELS[i]}: {np.round(classes_ap[i], 3)}")
        data_ap[f"AP_{LABELS[i]}"] = [classes_ap[i]]

    print("-" * 80)

    print(f"MCC:{mcc_score:.3f}")

    dataframes_acc = dataframes_acc._append(pd.DataFrame(data_acc))

    dataframes_pcs = dataframes_pcs._append(pd.DataFrame(data_pcs))

    dataframes_rc = dataframes_rc._append(pd.DataFrame(data_rc))

    dataframes_f1 = dataframes_f1._append(pd.DataFrame(data_f1))

    dataframes_auroc = dataframes_auroc._append(pd.DataFrame(data_auc))

    dataframes_ap = dataframes_ap._append(pd.DataFrame(data_ap))

    return (
        df,
        dataframes_acc,
        dataframes_pcs,
        dataframes_rc,
        dataframes_f1,
        dataframes_auroc,
        dataframes_ap,
    )


def indices_not_in(list1, list2):
    list1_lower = [item.lower() for item in list1]
    list2_lower = [item.lower() for item in list2]

    result = [(i, val) for i, val in enumerate(list1_lower) if val not in list2_lower]

    indices = [idx for idx, _ in result]

    return indices


def zsl_filter(np_array, filtered_indices):
    filtered = []
    for row in np_array:
        row = [row[i] for i in filtered_indices]
        filtered.append(row)
    return np.array(filtered)


if __name__ == "__main__":
    LABELS = get_labels("Chexpert")
    PREFIX = "G:/pspg/output/test"
    zsl = False
    indices = None

    df = pd.DataFrame()
    dataframes_acc = pd.DataFrame()
    dataframes_pcs = pd.DataFrame()
    dataframes_rc = pd.DataFrame()
    dataframes_f1 = pd.DataFrame()
    dataframes_auroc = pd.DataFrame()
    dataframes_ap = pd.DataFrame()

    if zsl:
        label_last = get_labels("Chexpert")
        indices = indices_not_in(LABELS, label_last)
        LABELS = [LABELS[i] for i in indices]

    for i in tqdm(glob(os.path.join(PREFIX, "*.json"))):
        filename = i.split("\\")[-1]
        print(i.split("\\")[-1])
        print("-" * 80)

        with open(i, "r") as f:
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
            dataframes_acc,
            dataframes_pcs,
            dataframes_rc,
            dataframes_f1,
            dataframes_auroc,
            dataframes_ap,
        ) = output_stats(
            ground_truth,
            labels,
            predictions,
            filename,
            df,
            dataframes_acc,
            dataframes_pcs,
            dataframes_rc,
            dataframes_f1,
            dataframes_auroc,
            dataframes_ap,
        )
        print("=" * 80)
        info = ""
        if zsl:
            info = "_zsl"
        df.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + info + "_new_global_stats.csv"),
            index=False,
        )
        dataframes_acc.to_csv(
            os.path.join(
                PREFIX, PREFIX.split("/")[0] + info + "_local_accurancy_stats.csv"
            ),
            index=False,
        )
        dataframes_pcs.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + "_local_precision_stats.csv"),
            index=False,
        )
        dataframes_rc.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + "_local_recall_stats.csv"),
            index=False,
        )
        dataframes_f1.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + "_local_f1_stats.csv"),
            index=False,
        )
        dataframes_auroc.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + "_local_auroc_stats.csv"),
            index=False,
        )
        dataframes_ap.to_csv(
            os.path.join(PREFIX, PREFIX.split("/")[0] + "_local_ap_stats.csv"),
            index=False,
        )
