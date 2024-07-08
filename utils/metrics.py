import numpy as np
from sklearn.metrics import (
    auc,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)


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


def accuracy(y_test, predictions):
    accuracymacro = 0.0
    accurancy_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        acc_class = (TP[i] + TN[i]) / (TP[i] + FP[i] + TN[i] + FN[i])
        accuracymacro += acc_class
        accurancy_list.append(acc_class)

    accuracymacro = float(accuracymacro / len(TP))

    return accuracymacro, accurancy_list


def precisionMacro(y_test, predictions):
    precisionmacro = 0.0
    precision_list = []
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        pcs_class = 0.0
        if TP[i] + FP[i] != 0:
            pcs_class = TP[i] / (TP[i] + FP[i])
            precisionmacro = precisionmacro + pcs_class
        precision_list.append(pcs_class)

    precisionmacro = float(precisionmacro / len(TP))
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
        rc_class = 0.0
        if TP[i] + FN[i] != 0:
            rc_class = TP[i] / (TP[i] + FN[i])
            recallmacro = recallmacro + rc_class
        recall_list.append(rc_class)

    recallmacro = float(recallmacro / len(TP))
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
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    for i in range(len(TP)):
        num = float((1 + pow(beta, 2)) * TP[i])
        den = float((1 + pow(beta, 2)) * TP[i] + pow(beta, 2) * FN[i] + FP[i])
        fbeta_class = 0.0
        if den != 0:
            fbeta_class = num / den
            fbetamacro = fbetamacro + fbeta_class
        fbeta_list.append(fbeta_class)

    fbetamacro = fbetamacro / len(TP)
    return fbetamacro, fbeta_list


def fbetaMicro(y_test, predictions, beta=1):
    fbetamicro = 0.0
    TP, FP, TN, FN = multilabelConfussionMatrix(y_test, predictions)
    TPMicro, FPMicro, TNMicro, FNMicro = multilabelMicroConfussionMatrix(TP, FP, TN, FN)

    num = float((1 + pow(beta, 2)) * TPMicro)
    den = float((1 + pow(beta, 2)) * TPMicro + pow(beta, 2) * FNMicro + FPMicro)
    fbetamicro = float(num / den)

    return fbetamicro


def Average_MCC(ground_truth, predictions, class_num):
    mcc_scores = [
        matthews_corrcoef(ground_truth[:, i], predictions[:, i])
        for i in range(class_num)
    ]
    average_mcc = np.mean(mcc_scores)
    return average_mcc


def calc_roc(ground_truth, predictions, class_num):
    # Micro.
    fpr, tpr, _ = roc_curve(ground_truth.ravel(), predictions.ravel())
    micro_roc_auc = auc(fpr, tpr)

    # Macro.
    roc_per_class = []
    macro_roc_auc = 0
    for i in range(class_num):
        fpr, tpr, _ = roc_curve(ground_truth[:, i], predictions[:, i])
        new_auc = auc(fpr, tpr)
        macro_roc_auc += new_auc
        roc_per_class.append(new_auc)

    macro_roc_auc /= class_num

    return macro_roc_auc, micro_roc_auc, roc_per_class


def calc_ap(ground_truth, predictions):
    precision, recall, _ = precision_recall_curve(ground_truth, predictions)
    return auc(recall, precision)


def calc_map(ground_truth, predictions, class_num):
    ap_list = []

    for i in range(class_num):
        gt_class = ground_truth[:, i]
        pred_class = predictions[:, i]
        ap = calc_ap(gt_class, pred_class)
        ap_list.append(ap)

    map_score = np.mean(ap_list)
    return map_score


def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return ap.mean()
