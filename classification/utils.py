import os, torch 
import numpy as np
from sklearn.metrics import confusion_matrix

def cls_score(pred, label):
    """
    pred:  [[1, 0], [0, 1]]
    label: [[0, 0, 1, 1], [1, 1, 1, 1]]
    """

    # case level
    # case_pred = np.array([np.any(item) for item in pred], dtype=int)
    # case_true = np.array([np.any(item) for item in label], dtype=int)
    tn, fp, fn, tp = confusion_matrix(np.array(pred), np.array(label)).ravel()
    # print(case_pred, case_true)
    # case_acc = np.sum(case_pred == case_true) / (len(case_true) + 1e-9)
    case_sen = tp / (tp + fn + 1e-9)
    case_prec = tp / (tp + fp + 1e-9)
    case_f1  = (2*case_prec*case_sen) / (case_prec + case_sen + 1e-9)
    case_acc = 0

    # # organ level
    # organ_pred = np.array(pred).ravel()
    # organ_true = np.array(label).ravel()
    # tn, fp, fn, tp = confusion_matrix(organ_true, organ_pred).ravel()
    
    # organ_acc = np.sum(organ_pred == organ_true) / (len(organ_true) + 1e-9)
    # organ_sen = tp / (tp + fn + 1e-9)
    # organ_prec = tp / (tp + fp + 1e-9)
    # organ_f1   = (2*organ_prec*organ_sen) / (organ_prec + organ_sen + 1e-9)


    score_table = {
        "case_acc": case_acc,
        "case_sensitive": case_sen,
        "case_precision": case_prec,
        "case_f1": case_f1
    }
    score = 0.3 * case_sen + 0.2 * case_f1 + 0.1 * case_acc

    return score, score_table