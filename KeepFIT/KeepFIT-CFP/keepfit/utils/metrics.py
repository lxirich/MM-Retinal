"""
下游任务迁移的评价指标
管理评价指标，k折交叉验证，结果保存
Evaluation metric of downstream task migration
Management evaluation metric, K-fold cross-validation, results saving
"""

import os
import numpy as np
import json

from sklearn.metrics import precision_recall_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score, f1_score, recall_score


# 主控函数调用接口  main function call interface
def evaluate(refs, preds, task="classification"):
    if task == "classification":
        metrics = classification_metrics(refs, preds)
        print('Metrics: aca=%2.5f - TAOP_aca=%2.5f - kappa=%2.3f - AMD_kappa=%2.3f - macro f1=%2.3f - AMD_F1=%2.3f - auc=%2.3f -AMD_auc=%2.3f'
              % (metrics["aca"], metrics["TAOP_acc"],metrics["kappa"],metrics["AMD_kappa"],metrics["f1_avg"],metrics["AMD_f1"],metrics["auc_avg"],metrics["AMD_auc"]))
    elif task == "segmentation":
        metrics = segmentation_metrics(refs, preds)
        print('Metrics: dsc=%2.5f - auprc=%2.3f' % (metrics["dsc"], metrics["auprc"]))
    else:
        metrics = {}
    return metrics


# auc
def au_prc(true_mask, pred_mask):
    precision, recall, threshold = precision_recall_curve(true_mask, pred_mask)
    au_prc = auc(recall, precision)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    f1[np.isnan(f1)] = 0
    th = threshold[np.argmax(f1)]                                                   

    return au_prc, th


# 特异性计算  specificity
def specificity(refs, preds):
    cm = confusion_matrix(refs, preds)
    specificity = cm[0,0]/(cm[0,0]+cm[0,1])
    return specificity


# 分类评价指标  Classification evaluation metric
def classification_metrics(refs, preds):
    k = np.round(cohen_kappa_score(refs, np.argmax(preds, -1), weights="quadratic"), 3)                             # Kappa quadatic

    cm = confusion_matrix(refs, np.argmax(preds, -1))                                                                       # 混淆矩阵  confusion matrix
    cm_norm = (cm / np.expand_dims(np.sum(cm, -1), 1))

    acc_class = list(np.round(np.diag(cm_norm), 3))
    aca = np.round(np.mean(np.diag(cm_norm)), 3)

    recall_class = [np.round(recall_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]       # 每个类别的召回率  recall per class

    specificity_class = [np.round(specificity(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]   # 每个类的特异性 specificity

    auc_class = [np.round(roc_auc_score(refs == i, preds[:, i]), 3) for i in np.unique(refs)]                       # 每个类的auc auc

    f1_class = [np.round(f1_score(refs == i, np.argmax(preds, -1) == i), 3) for i in np.unique(refs)]               # 每个类的f1  f1

    # AMD dataset
    AMD_auc = roc_auc_score(refs, preds[:,1])
    precision, recall, thresholds = precision_recall_curve(refs, preds[:,1])
    kappa_list = []
    f1_list = []
    for threshold in thresholds:
        y_scores = preds[:,1]
        y_scores = np.array(y_scores >= threshold, dtype=float)
        kappa = cohen_kappa_score(refs, y_scores)
        kappa_list.append(kappa)
        f1 = f1_score(refs, y_scores)
        f1_list.append(f1)
    kappa_f1 = np.array(kappa_list) + np.array(f1_list)
    AMD_kappa = kappa_list[np.argmax(kappa_f1)]
    AMD_f1 = f1_list[np.argmax(kappa_f1)]

    # TAOP dataset
    class_acc = accuracy_score(refs, np.argmax(preds, -1))

    metrics = {"aca": aca , "TAOP_acc":class_acc,"acc_class": acc_class,
               "kappa": k, "AMD_kappa":AMD_kappa,
               "auc_class": auc_class, "auc_avg": np.mean(auc_class), "AMD_auc":AMD_auc,
               "f1_class": f1_class, "f1_avg": np.mean(f1_class), "AMD_f1":AMD_f1,
               "sensitivity_class": recall_class, "sensitivity_avg": np.mean(recall_class),
               "specificity_class": specificity_class, "specificity_avg": np.mean(specificity_class),
               "cm": cm, "cm_norm": cm_norm}
    return metrics


# K折交叉验证平均  K-fold cross-verify average
def average_folds_results(list_folds_results, task):
    '''
    list_folds_results：存放了K折交叉验证的结果
    '''
    metrics_name = list(list_folds_results[0].keys())               # 全部评价指标的名称  Name of all evaluation metric

    out = {}
    for iMetric in metrics_name:
        values = np.concatenate([np.expand_dims(np.array(iFold[iMetric]), -1) for iFold in list_folds_results], -1)
        out[(iMetric + "_avg")] = np.round(np.mean(values, -1), 3).tolist()
        out[(iMetric + "_std")] = np.round(np.std(values, -1), 3).tolist()

    if task == "classification":
        print('Metrics: aca=%2.3f(%2.3f) - TAOP_aca=%2.3f(%2.3f) - kappa=%2.3f(%2.3f) - macro f1=%2.3f(%2.3f) -'
              'AMD_kappa=%2.3f(%2.3f) - AMD_auc=%2.3f(%2.3f) - AMD_f1=%2.3f(%2.3f)' % (
            out["aca_avg"], out["aca_std"], out["TAOP_acc_avg"], out["TAOP_acc_std"],
            out["kappa_avg"], out["kappa_std"], out["f1_avg_avg"], out["f1_avg_std"],
            out["AMD_kappa_avg"], out["AMD_kappa_std"], out["AMD_auc_avg"], out["AMD_auc_std"],
            out["AMD_f1_avg"], out["AMD_f1_std"]))

    return out


# 保存实验结果 适配器权重  Save experiment results and adapter weights
def save_results(metrics, out_path, id_experiment=None, id_metrics=None, save_model=False, weights=None):
    '''
    metrics：K折交叉验证的结果  Results of K-fold cross-validation
    id_experiment：实验id
    '''
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    if id_experiment is None:
        id_experiment = "experiment" + str(np.random.rand())
    else:
        id_experiment = id_experiment
    if not os.path.isdir(out_path + id_experiment):
        os.mkdir(out_path + id_experiment)

    # 以json格式保存结果  Save the results in json format
    with open(out_path + id_experiment + '/metrics_' + id_metrics + '.json', 'w') as fp:
        json.dump(metrics, fp)

    # 保存适配器权重  Save adapter weight
    if save_model:
        import torch
        for i in range(len(weights)):
            torch.save(weights[i], out_path + id_experiment + '/weights_' + str(i) + '.pth')