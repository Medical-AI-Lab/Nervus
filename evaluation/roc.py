#!/usr/bin/env python
# -*- coding: utf-8 -*-

from operator import ne
import os
import sys
import numpy as np
import pandas as pd
import dataclasses

from sklearn import metrics
from sklearn.preprocessing import label_binarize

import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import *
from options import MetricsOptions

logger = NervusLogger.get_logger('evaluation.roc')

nervusenv = NervusEnv()
args = MetricsOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['likelihood_datetime'])   # args['likelihood_datetime'] if exists or the latest
likelihood_path = os.path.join(datetime_dir, nervusenv.csv_likelihood)
df_likelihood = pd.read_csv(likelihood_path)


@dataclasses.dataclass
class AverageType:
    micro: np.ndarray = None   # Ordinary ROC when binary classification
    macro: np.ndarray = None   # macro-average ROC when multiclass(>2)

@dataclasses.dataclass
class ROC:
    fpr: AverageType
    tpr: AverageType
    auc: AverageType

@dataclasses.dataclass
class LabelROC:
    val: ROC
    test: ROC


# ROC for single-output-binary-class
def cal_roc_binary_class(label_name, df_likelihood):
    label_roc = LabelROC(val=None, test=None)
    POSITIVE = 1
    pred_name_list = list(df_likelihood.columns[df_likelihood.columns.str.startswith('pred')])
    pred_positive_name = pred_name_list[POSITIVE]
    class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
    positive_class = class_list[POSITIVE]

    for split in ['val', 'test']:
        df_likelihood_split = df_likelihood[df_likelihood['split']==split]
        y_true_split = df_likelihood_split[label_name]
        y_score_split = df_likelihood_split[pred_positive_name]
        fpr_split, tpr_split, thresholds_split = metrics.roc_curve(y_true_split.astype('str'), y_score_split, pos_label=positive_class)
        if split == 'val':
            label_roc.val = ROC(fpr=AverageType(micro=fpr_split),
                                 tpr=AverageType(micro=tpr_split),
                                 auc=AverageType(micro=metrics.auc(fpr_split, tpr_split)))
        else:
            label_roc.test = ROC(fpr=AverageType(micro=fpr_split),
                                  tpr=AverageType(micro=tpr_split),
                                  auc=AverageType(micro=metrics.auc(fpr_split, tpr_split)))
    logger.info(f"{label_name}, val: {label_roc.val.auc.micro:.2f}, test: {label_roc.test.auc.micro:.2f}")
    return label_roc


# Macro-average ROC for single-output-multi-class
def cal_roc_multi_class(label_name, df_likelihood):
    pred_name_list = list(df_likelihood.columns[df_likelihood.columns.str.startswith('pred')])
    class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
    num_classes = len(class_list)

    for split in ['val', 'test']:
        df_likelihood_split = df_likelihood[df_likelihood['split']==split]
        y_true_split = df_likelihood_split[label_name]
        df_y_score_split = df_likelihood_split[pred_name_list]
        #auc_one_vs_rest = metrics.roc_auc_score(y_true_binarized, df_y_score_split, multi_class='ovr', average='macro')   # OneVsRest
        #auc_one_vs_one = metrics.roc_auc_score(y_true_binarized, df_y_score_split, multi_class='ovo', average='macro')    # OneVsOne
        y_true_binarized = label_binarize(y_true_split, classes=class_list)

        label_fpr_split = {}
        label_tpr_split = {}
        #label_roc_auc = {}
        for i in range(num_classes):
            class_name = class_list[i]
            pred_name = 'pred_' + label_name + '_' + class_name
            label_fpr_split[class_name], label_tpr_split[class_name], _ = metrics.roc_curve(y_true_binarized[:, i], df_y_score_split[pred_name])
            #label_roc_auc[class_name] = metrics.auc(label_fpr[class_name], label_tpr[class_name])  # AUC for class_name

        # Compute macro-average ROC
        # Aggregate all false positive rates
        label_all_fpr_split = np.unique(np.concatenate([label_fpr_split[class_name] for class_name in class_list]))

        # Interpolate all ROC curves at this points
        label_mean_tpr_split = np.zeros_like(label_all_fpr_split)
        for class_name in class_list:
            label_mean_tpr_split += np.interp(label_all_fpr_split, label_fpr_split[class_name], label_tpr_split[class_name])

        # Average it and compute AUC
        label_mean_tpr_split /= num_classes

        if split == 'val':
            label_roc_val = ROC(fpr=AverageType(macro=label_all_fpr_split),
                                 tpr=AverageType(macro=label_mean_tpr_split),
                                 auc=AverageType(macro=metrics.auc(label_all_fpr_split, label_mean_tpr_split)))
        else:
            label_roc_test = ROC(fpr=AverageType(macro=label_all_fpr_split),
                                  tpr=AverageType(macro=label_mean_tpr_split),
                                  auc=AverageType(macro=metrics.auc(label_all_fpr_split, label_mean_tpr_split)))

    label_roc = LabelROC(val=label_roc_val, test=label_roc_test)
    logger.info(f"{label_name}, val: {label_roc.val.auc.macro:.2f}, test: {label_roc.test.auc.macro:.2f}")
    return label_roc


# Calculate ROC and AUC
label_list = list(df_likelihood.columns[df_likelihood.columns.str.startswith('label')])
metrics_roc = {}
for label_name in label_list:
    num_class_in_label = df_likelihood[label_name].nunique()
    if num_class_in_label > 2:
        metrics_roc[label_name] = cal_roc_multi_class(label_name, df_likelihood)
    else:
        metrics_roc[label_name] = cal_roc_binary_class(label_name, df_likelihood)


# Plot ROC
num_rows = 1
num_cols = len(label_list)
base_size = 7
height = num_rows * base_size
width = num_cols * height

fig = plt.figure(figsize=(width, height))
for i in range(len(label_list)):
    offset = i + 1
    label_name = label_list[i]
    label_roc = metrics_roc[label_name]
    num_class_in_label = df_likelihood[label_name].nunique()

    if num_class_in_label > 2:
        # macro-average ROC for multiclass
        fpr_val = label_roc.val.fpr.macro
        tpr_val = label_roc.val.tpr.macro
        auc_val = label_roc.val.auc.macro
        fpr_test = label_roc.test.fpr.macro
        tpr_test = label_roc.test.tpr.macro
        auc_test = label_roc.test.auc.macro
    else:
        # Ordinary ROC for binaryclass
        fpr_val = label_roc.val.fpr.micro
        tpr_val = label_roc.val.tpr.micro
        auc_val = label_roc.val.auc.micro
        fpr_test = label_roc.test.fpr.micro
        tpr_test = label_roc.test.tpr.micro
        auc_test = label_roc.test.auc.micro

    ax_i = fig.add_subplot(num_rows,
                           num_cols,
                           offset,
                           title=label_name,
                           xlabel='1 - Specificity',
                           ylabel='Sensitivity',
                           xmargin=0,
                           ymargin=0)

    ax_i.plot(fpr_val, tpr_val, label=f"AUC_val = {auc_val:.2f}", marker='x')
    ax_i.plot(fpr_test, tpr_test, label=f"AUC_test = {auc_test:.2f}", marker='o')
    ax_i.grid()
    ax_i.legend()

# Align subplots
fig.tight_layout()

# Save ROC
roc_path = os.path.join(datetime_dir, nervusenv.roc)
plt.savefig(roc_path)

# Save AUC
datetime = os.path.basename(datetime_dir)
summary_new = dict()
summary_new['datetime'] = [datetime]
for label_name, label_roc in metrics_roc.items():
    auc_val = label_roc.val.auc.micro if (label_roc.val.auc.micro is not None) else label_roc.val.auc.macro
    auc_test = label_roc.test.auc.micro if (label_roc.test.auc.micro is not None) else label_roc.test.auc.macro
    summary_new[label_name+'_val_auc'] = [f"{auc_val:.2f}"]
    summary_new[label_name+'_test_auc'] = [f"{auc_test:.2f}"]
df_summary_new = pd.DataFrame(summary_new)
update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)
