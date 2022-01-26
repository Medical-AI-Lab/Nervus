#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from lib.util import *
from lib.align_env import *
from options.metrics_options import MetricsOptions


args = MetricsOptions().parse()

dirs_dict = set_dirs()
likelilhood_dir = dirs_dict['likelihood']
roc_dir = dirs_dict['roc']
roc_summary_dir = dirs_dict['roc_summary']
path_likelihood = get_target(dirs_dict['likelihood'], args['likelihood_datetime'])
# Parse csv odf likelihood
df_likelihood = pd.read_csv(path_likelihood)
output_list = [column_name for column_name in df_likelihood.columns if column_name.startswith('output')]

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
class OutputROC:
    val: ROC
    test: ROC

# ROC for single-output-binary-class
def cal_roc_binary_class(output_name, df_likelihood):
    POSITIVE = 1
    pred_name_list = [column_name for column_name in df_likelihood.columns if column_name.startswith('pred')]
    pred_positive_name = pred_name_list[POSITIVE]
    class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # ['pred_output_1_discharge', pred_output_decease] -> ['discharge', 'decease']
    positive_class = class_list[POSITIVE]
    print(pred_positive_name)
    for split in ['val', 'test']:
        df_likelihood_split = df_likelihood[df_likelihood['split']==split]
        y_true_split = df_likelihood_split[output_name]
        y_score_split = df_likelihood_split[pred_positive_name]
        fpr_split, tpr_split, thresholds_split = metrics.roc_curve(y_true_split, y_score_split, pos_label=positive_class)
        if split == 'val':
            output_roc_val = ROC(fpr=AverageType(micro=fpr_split), tpr=AverageType(micro=tpr_split), auc=AverageType(micro=metrics.auc(fpr_split, tpr_split)))
        else:
            output_roc_test = ROC(fpr=AverageType(micro=fpr_split), tpr=AverageType(micro=tpr_split), auc=AverageType(micro=metrics.auc(fpr_split, tpr_split)))
    output_roc = OutputROC(val=output_roc_val, test=output_roc_test)
    print(f"{output_name}, val: {output_roc.val.auc.micro:.2f}, test: {output_roc.test.auc.micro:.2f}")
    return output_roc


# Macro-average ROC for single-output-multi-class
def cal_roc_multi_class(output_name, df_likelihood):
    pred_name_list = [column_name for column_name in df_likelihood if column_name.startswith('pred')]
    class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # ['pred_output_1_discharge', ...] -> ['discharge', ...]
    num_classes = len(class_list)

    for split in ['val', 'test']:
        df_likelihood_split = df_likelihood[df_likelihood['split']==split]
        df_y_score_split = df_likelihood_split[pred_name_list]
        #auc_one_vs_rest = metrics.roc_auc_score(y_true_binarized, df_y_score_split, multi_class='ovr', average='macro')   # OneVsRest
        #auc_one_vs_one = metrics.roc_auc_score(y_true_binarized, df_y_score_split, multi_class='ovo', average='macro')    # OneVsOne
        y_true_binarized = label_binarize(df_likelihood_split[output_name], classes=class_list)

        output_fpr_split = {}
        output_tpr_split = {}
        #output_roc_auc = {}
        for i in range(num_classes):
            class_name = class_list[i]
            pred_name = 'pred_' + output_name + '_' + class_name
            output_fpr_split[class_name], output_tpr_split[class_name], _ = metrics.roc_curve(y_true_binarized[:, i], df_y_score_split[pred_name])
            #output_roc_auc[class_name] = metrics.auc(output_fpr[class_name], output_tpr[class_name])  # AUC for class_name

        # Compute macro-average ROC
        # Aggregate all false positive rates
        output_all_fpr_split = np.unique(np.concatenate([output_fpr_split[class_name] for class_name in class_list]))

        # Interpolate all ROC curves at this points
        output_mean_tpr_split = np.zeros_like(output_all_fpr_split)
        for class_name in class_list:
            output_mean_tpr_split += np.interp(output_all_fpr_split, output_fpr_split[class_name], output_tpr_split[class_name])

        # Average it and compute AUC
        output_mean_tpr_split /= num_classes

        if split == 'val':
            output_roc_val = ROC(fpr=AverageType(macro=output_all_fpr_split),
                                    tpr=AverageType(macro=output_mean_tpr_split),
                                    auc=AverageType(macro=metrics.auc(output_all_fpr_split, output_mean_tpr_split)))
        else:
            output_roc_test = ROC(fpr=AverageType(macro=output_all_fpr_split),
                                    tpr=AverageType(macro=output_mean_tpr_split),
                                    auc=AverageType(macro=metrics.auc(output_all_fpr_split, output_mean_tpr_split)))

    output_roc = OutputROC(val=output_roc_val, test=output_roc_test)
    print(f"{output_name}, val: {output_roc.val.auc.macro:.2f}, test: {output_roc.test.auc.macro:.2f}")
    return output_roc

metrics_roc = {}
for output_name in output_list:
    num_class_of_output = df_likelihood[output_name].nunique()
    if num_class_of_output <= 2:
        metrics_roc[output_name] = cal_roc_binary_class(output_name, df_likelihood)
    else:
        metrics_roc[output_name] = cal_roc_multi_class(output_name, df_likelihood)


# Plot ROC
num_rows = 1
num_cols = len(output_list)
base_size = 7
height = num_rows * base_size
width = num_cols * height 

fig = plt.figure(figsize=(width, height))
for i in range(len(output_list)):
    offset = i + 1
    output_name = output_list[i]
    output_roc = metrics_roc[output_name]

    num_class_of_output = df_likelihood[output_name].nunique()

    if num_class_of_output <= 2:
        # Ordinary ROC for binaryclass
        fpr_val = output_roc.val.fpr.micro
        tpr_val = output_roc.val.tpr.micro
        auc_val = output_roc.val.auc.micro
        fpr_test = output_roc.test.fpr.micro
        tpr_test = output_roc.test.tpr.micro
        auc_test = output_roc.test.auc.micro
    else:
        # macro-average ROC for multiclass
        fpr_val = output_roc.val.fpr.macro
        tpr_val = output_roc.val.tpr.macro
        auc_val = output_roc.val.auc.macro
        fpr_test = output_roc.test.fpr.macro
        tpr_test = output_roc.test.tpr.macro
        auc_test = output_roc.test.auc.macro

    ax_i = fig.add_subplot(num_rows,
                           num_cols,
                           offset,
                           title=output_name,
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

# Save Fig
os.makedirs(roc_dir, exist_ok=True)
basename = get_basename(path_likelihood)
roc_path = os.path.join(roc_dir, 'roc_' + basename) + '.png'
plt.savefig(roc_path)

# Save AUC
date_name = basename.rsplit('_', 1)[-1]
summary_new = {'datetime': [date_name]}
for output_name in metrics_roc.keys():
    summary_new[output_name+'_val_auc'] = [f"{auc_val:.2f}"]
    summary_new[output_name+'_test_auc'] = [f"{auc_test:.2f}"]
df_summary_new = pd.DataFrame(summary_new)

summary_path = os.path.join(roc_summary_dir, 'summary.csv')   # Previous summary
if os.path.isfile(summary_path):
    df_summary = pd.read_csv(summary_path, dtype=str)
    df_summary = pd.concat([df_summary, df_summary_new], axis=0)
else:
    df_summary = df_summary_new
os.makedirs(roc_summary_dir, exist_ok=True)
roc_summary_path = os.path.join(roc_summary_dir, 'summary.csv')
df_summary.to_csv(roc_summary_path, index=False)


# ----- EOF -----
