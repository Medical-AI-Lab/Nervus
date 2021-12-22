#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import dataclasses

from sklearn import metrics
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
df_likelihood = pd.read_csv(path_likelihood)
label_list = [column_name for column_name in df_likelihood.columns if column_name.startswith('label_')]

@dataclasses.dataclass
class ROC:
    fpr: list
    tpr: list
    auc: float

@dataclasses.dataclass
class LabelROC:
    val: ROC
    test: ROC

def cal_label_roc(label_name, df_likelihood):
    pred_p_label_name = 'pred_p_' + label_name.replace('label_', '')
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        y_true_split = df_likelihood_split[label_name]
        y_score_split = df_likelihood_split[pred_p_label_name]
        fpr_split, tpr_split, thresholds_split = metrics.roc_curve(y_true_split, y_score_split)
        auc_split = metrics.auc(fpr_split, tpr_split)
        if split == 'val':
            label_roc_val = ROC(fpr=fpr_split, tpr=tpr_split, auc=auc_split)
        else:
            label_roc_test = ROC(fpr=fpr_split, tpr=tpr_split, auc=auc_split)
    label_roc = LabelROC(val=label_roc_val, test=label_roc_test)
    print(f"{label_name}, val: {label_roc.val.auc:.2f}, test: {label_roc.test.auc:.2f}")
    return label_roc

metrics_roc = {label_name: cal_label_roc(label_name, df_likelihood) for label_name in label_list}


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
    
    ax_i = fig.add_subplot(num_rows,
                           num_cols,
                           offset,
                           title=label_list[i],
                           xlabel='1 - Specificity',
                           ylabel='Sensitivity',
                           xmargin=0,
                           ymargin=0)
    ax_i.plot(label_roc.val.fpr,
              label_roc.val.tpr,
              label=f"AUC_val = {label_roc.val.auc:.2f}",
              marker='x')
    ax_i.plot(label_roc.test.fpr,
              label_roc.test.tpr,
              label=f"AUC_test = {label_roc.test.auc:.2f}",
              marker='o')
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
#df_summary_new = pd.DataFrame({})
date_name = basename.rsplit('_', 1)[-1]
summary_new = {'datetime': [date_name]}
for label_name in metrics_roc.keys():
    summary_new[label_name+'_val_auc'] = [f"{metrics_roc[label_name].val.auc:.2f}"]
    summary_new[label_name+'_test_auc'] = [f"{metrics_roc[label_name].test.auc:.2f}"]
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
