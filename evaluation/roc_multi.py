#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
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

path_likelihood = get_target(dirs_dict['likelihood'], args['likelihood_datetime'])   # the latest likelihod if datatime is None
df_likelihood = pd.read_csv(path_likelihood)
label_list = [ column_name for column_name in df_likelihood.columns if column_name.startswith('label_') ]



def calc_roc_each(label_name):
    metrics_each = { 'val': {}, 'test':{} }
    pred_p_label_name = 'pred_p_' + label_name.replace('label_', '')

    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])

        y_true_split = df_likelihood_split[label_name]
        y_score_split = df_likelihood_split[pred_p_label_name]

        # Calcurate AUC
        fpr_split, tpr_split, thresholds_split = metrics.roc_curve(y_true_split, y_score_split)
        auc_split = metrics.auc(fpr_split, tpr_split)

        metrics_each[split]['fpr'] = fpr_split
        metrics_each[split]['tpr'] = tpr_split
        metrics_each[split]['auc'] = auc_split
    
    print( '{label}, val: {auc_val:.2f}, test: {auc_test:.2f}'.format(label=label_name,
                                                                      auc_val=metrics_each['val']['auc'], 
                                                                      auc_test=metrics_each['test']['auc']) )
    return  metrics_each



metrics_multi = { label_name: calc_roc_each(label_name) for label_name in label_list }


# Plot ROC
num_rows = 1
num_cols = len(label_list)

base_size = 7
height = 1 * base_size
width = num_cols * height 

fig = plt.figure(figsize=(width, height))


for i in range(len(label_list)):
    offset = i + 1
    label_name = label_list[i]
    metrics_each = metrics_multi[label_name]
    
    ax_each = fig.add_subplot(num_rows,
                              num_cols,
                              offset,
                              title=label_list[i],
                              xlabel='1 - Specificity',
                              ylabel='Sensitivity',
                              xmargin=0,
                              ymargin=0
                             )

    ax_each.plot(metrics_each['val']['fpr'],
                 metrics_each['val']['tpr'],
                 label='AUC_val = %.2f'%metrics_each['val']['auc'],
                 marker='x'
                )

    ax_each.plot(metrics_each['test']['fpr'],
                 metrics_each['test']['tpr'],
                 label='AUC_test = %.2f'%metrics_each['test']['auc'],
                 marker='o'
                )
    
    ax_each.grid()
    ax_each.legend()
    

# Align subplots
fig.tight_layout()


# Save Fig
os.makedirs(roc_dir, exist_ok=True)
basename = get_basename(path_likelihood)
roc_path = os.path.join(roc_dir, 'roc_' + basename) + '.png'
plt.savefig(roc_path)


# ----- EOF -----
