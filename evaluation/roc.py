#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.options import Options
from lib.static import Static
from lib.util import *


args_init = Options().parse()

args_min = Static(None).align()                                                             # just get 'likelihood_dir'

roc_dir = args_min['roc_dir']
path_likelihood = get_target(args_min['likelihood_dir'], args_init['likelihood_datetime'])  # the latest likelihod if datatime is None
df_likelihood = pd.read_csv(path_likelihood)
label_list = [ column_name for column_name in df_likelihood.columns if column_name.startswith('label_') ]
label_name = label_list[0]


#def plot_roc():
# Scores
df_likelihood_val = get_column_value(df_likelihood, 'split', ['val'])
y_val_score = df_likelihood_val['1']
y_val_true = df_likelihood_val[label_name]

df_likelihood_test = get_column_value(df_likelihood, 'split', ['test'])
y_test_score = df_likelihood_test['1']
y_test_true = df_likelihood_test[label_name]


# Calculate AUC
fpr_val, tpr_val, thresholds_val = metrics.roc_curve(y_val_true, y_val_score)
auc_val = metrics.auc(fpr_val, tpr_val)

fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test_true, y_test_score)
auc_test = metrics.auc(fpr_test, tpr_test)

print('val: {auc_val:.2f}, test: {auc_test:.2f}'.format(auc_val=auc_val, auc_test=auc_test))


# Plot ROC
plt.plot(fpr_val, tpr_val, label='AUC_val = %.2f'%auc_val, marker='x')
plt.plot(fpr_test, tpr_test, label='AUC_test = %.2f'%auc_test, marker='o')
plt.legend()
plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.grid()


# Save Fig
os.makedirs(roc_dir, exist_ok=True)
basename = get_basename(path_likelihood)
roc_name = strcat('_', 'roc', 'val-%.2f'%auc_val, 'test-%.2f'%auc_test, basename)
roc_path = os.path.join(roc_dir, roc_name) + '.png'
plt.savefig(roc_path)


# ----- EOF -----
