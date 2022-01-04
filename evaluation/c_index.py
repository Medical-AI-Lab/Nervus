#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import dataclasses

from lifelines.utils import concordance_index

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
c_index_dir = dirs_dict['c_index']
c_index_summary_dir = dirs_dict['c_index_summary']
path_likelihood = get_target(dirs_dict['likelihood'], args['likelihood_datetime'])
df_likelihood = pd.read_csv(path_likelihood)
label_name = [column_name for column_name in df_likelihood.columns if column_name.startswith('label_')][0]
periods_column_name = [column_name for column_name in df_likelihood.columns if column_name.startswith('periods_')][0]

@dataclasses.dataclass
class LabelCIndex:
    val: float
    test: float

def cal_c_index(df_likelihood, label_name, periods_column_name):
    pred_column_name = 'pred_' + label_name.replace('label_', '')
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        periods_split = df_likelihood_split[periods_column_name].values
        preds_split = df_likelihood_split[pred_column_name].values
        labels_split = df_likelihood_split[label_name].values
        c_index_split = concordance_index(periods_split, (-1)*preds_split, labels_split)
        if split == 'val':
            c_index_val = c_index_split
        else:
            c_index_test = c_index_split
    label_c_index = LabelCIndex(val=c_index_val, test=c_index_test)
    print(f"{label_name}, val: {label_c_index.val:.2f}, test: {label_c_index.test:.2f}")
    return label_c_index

label_c_index = cal_c_index(df_likelihood, label_name, periods_column_name)


# Save c-index
basename = get_basename(path_likelihood)
date_name = basename.rsplit('_', 1)[-1]
df_summary_new = pd.DataFrame({'datetime': [date_name],
                               label_name+'_val_c_index': [f"{label_c_index.val:.2f}"],
                               label_name+'_test_c_index': [f"{label_c_index.test:.2f}"]})

summary_path = os.path.join(c_index_summary_dir, 'summary.csv')   # Previous summary
if os.path.isfile(summary_path):
    df_summary = pd.read_csv(summary_path, dtype=str)
    df_summary = pd.concat([df_summary, df_summary_new], axis=0)
else:
    df_summary = df_summary_new
os.makedirs(c_index_summary_dir, exist_ok=True)
c_index_summary_path = os.path.join(c_index_summary_dir, 'summary.csv')
df_summary.to_csv(c_index_summary_path, index=False)


# ----- EOF -----
