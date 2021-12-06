#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt

from lifelines.utils import concordance_index

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib.align_env import *
from options.metrics_options import MetricsOptions



args = MetricsOptions().parse()

dirs_dict = set_dirs()
likelilhood_dir = dirs_dict['likelihood']
c_index_dir = './results/c_index'


path_likelihood = get_target(dirs_dict['likelihood'], args['likelihood_datetime'])  # the latest likelihod if datatime is None
df_likelihood = pd.read_csv(path_likelihood)
label_list = [ column_name for column_name in df_likelihood.columns if column_name.startswith('label_') ]
label_name = label_list[0]
pred_column_name = 'pred_' + label_name.replace('label_', '')   # 'pred_last_status'
periods_column_name = 'periods_length_of_stay'


df_likelihood_val = get_column_value(df_likelihood, 'split', ['val'])
df_likelihood_test = get_column_value(df_likelihood, 'split', ['test'])


# Calculate c-index
periods_val = df_likelihood_val[periods_column_name].values
preds_val = df_likelihood_val[pred_column_name].values
labels_val = df_likelihood_val[label_name].values
c_index_val = concordance_index(periods_val, (-1)*preds_val, labels_val)

periods_test = df_likelihood_test[periods_column_name].values
preds_test = df_likelihood_test[pred_column_name].values
labels_test = df_likelihood_test[label_name].values
c_index_test = concordance_index(periods_test, (-1)*preds_test, labels_test)


print('val: {c_index_val:.2f}, test: {c_index_test:.2f}'.format(c_index_val=c_index_val, c_index_test=c_index_test))

df_c_index = pd.DataFrame([('val', c_index_val),
                           ('test', c_index_test)],
                           columns=['split', 'c-index'])


# Save c-index
os.makedirs(c_index_dir, exist_ok=True)
basename = get_basename(path_likelihood)
c_index_name = strcat('_', 'c-index', 'val-%.2f'%c_index_val, 'test-%.2f'%c_index_test, basename)
c_index_path = os.path.join(c_index_dir, c_index_name) + '.csv'
df_c_index.to_csv(c_index_path, index=False)


# ----- EOF -----
