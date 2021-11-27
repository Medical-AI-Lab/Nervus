#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from sklearn import metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib.align_env import *
from options.metrics_options import MetricsOptions



args = MetricsOptions().parse()

dirs_dict = set_dirs()
likelilhood_dir = dirs_dict['likelihood']
yy_dir = dirs_dict['yy']

path_likelihood = get_target(dirs_dict['likelihood'], args['likelihood_datetime'])  # the latest likelihod if datatime is None
df_likelihood = pd.read_csv(path_likelihood)
label_list = [ column_name for column_name in df_likelihood.columns if column_name.startswith('label_') ]



def calc_metrics_each(label_name):
    metrics_each = { 'val': {}, 'test':{} }
    pred_label_name = 'pred_' + label_name.replace('label_', '')

    print(label_name)

    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        y_obs = df_likelihood_split[label_name].values
        y_pred = df_likelihood_split[pred_label_name].values

        r2 = metrics.r2_score(y_obs, y_pred)
        mse = metrics.mean_squared_error(y_obs, y_pred)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_obs, y_pred)

        metrics_each[split]['r2'] = r2
        metrics_each[split]['mse'] = mse
        metrics_each[split]['rmse'] = rmse
        metrics_each[split]['mae'] = mae

        print ('{split:>5}, R2: {r2:>5.2f}, MSE: {mse:.2f}, RMES: {rmse:.2f}, MAE: {mae:.2f}'.format(split=split, r2=r2, mse=mse, rmse=rmse, mae=mae))

    return metrics_each


metrics_multi = { label_name: calc_metrics_each(label_name) for label_name in label_list }



# Plot yy-graph
num_rows = 1
num_cols = len(label_list) * len(['vat', 'test'])


base_size = 7
height = 1 * base_size
width = num_cols * height 

fig = plt.figure(figsize=(width, height))


for i in range(len(label_list)):
    offset_val = (i * len(['vat', 'test'])) + 1
    offset_test = offset_val + 1

    label_name = label_list[i]
    pred_label_name = 'pred_' + label_name.replace('label_', '')

    ax_val = fig.add_subplot(num_rows,
                             num_cols,
                             offset_val,
                             title=  label_name + '\n' + 'val: Observed-Predicted Plot',
                             xlabel='Observed',
                             ylabel='Predicted',
                             xmargin=0,
                             ymargin=0
                            )

    ax_test = fig.add_subplot(num_rows,
                              num_cols,
                              offset_test,
                              title= label_name + '\n' + 'test: Observed-Predicted Plot',
                              xlabel='Observed',
                              ylabel='Predicted',
                              xmargin=0,
                              ymargin=0
                             )

    # Scores
    df_val = get_column_value(df_likelihood, 'split', ['val'])
    y_obs_val = df_val[label_name].values
    y_pred_val = df_val[pred_label_name].values

    df_test = get_column_value(df_likelihood, 'split', ['test'])
    y_obs_test = df_test[label_name].values
    y_pred_test = df_test[pred_label_name].values

    y_values_val = np.concatenate([y_obs_val.flatten(), y_pred_val.flatten()])
    y_values_test = np.concatenate([y_obs_test.flatten(), y_pred_test.flatten()])

    y_values_val_min, y_values_val_max, y_values_val_range = np.amin(y_values_val), np.amax(y_values_val), np.ptp(y_values_val)
    y_values_test_min, y_values_test_max, y_values_test_range = np.amin(y_values_test), np.amax(y_values_test), np.ptp(y_values_test)

    color = mcolors.TABLEAU_COLORS
    ax_val.scatter(y_obs_val, y_pred_val, color=color['tab:blue'], label='val')
    ax_test.scatter(y_obs_test, y_pred_test, color=color['tab:orange'], label='test')

    # Draw diagonal line
    ax_val.plot([y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)],
                [y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)],
                color='red')

    ax_test.plot([y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)],
                 [y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)],
                 color='red')


# Align graph
fig.tight_layout()


# Save inference result
os.makedirs(yy_dir, exist_ok=True)
basename = get_basename(path_likelihood)
yy_path = os.path.join(yy_dir, 'yy_' + basename) + '.png'
fig.savefig(yy_path) 


# ----- EOF -----
