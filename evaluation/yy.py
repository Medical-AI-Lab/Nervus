#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import dataclasses

from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib.align_env import *
from options.metrics_options import MetricsOptions


args = MetricsOptions().parse()
nervusenv = NervusEnv()
datetime_dir = get_target(nervusenv.sets_dir, args['likelihood_datetime'])   # args['likelihood_datetime'] if exists or the latest
likelihood_path = os.path.join(datetime_dir, nervusenv.csv_likelihood)
df_likelihood = pd.read_csv(likelihood_path)
output_name_list = [column_name for column_name in df_likelihood.columns if column_name.startswith('output')]

@dataclasses.dataclass
class MetricsYY:
    r2: float
    mse: float
    rmse: float
    mae: float

@dataclasses.dataclass
class LabelMetricsYY:
    val: MetricsYY
    test: MetricsYY

def cal_label_metrics_yy(output_name, df_likelihood):
    output_metrics_yy = LabelMetricsYY(val=None, test=None)
    pred_name = 'pred_' + output_name
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        y_obs_split = df_likelihood_split[output_name].values
        y_pred_split = df_likelihood_split[pred_name].values
        r2 = metrics.r2_score(y_obs_split, y_pred_split)
        mse = metrics.mean_squared_error(y_obs_split, y_pred_split)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_obs_split, y_pred_split)
        if split == 'val':
            output_metrics_yy.val = MetricsYY(r2=r2, mse=mse, rmse=rmse, mae=mae)
        else:
            output_metrics_yy.test = MetricsYY(r2=r2, mse=mse, rmse=rmse, mae=mae)
    print(output_name + ': ')
    print(f"{'val':>5}, R2: {output_metrics_yy.val.r2:>5.2f}, MSE: {output_metrics_yy.val.mse:.2f}, RMES: {output_metrics_yy.val.rmse:.2f}, MAE: {output_metrics_yy.val.mae:.2f}")
    print(f"{'test':>5}, R2: {output_metrics_yy.test.r2:>5.2f}, MSE: {output_metrics_yy.test.mse:.2f}, RMES: {output_metrics_yy.test.rmse:.2f}, MAE: {output_metrics_yy.test.mae:.2f}")
    return output_metrics_yy

metrics_yy = {output_name: cal_label_metrics_yy(output_name, df_likelihood) for output_name in output_name_list}

# Plot yy-graph
len_splits = len(['vat', 'test'])
num_rows = 1
num_cols = len(output_name_list) * len_splits
base_size = 7
height = num_rows * base_size
width = num_cols * height 

fig = plt.figure(figsize=(width, height))
for i in range(len(output_name_list)):
    offset_val = (i * len_splits) + 1
    offset_test = offset_val + 1
    output_name = output_name_list[i]
    pred_name = 'pred_' + output_name

    ax_val = fig.add_subplot(num_rows,
                             num_cols,
                             offset_val,
                             title=output_name + '\n' + 'val: Observed-Predicted Plot',
                             xlabel='Observed',
                             ylabel='Predicted',
                             xmargin=0,
                             ymargin=0)
    ax_test = fig.add_subplot(num_rows,
                              num_cols,
                              offset_test,
                              title=output_name + '\n' + 'test: Observed-Predicted Plot',
                              xlabel='Observed',
                              ylabel='Predicted',
                              xmargin=0,
                              ymargin=0)

    df_val = get_column_value(df_likelihood, 'split', ['val'])
    y_obs_val = df_val[output_name].values
    y_pred_val = df_val[pred_name].values

    df_test = get_column_value(df_likelihood, 'split', ['test'])
    y_obs_test = df_test[output_name].values
    y_pred_test = df_test[pred_name].values

    y_values_val = np.concatenate([y_obs_val.flatten(), y_pred_val.flatten()])
    y_values_test = np.concatenate([y_obs_test.flatten(), y_pred_test.flatten()])
    y_values_val_min, y_values_val_max, y_values_val_range = np.amin(y_values_val), np.amax(y_values_val), np.ptp(y_values_val)
    y_values_test_min, y_values_test_max, y_values_test_range = np.amin(y_values_test), np.amax(y_values_test), np.ptp(y_values_test)

    # Plot
    color = mcolors.TABLEAU_COLORS
    ax_val.scatter(y_obs_val, y_pred_val, color=color['tab:blue'], label='val')
    ax_test.scatter(y_obs_test, y_pred_test, color=color['tab:orange'], label='test')

    # Draw diagonal line
    ax_val.plot([y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)],
                [y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)], color='red')
    ax_test.plot([y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)],
                 [y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)], color='red')

# Align graph
fig.tight_layout()

# Save Fig
yy_path = os.path.join(datetime_dir, nervusenv.yy)
fig.savefig(yy_path)


# Save metrics of yy
datetime = os.path.basename(datetime_dir)
summary_new = dict()
summary_new['datetime'] = [datetime]
for output_name in metrics_yy.keys():
    output_metrics_yy = metrics_yy[output_name]
    summary_new[output_name+'_val_r2'] = [f"{output_metrics_yy.val.r2:.2f}"]
    summary_new[output_name+'_val_mse'] = [f"{output_metrics_yy.val.mse:.2f}"]
    summary_new[output_name+'_val_rmse'] = [f"{output_metrics_yy.val.rmse:.2f}"]
    summary_new[output_name+'_val_mae'] = [f"{output_metrics_yy.val.mae:.2f}"]

    summary_new[output_name+'_test_r2'] = [f"{output_metrics_yy.test.r2:.2f}"]
    summary_new[output_name+'_test_mse'] = [f"{output_metrics_yy.test.mse:.2f}"]
    summary_new[output_name+'_test_rmse'] = [f"{output_metrics_yy.test.rmse:.2f}"]
    summary_new[output_name+'_test_mae'] = [f"{output_metrics_yy.test.mae:.2f}"]
df_summary_new = pd.DataFrame(summary_new)
update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)


# ----- EOF -----
