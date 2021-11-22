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
label_name = label_list[0]
pred_label_name = 'pred_' + label_name.replace('label_', '')


def reg_metric(y_obs, y_pred):
    r2 = metrics.r2_score(y_obs, y_pred)
    mse = metrics.mean_squared_error(y_obs, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_obs, y_pred)
    return r2, mse, rmse, mae


#def plot_yy():
# Scores
df_val = get_column_value(df_likelihood, 'split', ['val'])
y_obs_val = df_val[label_name].values
y_pred_val = df_val[pred_label_name].values

df_test = get_column_value(df_likelihood, 'split', ['test'])
y_obs_test = df_test[label_name].values
y_pred_test = df_test[pred_label_name].values


# Metrics
r2_val, mse_val, rmse_val, mae_val = reg_metric(y_obs_val, y_pred_val)
r2_test, mse_test, rmse_test, mae_test = reg_metric(y_obs_test, y_pred_test)

print(' val: R2: {r2_val:.2f}, MSE: {mse_val:.2f}, RMES: {rmse_val:.2f}, MAE: {mae_val:.2f}'
.format(r2_val=r2_val, mse_val=mse_val, rmse_val=rmse_val, mae_val=mae_val))

print('test: R2: {r2_test:.2f}, MSE: {mse_test:.2f}, RMES: {rmse_test:.2f}, MAE: {mae_test:.2f}'
.format(r2_test=r2_test, mse_test=mse_test, rmse_test=rmse_test, mae_test=mae_test))



# Make figure
fig = plt.figure(figsize=(16, 8))

ax_val = fig.add_subplot(1, 2, 1, title='val: Observed-Predicted Plot', xlabel='Observed', ylabel='Predicted', xmargin=0, ymargin=0)
ax_test = fig.add_subplot(1, 2, 2, title='test: Observed-Predicted Plot', xlabel='Observed', ylabel='Predicted', xmargin=0, ymargin=0)
#ax_val.legend(loc = 'lower right')
#ax_test.legend(loc = 'lower right')


# Plot
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
yy_prefix = ('yy_mse-{mse_val:.2f}-{mse_test:.2f}_rmse-{rmse_val:.2f}-{rmse_test:.2f}_mae-{mae_val:.2f}-{mae_test:.2f}'
                .format(mse_val=mse_val,
                rmse_val=rmse_val,
                mae_val=mae_val,
                mse_test=mse_test,
                rmse_test=rmse_test,
                mae_test=mae_test))

yy_name = strcat('_', yy_prefix ,basename)
yy_path = os.path.join(yy_dir, yy_name) + '.png'
fig.savefig(yy_path) 


# ----- EOF -----
