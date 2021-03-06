#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import pandas as pd
import dataclasses

from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import *
from options import MetricsOptions

logger = NervusLogger.get_logger('evaluation.yy')

nervusenv = NervusEnv()
args = MetricsOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['likelihood_datetime'])   # args['likelihood_datetime'] if exists or the latest


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


def cal_label_metrics_yy(label_name, df_likelihood):
    label_metrics_yy = LabelMetricsYY(val=None, test=None)
    pred_name = 'pred_' + label_name
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        y_obs_split = df_likelihood_split[label_name].values
        y_pred_split = df_likelihood_split[pred_name].values
        r2 = metrics.r2_score(y_obs_split, y_pred_split)
        mse = metrics.mean_squared_error(y_obs_split, y_pred_split)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(y_obs_split, y_pred_split)
        if split == 'val':
            label_metrics_yy.val = MetricsYY(r2=r2, mse=mse, rmse=rmse, mae=mae)
        else:
            label_metrics_yy.test = MetricsYY(r2=r2, mse=mse, rmse=rmse, mae=mae)

    logger.info(label_name + ':')
    logger.info(f"{'val':>5}, R2: {label_metrics_yy.val.r2:>6.2f}, MSE: {label_metrics_yy.val.mse:>6.2f}, RMSE: {label_metrics_yy.val.rmse:>6.2f}, MAE: {label_metrics_yy.val.mae:>6.2f}")
    logger.info(f"{'test':>5}, R2: {label_metrics_yy.test.r2:>6.2f}, MSE: {label_metrics_yy.test.mse:>6.2f}, RMSE: {label_metrics_yy.test.rmse:>6.2f}, MAE: {label_metrics_yy.test.mae:>6.2f}")
    return label_metrics_yy



df_summary_new = pd.DataFrame([])

# Make yy-plot for each weight
likelihood_dir = os.path.join(datetime_dir, nervusenv.likelihood_dir)
likelihood_path_list = sorted(glob.glob(likelihood_dir + '/*.csv'))

for likelihood_path in likelihood_path_list:
    likelihood_name = os.path.basename(likelihood_path)   # likelihood_weight_epoch-004-best.csv
    logger.info(f"\nLikelihood: {likelihood_name}")
    df_likelihood = pd.read_csv(likelihood_path)

    for institution in df_likelihood['Institution'].unique():
        df_institution_likelihood = df_likelihood.query('Institution == @institution')

        # Calculate metrics
        label_list = list(df_institution_likelihood.columns[df_institution_likelihood.columns.str.startswith('label')])

        logger.info(f"{institution}:")
        metrics_yy = {label_name: cal_label_metrics_yy(label_name, df_institution_likelihood) for label_name in label_list}

        # Plot yy-graph
        len_splits = len(['val', 'test'])
        num_rows = 1
        num_cols = len(label_list) * len_splits
        base_size = 7
        height = num_rows * base_size
        width = num_cols * height

        fig = plt.figure(figsize=(width, height))
        for i in range(len(label_list)):
            offset_val = (i * len_splits) + 1
            offset_test = offset_val + 1
            label_name = label_list[i]
            pred_name = 'pred_' + label_name

            ax_val = fig.add_subplot(num_rows,
                                    num_cols,
                                    offset_val,
                                    title=institution + ': ' + label_name + '\n' + 'val: Observed-Predicted Plot',
                                    xlabel='Observed',
                                    ylabel='Predicted',
                                    xmargin=0,
                                    ymargin=0)
            ax_test = fig.add_subplot(num_rows,
                                    num_cols,
                                    offset_test,
                                    title=institution + ': ' + label_name + '\n' + 'test: Observed-Predicted Plot',
                                    xlabel='Observed',
                                    ylabel='Predicted',
                                    xmargin=0,
                                    ymargin=0)

            df_val = get_column_value(df_institution_likelihood, 'split', ['val'])
            y_obs_val = df_val[label_name].values
            y_pred_val = df_val[pred_name].values

            df_test = get_column_value(df_institution_likelihood, 'split', ['test'])
            y_obs_test = df_test[label_name].values
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
        #yy_path = os.path.join(datetime_dir, nervusenv.yy)
        yy_name = institution + '_' + likelihood_name.replace(nervusenv.csv_name_likelihood, nervusenv.yy_name).replace('.csv', '.png')   # yy_weight_epoch-004-best.png
        yy_dir = os.path.join(datetime_dir, nervusenv.yy_dir)
        os.makedirs(yy_dir, exist_ok=True)
        yy_path = os.path.join(yy_dir, yy_name)
        fig.savefig(yy_path)


        # Save metrics of yy
        datetime = os.path.basename(datetime_dir)
        weight_name = likelihood_name.replace(nervusenv.csv_name_likelihood + '_', '').replace('.csv', '.pt')   # weight_epoch-004-best.pt
        summary_tmp = dict()
        summary_tmp['datetime'] = [datetime]
        summary_tmp['weight'] = [weight_name]
        summary_tmp['Institution'] = [institution]
        for label_name in metrics_yy.keys():
            label_metrics_yy = metrics_yy[label_name]
            summary_tmp[label_name+'_val_r2'] = [f"{label_metrics_yy.val.r2:.2f}"]
            summary_tmp[label_name+'_val_mse'] = [f"{label_metrics_yy.val.mse:.2f}"]
            summary_tmp[label_name+'_val_rmse'] = [f"{label_metrics_yy.val.rmse:.2f}"]
            summary_tmp[label_name+'_val_mae'] = [f"{label_metrics_yy.val.mae:.2f}"]

            summary_tmp[label_name+'_test_r2'] = [f"{label_metrics_yy.test.r2:.2f}"]
            summary_tmp[label_name+'_test_mse'] = [f"{label_metrics_yy.test.mse:.2f}"]
            summary_tmp[label_name+'_test_rmse'] = [f"{label_metrics_yy.test.rmse:.2f}"]
            summary_tmp[label_name+'_test_mae'] = [f"{label_metrics_yy.test.mae:.2f}"]

        df_summary_tmp = pd.DataFrame(summary_tmp)
        df_summary_new = pd.concat([df_summary_new, df_summary_tmp], ignore_index=True)


# Update summary
#df_summary_new = pd.DataFrame(summary_new)
update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)
