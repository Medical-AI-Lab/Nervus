#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from lib import get_logger


log = get_logger('metrics.yy')


class R2:
    def __init__(self):
        self.y_obs = None
        self.y_pred = None
        self.r2 = None

    def set_r2(self, y_obs, y_pred):
        self.y_obs = y_obs.values
        self.y_pred = y_pred.values
        self.r2 = metrics.r2_score(y_obs, y_pred)


class LabelR2:
    def __init__(self):
        self.val = R2()
        self.test = R2()

    def cal_label_r2(self, raw_label_name, df_label):
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_obs = df_split[raw_label_name]
            y_pred = df_split['pred_' + raw_label_name]
            if split == 'val':
                self.val.set_r2(y_obs, y_pred)
            elif split == 'test':
                self.test.set_r2(y_obs, y_pred)
            else:
                log.error('Invalid split.')
                exit()


def cal_inst_r2(df_inst):
    raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])
    inst_r2 = dict()
    for raw_label_name in raw_label_list:
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
        df_label = df_inst[required_columns]
        label_r2 = LabelR2()
        label_r2.cal_label_r2(raw_label_name, df_label)
        inst_r2[raw_label_name] = label_r2
    return inst_r2


def cal_r2(likelihood_path):
    df_likelihood = pd.read_csv(likelihood_path)
    whole_r2 = dict()
    for inst in df_likelihood['Institution'].unique():
        df_inst = df_likelihood.query('Institution == @inst')
        whole_r2[inst] = cal_inst_r2(df_inst)
    return whole_r2


def plot_inst_yy(inst, inst_r2):
    raw_label_list = inst_r2.keys()
    num_splits = len(['val', 'test'])
    num_rows = 1
    num_cols = len(raw_label_list) * num_splits
    base_size = 7
    height = num_rows * base_size
    width = num_cols * height
    fig = plt.figure(figsize=(width, height))

    for i, raw_label_name in enumerate(raw_label_list):
        label_r2 = inst_r2[raw_label_name]
        val_offset = (i * num_splits) + 1
        test_offset = val_offset + 1

        val_ax = fig.add_subplot(
                                num_rows,
                                num_cols,
                                val_offset,
                                title=inst + ': ' + raw_label_name + '\n' + 'val: Observed-Predicted Plot',
                                xlabel='Observed',
                                ylabel='Predicted',
                                xmargin=0,
                                ymargin=0
                                )
        test_ax = fig.add_subplot(
                                num_rows,
                                num_cols,
                                test_offset,
                                title=inst + ': ' + raw_label_name + '\n' + 'test: Observed-Predicted Plot',
                                xlabel='Observed',
                                ylabel='Predicted',
                                xmargin=0,
                                ymargin=0
                                )

        y_obs_val = label_r2.val.y_obs
        y_pred_val = label_r2.val.y_pred
        y_obs_test = label_r2.test.y_obs
        y_pred_test = label_r2.test.y_pred

        y_values_val = np.concatenate([y_obs_val.flatten(), y_pred_val.flatten()])
        y_values_test = np.concatenate([y_obs_test.flatten(), y_pred_test.flatten()])

        y_values_val_min, y_values_val_max, y_values_val_range = np.amin(y_values_val), np.amax(y_values_val), np.ptp(y_values_val)
        y_values_test_min, y_values_test_max, y_values_test_range = np.amin(y_values_test), np.amax(y_values_test), np.ptp(y_values_test)

        # Plot
        color = mcolors.TABLEAU_COLORS
        val_ax.scatter(y_obs_val, y_pred_val, color=color['tab:blue'], label='val')
        test_ax.scatter(y_obs_test, y_pred_test, color=color['tab:orange'], label='test')

        # Draw diagonal line
        val_ax.plot([y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)],
                    [y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)], color='red')

        test_ax.plot([y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)],
                     [y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)], color='red')

    fig.tight_layout()
    return fig


def save_yy(whole_r2, datetime, likelihood_path):
    for inst, inst_roc in whole_r2.items():
        fig = plot_inst_yy(inst, inst_roc)
        save_dir = Path('./results/sets', datetime, 'yy')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, inst + '_yy_' + likelihood_path.stem.replace('likelihood_', '') + '.png')  # 'likelihood_weight_epoch-010_best.csv'  -> inst_yy_weight_epoch-010_best.png
        fig.savefig(save_path)
        plt.close()


def make_summary(whole_r2, datetime, likelihood_path):
    df_summary = pd.DataFrame()
    for inst, inst_r2 in whole_r2.items():
        _new = dict()
        _new['datetime'] = [datetime]
        _new['weight'] = [likelihood_path.name.replace('likelihood_', '')]
        _new['Institution'] = [inst]
        for raw_label_name, label_r2 in inst_r2.items():
            _new[raw_label_name + '_val_r2'] = [f"{label_r2.val.r2:.2f}"]
            _new[raw_label_name + '_test_r2'] = [f"{label_r2.test.r2:.2f}"]
        df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

    df_summary = df_summary.sort_values('Institution')
    return df_summary


def print_r2(df_summary):
    label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])
    num_splits = len(['val', 'test'])
    _column_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]
    for _, row in df_summary.iterrows():
        log.info(row['Institution'])
        for _column in _column_list:
            label_name = _column[0].replace('_val_r2', '')
            log.info(f"{label_name:<25} val_r2: {row[_column[0]]:>7}, test_r2: {row[_column[1]]:>7}")


def make_yy(datetime, likelihood_path):
    whole_r2 = cal_r2(likelihood_path)
    save_yy(whole_r2, datetime, likelihood_path)
    df_summary = make_summary(whole_r2, datetime, likelihood_path)
    print_r2(df_summary)
    return df_summary
