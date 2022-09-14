#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from lib.logger import Logger as logger


class ROC:
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.auc = None

    def set_roc(self, fpr, tpr):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = metrics.auc(fpr, tpr)


class LabelROC:
    def __init__(self):
        self.val = ROC()
        self.test = ROC()

    def set_split_roc(self, split, fpr, tpr):
        if split == 'val':
            self.val.set_roc(fpr, tpr)
        elif split == 'test':
            self.test.set_roc(fpr, tpr)
        else:
            logger.logger.error('Invalid split.')

    def _cal_label_roc_binary(self, raw_label_name, df_label):
        """_summary_

        Args:
            raw_label_name (_type_): _description_
            df_label (_type_): _description_
        """
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
        POSITIVE = 1
        pred_positive_name = pred_name_list[POSITIVE]
        positive_class = class_list[POSITIVE]

        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_true = df_split[raw_label_name]
            y_score = df_split[pred_positive_name]
            _fpr, _tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=positive_class)
            self.set_split_roc(split, _fpr, _tpr)

    def _cal_label_roc_multi(self, raw_label_name, df_label):
        """Calculate ROC for multi-class by macro average.

        Args:
            raw_label_name (str): labe name
            df_label (DataFrame): likelihood for raw_label_name
        """
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]
        num_classes = len(class_list)

        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_true = df_split[raw_label_name]
            y_true_bin = label_binarize(y_true, classes=class_list)

            # Compute ROC for each class by OneVsRest
            _fpr = dict()
            _tpr = dict()
            for i, class_name in enumerate(class_list):
                pred_name = 'pred_' + raw_label_name + '_' + class_name
                _fpr[class_name], _tpr[class_name], _ = metrics.roc_curve(y_true_bin[:, i], df_split[pred_name])

            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([_fpr[class_name] for class_name in class_list]))

            # Then interpolate all ROC at this points
            mean_tpr = np.zeros_like(all_fpr)
            for class_name in class_list:
                mean_tpr += np.interp(all_fpr, _fpr[class_name], _tpr[class_name])

            # Finally average it and compute AUC
            mean_tpr /= num_classes

            _fpr['macro'] = all_fpr
            _tpr['macro'] = mean_tpr
            self.set_split_roc(split, _fpr['macro'], _tpr['macro'])

    def cal_label_roc(self, raw_label_name, df_label):
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        isMulti = (len(pred_name_list) > 2)
        if isMulti:
            self._cal_label_roc_multi(raw_label_name, df_label)
        else:
            self._cal_label_roc_binary(raw_label_name, df_label)


def cal_inst_roc(df_inst):
    raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])
    inst_roc = dict()
    for raw_label_name in raw_label_list:
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
        df_label = df_inst[required_columns]
        label_roc = LabelROC()
        label_roc.cal_label_roc(raw_label_name, df_label)
        inst_roc[raw_label_name] = label_roc
    return inst_roc


def cal_roc(likelihood_path):
    df_likelihood = pd.read_csv(likelihood_path)
    whole_roc = dict()
    for inst in df_likelihood['Institution'].unique():
        df_inst = df_likelihood.query('Institution == @inst')
        whole_roc[inst] = cal_inst_roc(df_inst)
    return whole_roc


def plot_inst_roc(inst, inst_roc):
    raw_label_list = inst_roc.keys()
    num_rows = 1
    num_cols = len(raw_label_list)
    base_size = 7
    height = num_rows * base_size
    width = num_cols * height
    fig = plt.figure(figsize=(width, height))

    for i, raw_label_name in enumerate(raw_label_list):
        label_roc = inst_roc[raw_label_name]
        val_fpr = label_roc.val.fpr
        val_tpr = label_roc.val.tpr
        val_auc = label_roc.val.auc
        test_fpr = label_roc.test.fpr
        test_tpr = label_roc.test.tpr
        test_auc = label_roc.test.auc

        offset = i + 1
        ax_i = fig.add_subplot(
                                num_rows,
                                num_cols,
                                offset,
                                title=inst + ': ' + raw_label_name,
                                xlabel='1 - Specificity',
                                ylabel='Sensitivity',
                                xmargin=0,
                                ymargin=0
                                )
        ax_i.plot(val_fpr, val_tpr, label=f"AUC_val = {val_auc:.2f}", marker='x')
        ax_i.plot(test_fpr, test_tpr, label=f"AUC_test = {test_auc:.2f}", marker='o')
        ax_i.grid()
        ax_i.legend()
        fig.tight_layout()
    return fig


def save_roc(whole_roc, datetime, likelihood_path):
    for inst, inst_roc in whole_roc.items():
        fig = plot_inst_roc(inst, inst_roc)
        save_dir = Path('./results/sets', datetime, 'roc')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, inst + '_roc_' + likelihood_path.stem.replace('likelihood_', '') + '.png')  # 'likelihood_weight_epoch-010_best.csv'  -> inst_roc_weight_epoch-010_best.png
        fig.savefig(save_path)
        plt.close()


def make_summary(whole_roc, datetime, likelihood_path):
    df_summary = pd.DataFrame()
    for inst, inst_roc in whole_roc.items():
        _new = dict()
        _new['datetime'] = [datetime]
        _new['weight'] = [likelihood_path.stem.replace('likelihood_', '') + '.pt']
        _new['Institution'] = [inst]
        for raw_label_name, label_roc in inst_roc.items():
            _new[raw_label_name + '_val_auc'] = [f"{label_roc.val.auc:.2f}"]
            _new[raw_label_name + '_test_auc'] = [f"{label_roc.test.auc:.2f}"]
        df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

    df_summary = df_summary.sort_values('Institution')
    return df_summary


def print_auc(df_summary):
    label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])
    num_splits = len(['val', 'test'])
    _column_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]
    for _, row in df_summary.iterrows():
        logger.logger.info(row['Institution'])
        for _column in _column_list:
            label_name = _column[0].replace('_val_auc', '')
            logger.logger.info(f"{label_name:<25} val_auc: {row[_column[0]]:>5}, test_auc: {row[_column[1]]:>5}")


def make_roc(datetime, likelihood_path):
    whole_roc = cal_roc(likelihood_path)
    save_roc(whole_roc, datetime, likelihood_path)
    df_summary = make_summary(whole_roc, datetime, likelihood_path)
    print_auc(df_summary)
    return df_summary
