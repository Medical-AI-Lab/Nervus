#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('metrics.roc')


class ROC:
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.auc = None

    def cal_binary_class_roc(self, y_true, y_score, positive_class):
        _fpr, _tpr, _thresholds = metrics.roc_curve(y_true.astype('str'), y_score, pos_label=positive_class)
        self.fpr = _fpr
        self.tpr = _tpr
        self.auc = metrics.auc(self.fpr, self.tpr)

    def set_multi_class_roc(self, y_true, y_score):
        pass


class LabelROC:
    def __init__(self):
        self.val = ROC()
        self.test = ROC()

    def cal_label_roc(self, raw_label_name, df_label):
        POSITIVE = 1
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        pred_positive_name = pred_name_list[POSITIVE]
        class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
        positive_class = class_list[POSITIVE]

        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_true = df_split[raw_label_name]
            y_score = df_split[pred_positive_name]

            if split == 'val':
                self.val.cal_binary_class_roc(y_true, y_score, positive_class)  # binary or multi-class ?
            elif split == 'test':
                self.test.cal_binary_class_roc(y_true, y_score, positive_class)
            else:
                logger.error('Invalid split.')
                exit()


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


def make_summary(whole_roc, datetime, likelihood_path):
    df_summary = pd.DataFrame()
    _new = dict()
    for inst, inst_roc in whole_roc.items():
        _new['datetime'] = [datetime]
        _new['weight'] = [likelihood_path.name.replace('likelihood_', '')]
        _new['Institution'] = [inst]
        for raw_label_name, label_roc in inst_roc.items():
            _new[raw_label_name + '_val_auc'] = [f"{label_roc.val.auc:.2f}"]
            _new[raw_label_name + '_test_auc'] = [f"{label_roc.test.auc:.2f}"]
        df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

    df_summary = df_summary.sort_values('Institution')
    return df_summary


def print_auc(whole_roc):
    for inst, inst_roc in whole_roc.items():
        logger.info(inst)
        for raw_label_name, label_roc in inst_roc.items():
            logger.info(f"{raw_label_name}, val: {label_roc.val.auc:.2f}, test: {label_roc.test.auc:.2f}")


def make_roc(datetime, likelihood_path):
    whole_roc = cal_roc(likelihood_path)
    save_roc(whole_roc, datetime, likelihood_path)
    df_summary = make_summary(whole_roc, datetime, likelihood_path)
    print_auc(whole_roc)
    return df_summary
