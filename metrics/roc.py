#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import dataclasses

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('metrics.roc')


@dataclasses.dataclass
class ROC:
    fpr: np.ndarray = np.array([])
    tpr: np.ndarray = np.array([])
    auc: np.ndarray = np.array([])


class LabelROC:
    def __init__(self):
        self.val = ROC()
        self.test = ROC()

    def set_roc(self, split, fpr=None, tpr=None, auc=None):
        if split == 'val':
            self.val.fpr = fpr,
            self.val.tpr = tpr,
            self.val.auc = auc
        elif split == 'test':
            self.test.fpr = fpr,
            self.test.tpr = tpr,
            self.test.auc = auc
        else:
            logger.error(f"Invalid split: {split}.")


# ROC for each label with binary classes
def cal_roc_binary_class(raw_label_name, df_likelihood):
    """Calculate ROC for eacl raw_label

    Args:
        raw_label_name (str): raw label bame
        df_likelihood (DataFarame): likelihood

    Returns:
        Dict: Dict[LabelROC]
    """
    POSITIVE = 1
    pred_name_list = list(df_likelihood.columns[df_likelihood.columns.str.startswith('pred')])
    pred_positive_name = pred_name_list[POSITIVE]
    class_list = [column_name.rsplit('_', 1)[1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
    positive_class = class_list[POSITIVE]

    roc_raw_label = LabelROC()
    for split in ['val', 'test']:
        df_likelihood_split = df_likelihood[df_likelihood['split'] == split]
        y_true_split = df_likelihood_split[raw_label_name]
        y_score_split = df_likelihood_split[pred_positive_name]
        fpr_split, tpr_split, thresholds_split = metrics.roc_curve(y_true_split.astype('str'), y_score_split, pos_label=positive_class)
        roc_raw_label.set_roc(split, fpr=fpr_split, tpr=tpr_split, auc=metrics.auc(fpr_split, tpr_split))

    logger.info(f"{raw_label_name}, val: {roc_raw_label.val.auc:.2f}, test: {roc_raw_label.test.auc:.2f}")
    return roc_raw_label


def cal_roc_multi_class(label_name, df_likelihood):
    pass


def plot_roc(metrics_roc):
    num_rows = 1
    num_cols = len(label_list)
    base_size = 7
    height = num_rows * base_size
    width = num_cols * height

    fig = plt.figure(figsize=(width, height))
    for i in range(len(label_list)):
        offset = i + 1
        label_name = label_list[i]
        label_roc = metrics_roc[label_name]
        num_class_in_label = df_institution_likelihood[label_name].nunique()

        if num_class_in_label > 2:
            # macro-average ROC for multiclass
            fpr_val = label_roc.val.fpr.macro
            tpr_val = label_roc.val.tpr.macro
            auc_val = label_roc.val.auc.macro
            fpr_test = label_roc.test.fpr.macro
            tpr_test = label_roc.test.tpr.macro
            auc_test = label_roc.test.auc.macro
        else:
            # Ordinary ROC for binaryclass
            fpr_val = label_roc.val.fpr.micro
            tpr_val = label_roc.val.tpr.micro
            auc_val = label_roc.val.auc.micro
            fpr_test = label_roc.test.fpr.micro
            tpr_test = label_roc.test.tpr.micro
            auc_test = label_roc.test.auc.micro

        ax_i = fig.add_subplot(num_rows,
                            num_cols,
                            offset,
                            title=institution + ': ' + label_name,
                            xlabel='1 - Specificity',
                            ylabel='Sensitivity',
                            xmargin=0,
                            ymargin=0)

        ax_i.plot(fpr_val, tpr_val, label=f"AUC_val = {auc_val:.2f}", marker='x')
        ax_i.plot(fpr_test, tpr_test, label=f"AUC_test = {auc_test:.2f}", marker='o')
        ax_i.grid()
        ax_i.legend()

        # Align subplots
        fig.tight_layout()
        return fig


def cal_roc(likelihood_path):
    df_likelihood = pd.read_csv(likelihood_path)

    for institution in df_likelihood['Institution'].unique():
        df_institution_likelihood = df_likelihood.query('Institution == @institution')

        # Calculate ROC and AUC
        raw_label_list = list(df_institution_likelihood.columns[df_institution_likelihood.columns.str.startswith('label')])
        metrics_roc = {}

        logger.info(f"{institution}:")
        for raw_label_name in raw_label_list:
            num_class_in_label = df_institution_likelihood[raw_label_name].nunique()
            if num_class_in_label > 2:
                metrics_roc[raw_label_name] = cal_roc_multi_class(raw_label_name, df_institution_likelihood)
            else:
                metrics_roc[raw_label_name] = cal_roc_binary_class(raw_label_name, df_institution_likelihood)
        logger.info('')


    # Plot ROC
    #fig =  plot_roc(metrics_roc)
    fig = None

    return metrics_roc, fig
