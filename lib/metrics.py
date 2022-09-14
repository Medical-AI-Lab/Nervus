#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from lifelines.utils import concordance_index
from lib.logger import Logger as logger


from abc import ABC, abstractmethod


# container
# calculation
# plot fig
# save fig
# make summary
# print metrics
# update summary  -> eval.py


# inst -> label -> cal


# Container
class ROC:
    def __init__(self):
        self.fpr = None
        self.tpr = None
        self.auc = None

    def set_roc(self, fpr, tpr):
        self.fpr = fpr
        self.tpr = tpr
        self.auc = metrics.auc(fpr, tpr)


class R2:
    def __init__(self):
        self.y_obs = None
        self.y_pred = None
        self.r2 = None

    def set_r2(self, y_obs, y_pred):
        self.y_obs = y_obs.values
        self.y_pred = y_pred.values
        self.r2 = metrics.r2_score(y_obs, y_pred)


class C_Index:
    def __init__(self):
        self.c_index = None

    def set_c_index(self, periods, preds, internal_labels):
        self.c_index = concordance_index(periods, (-1)*preds, internal_labels)


# Calculate for each label
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
                logger.logger.error('Invalid split.')
                exit()


class Label_C_Index:
    def __init__(self):
        self.val = C_Index()
        self.test = C_Index()

    def cal_label_c_index(self, raw_label_name, df_label):
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            periods = df_split['period']
            preds = df_split['pred_' + raw_label_name]
            internal_labels = df_split['internal_' + raw_label_name]
            if split == 'val':
                self.val.set_c_index(periods, preds, internal_labels)
            elif split == 'test':
                self.test.set_c_index(periods, preds, internal_labels)
            else:
                logger.logger.error('Invalid split.')
                exit()


class BaseMetrics(ABC):
    def __init__(self, task):
        self.task = task

    def cal_inst_metrics(self, df_inst):
        raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])
        inst_metrics = dict()
        for raw_label_name in raw_label_list:
            required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
            df_label = df_inst[required_columns]
            label_roc = LabelROC()
            label_roc.cal_label_roc(raw_label_name, df_label)
            inst_roc[raw_label_name] = label_roc
        return inst_roc


class LabelMetrics(ABC):
    def __init__(self, task):
        self.task = task


class MetricsMixin:
    def save_metrics(self):
        pass

    def plot_fig(self):
        pass

    def save_fig(self):
        pass

    def make_summary(self, whole_metrics, datetime, likelihood_path, metrics_kind: str):
        df_summary = pd.DataFrame()
        for inst, inst_metrics in whole_metrics.items():
            _new = dict()
            _new['datetime'] = [datetime]
            _new['weight'] = [likelihood_path.stem.replace('likelihood_', '') + '.pt']
            _new['Institution'] = [inst]
            for raw_label_name, label_metrics in inst_metrics.items():
                val_metrics = getattr(label_metrics.val, metrics_kind)
                test_metrics = getattr(label_metrics.test, metrics_kind)
                _new[raw_label_name + '_val_' + metrics_kind] = [f"{val_metrics:.2f}"]
                _new[raw_label_name + '_test_' + metrics_kind] = [f"{test_metrics:.2f}"]
            df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

        df_summary = df_summary.sort_values('Institution')
        return df_summary

    def update_summary(self):
        pass



class MetricsWidget(BaseMetrics, MetricsMixin):
    pass

class MetricsROC(MetricsWidget):
    def __init__(self):
        self.task = 'classification'
        self.metrics = 'auc'

    # overwrite
    def make_summary(self, whole_metrics, datetime, likelihood_path):
        return self.make_summary(whole_metrics, datetime, likelihood_path, 'roc')


class MetricsYY(MetricsWidget):
    def __init__(self):
        self.task = 'regression'
        self.metrics = 'r2'

    # overwrite
    def make_summary(self, whole_metrics, datetime, likelihood_path):
        return self.make_summary(whole_metrics, datetime, likelihood_path, 'r2')


class MetricsC_Index(MetricsWidget):
    def __init__(self):
        self.task = 'deepsurv'
        self.metrics = 'c_index'

    # overwrite
    def make_summary(self, whole_metrics, datetime, likelihood_path):
        return self.make_summary(whole_metrics, datetime, likelihood_path, 'c_index')


def set_metrics(task):
    if task == 'classification':
        return  MetricsROC()
    elif task == 'regression':
        return  MetricsYY()
    elif task == 'deepsurv':
        return MetricsC_Index()
    else:
        logger.logger.error(f"Invalid task: {task}.")
        exit()
