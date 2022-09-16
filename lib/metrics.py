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

# container
# calculation
# plot fig
# save fig
# make summary
# print metrics
# update summary  -> eval.py
# inst -> label -> cal


class LabelROC:
    """
    Class just to store ROC and AUC
    """
    def __init__(self) -> None:
        self.val_frp = None
        self.val_trp = None
        self.val_auc = None

        self.test_frp = None
        self.test_trp = None
        self.test_auc = None


class ROCMixin:
    def _set_roc(self, label_roc, split, fpr, tpr):
        setattr(label_roc, split + '_fpr', fpr)
        setattr(label_roc, split + '_tpr', tpr)
        setattr(label_roc, split + '_auc', metrics.auc(fpr, tpr))
        return label_roc

    def _cal_label_roc_binary(self, raw_label_name: str, df_label: pd.DataFrame) -> None:
        """
        Calculate ROC for binaly-class

        Args:
            raw_label_name (str): raw label name
            df_label (pd.DataFrame): likelihood for raw label name
        """
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
        POSITIVE = 1
        pred_positive_name = pred_name_list[POSITIVE]
        positive_class = class_list[POSITIVE]

        label_roc = LabelROC()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_true = df_split[raw_label_name]
            y_score = df_split[pred_positive_name]
            _fpr, _tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=positive_class)
            label_roc = self._set_roc(label_roc, split, _fpr, _tpr)
        return label_roc

    def _cal_label_roc_multi(self, raw_label_name, df_label):
        """
        Calculate ROC for multi-class by macro average.

        Args:
            raw_label_name (str): raw labe name
            df_label (pd.DataFrame): likelihood for raw label name
        """
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]
        num_classes = len(class_list)

        label_roc = LabelROC()
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
            label_roc = self._set_roc(label_roc, split, _fpr, _tpr)
        return label_roc

    def cal_label_roc(self, raw_label_name, df_label):
        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        isMultiClass = (len(pred_name_list) > 2)
        if isMultiClass:
            self._cal_label_roc_multi(raw_label_name, df_label)
        else:
            self._cal_label_roc_binary(raw_label_name, df_label)

    def plot_inst_roc(self, inst, inst_roc):
        raw_label_list = inst_roc.keys()
        num_rows = 1
        num_cols = len(raw_label_list)
        base_size = 7
        height = num_rows * base_size
        width = num_cols * height
        fig = plt.figure(figsize=(width, height))

        for i, raw_label_name in enumerate(raw_label_list):
            label_roc = inst_roc[raw_label_name]
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
            ax_i.plot(label_roc.val_fpr, label_roc.val_tpr, label=f"AUC_val = {label_roc.val_auc:.2f}", marker='x')
            ax_i.plot(label_roc.test_fpr, label_roc.test_tpr, label=f"AUC_test = {label_roc.test_auc:.2f}", marker='o')
            ax_i.grid()
            ax_i.legend()
            fig.tight_layout()
        return fig


class LabelYY:
    """
    Class just to store R2
    """
    def __init__(self) -> None:
        self.val_y_obs = None
        self.val_y_pred = None
        self.val_r2 = None

        self.test_y_obs = None
        self.test_y_pred = None
        self.test_r2 = None


class YYMixin:
    def _set_yy(self, label_yy, split, y_obs, y_pred):
        setattr(label_yy, split + '_y_obs', y_obs)
        setattr(label_yy, split + '_y_pred', y_pred)
        setattr(label_yy, split + '_r2', metrics.r2_score(y_obs, y_pred))
        return label_yy

    def cal_label_yy(self, raw_label_name, df_label):
        label_yy = LabelYY()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_obs = df_split[raw_label_name]
            y_pred = df_split['pred_' + raw_label_name]
            self._set_yy(label_yy, split, y_obs, y_pred)

    def plot_inst_yy(self, inst, inst_r2):
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

            y_obs_val = label_r2.val_y_obs
            y_pred_val = label_r2.val_y_pred
            y_obs_test = label_r2.test_y_obs
            y_pred_test = label_r2.test_y_pred

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


class LabelC_Index:
    """
    Class just to store c-index
    """
    def __init__(self) -> None:
        self.val_c_index = None
        self.test_c_index = None


class C_IndexMixin:
    def _set_c_index(self, label_c_index, split, periods, preds, internal_labels):
        setattr(label_c_index, split + '_c_index', concordance_index(periods, (-1)*preds, internal_labels))
        return label_c_index

    def cal_label_c_index(self, raw_label_name, df_label):
        label_c_index = LabelC_Index()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            periods = df_split['period']
            preds = df_split['pred_' + raw_label_name]
            internal_labels = df_split['internal_' + raw_label_name]
            self._set_c_index(label_c_index, split, periods, preds, internal_labels)


class MetricsMixin:
    def _cal_inst_metrics(self, df_inst):
        raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])
        inst_metrics = dict()
        for raw_label_name in raw_label_list:
            required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
            df_label = df_inst[required_columns]
            label_roc = self.cal_label_metrics(raw_label_name, df_label)
            inst_metrics[raw_label_name] = label_roc
        return inst_metrics

    def cal_whole_metrics(self, likelihood_path):
        df_likelihood = pd.read_csv(likelihood_path)
        whole_roc = dict()
        for inst in df_likelihood['Institution'].unique():
            df_inst = df_likelihood.query('Institution == @inst')
            whole_roc[inst] = self._cal_inst_metrics(df_inst)
        return whole_roc

    def save_fig(self, whole_metrics, datetime, likelihood_path, metrics_kind):
        for inst, inst_metrics in whole_metrics.items():
            fig = self.plot_inst_metrics(inst, inst_metrics)
            save_dir = Path('./results/sets', datetime, metrics_kind)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir, inst + '_' + metrics_kind + '_' + likelihood_path.stem.replace('likelihood_', '') + '.png')  # 'likelihood_weight_epoch-010_best.csv'  -> inst_roc_weight_epoch-010_best.png
            fig.savefig(save_path)
            plt.close()

    def make_summary(self, whole_metrics, datetime, likelihood_path, metrics_kind):
        df_summary = pd.DataFrame()
        for inst, inst_metrics in whole_metrics.items():
            _new = dict()
            _new['datetime'] = [datetime]
            _new['weight'] = [likelihood_path.stem.replace('likelihood_', '') + '.pt']
            _new['Institution'] = [inst]
            for raw_label_name, label_metrics in inst_metrics.items():
                _val_metrics = getattr(label_metrics.val, metrics_kind)
                _test_metrics = getattr(label_metrics.test, metrics_kind)
                _new[raw_label_name + '_val_' + metrics_kind] = [f"{_val_metrics:.2f}"]
                _new[raw_label_name + '_test_' + metrics_kind] = [f"{_test_metrics:.2f}"]
            df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

        df_summary = df_summary.sort_values('Institution')
        return df_summary

    def print_metrics(df_summary, metrics_kind):
        label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])  # [label_1_val, label_1_test, label_2_val, label_2_test, ...]
        num_splits = len(['val', 'test'])
        _column_val_test_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]  # [[label_1_val, label_1_test], [label_2_val, label_2_test], ...]
        for _, row in df_summary.iterrows():
            logger.logger.info(row['Institution'])
            for _column_val_test in _column_val_test_list:
                _label_name_val = _column_val_test[0]
                _label_name_test = _column_val_test[1]
                _label_name = _label_name_val.replace('_val_' + metrics_kind, '')
                logger.logger.info(f"{_label_name:<25} val_{metrics_kind}: {row[_label_name_val]:>7}, test_{metrics_kind}: {row[_label_name_test]:>7}")

    def update_summary(df_summary):
        summary_dir = Path('./results/summary')
        summary_path = Path(summary_dir, 'summary.csv')
        if summary_path.exists():
            df_prev = pd.read_csv(summary_path)
            df_updated = pd.concat([df_prev, df_summary], axis=0)
        else:
            summary_dir.mkdir(parents=True, exist_ok=True)
            df_updated = df_summary
        df_updated.to_csv(summary_path, index=False)


class MetricsROC(ROCMixin, MetricsMixin):
    def __init__(self):
        pass

    def cal_label_metrics(self, raw_label_name, df_label):
        return self.cal_label_roc(raw_label_name, df_label)

    def plot_inst_metrics(self, inst, inst_roc):
        return self.plot_inst_roc(inst, inst_roc)


class MetricsYY(YYMixin, MetricsMixin):
    def __init__(self):
        pass

    def cal_label_metrics(self, raw_label_name, df_label):
        return self.cal_label_yy(raw_label_name, df_label)

    def plot_inst_metrics(self, inst, inst_r2):
        return self.plot_inst_yy(inst, inst_r2)


class MetricsC_Index(C_IndexMixin, MetricsMixin):
    def __init__(self):
        pass

    def cal_label_metrics(self, raw_label_name, df_label):
        return self.cal_label_c_index(raw_label_name, df_label)

    def plot_inst_metrics(self):
        raise NotImplementedError('No figure for c-index.')


def set_metrics(task):
    if task == 'classification':
        return MetricsROC()
    elif task == 'regression':
        return MetricsYY()
    elif task == 'seepsurv':
        return MetricsC_Index()
    else:
        logger.logger.error(f"Invalid task: {task}.")


def make_metrics(datetime, likelihood_path):
    cls_metrics = MetricsROC()
    whole_roc = cls_metrics.cal_whole_metrics(likelihood_path)
    cls_metrics.save_fig(whole_roc, datetime, likelihood_path)
    df_summary = cls_metrics.make_summary(whole_roc, datetime, likelihood_path)
    cls_metrics.print_metrics(df_summary)
    cls_metrics.update_summary(df_summary)
    return df_summary