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
from .logger import Logger as logger
from typing import Dict, Union


class MetricsData:
    """
    Class to store metrics as class variable.
    Metrics are defined depenging on task.

    For ROC
        self.fpr: np.ndarray
        self.tpr: np.ndarray
        self.auc: float

    For Regresion
        self.y_obs: np.ndarray
        self.y_pred: np.ndarray
        self.r2: float

    For DeepSurv
        self.c_index: float
    """
    def __init__(self) -> None:
        pass


class LabelMetrics:
    """
    Class to store metrics of each split for each label.
    """
    def __init__(self) -> None:
        """
        Metrics of split, ie 'val' and 'test'
        """
        self.val = MetricsData()
        self.test = MetricsData()

    def set_label_metrics(self, split: str, attr: str, value: Union[np.ndarray, float]) -> None:
        """
        Set value as appropriate metrics of split.

        Args:
            split (str): split
            attr (str): metrics name as follows:
                        FPR, TPR, and AUC, or
                        ground truth, prediction and R2, or
                        c_index
            value (Union[np.ndarray,float]): value of attr
        """
        setattr(getattr(self, split), attr, value)

    def get_label_metrics(self, split: str, attr: str) -> Union[np.ndarray, float]:
        """
        Return value of metrics of split.

        Args:
            split (str): split
            attr (str): metrics name

        Returns:
            Union[np.ndarray,float]: value of attr
        """
        return getattr(getattr(self, split), attr)


class ROCMixin:
    """
    Class for calculating ROC and AUC.
    """
    def _set_roc(self, label_metrics: LabelMetrics, split: str, fpr: np.ndarray, tpr: np.ndarray) -> None:
        """
        Set fpr, tpr, and auc.

        Args:
            label_metrics (LabelMetrics): metrics of 'val' and 'test'
            split (str): 'val' or 'test'
            fpr (np.ndarray): FPR
            tpr (np.ndarray): TPR

        self.metrics_kind = 'auc' is defined in class ClsEval below.
        """
        label_metrics.set_label_metrics(split, 'fpr', fpr)
        label_metrics.set_label_metrics(split, 'tpr', tpr)
        label_metrics.set_label_metrics(split, self.metrics_kind, metrics.auc(fpr, tpr))

    def _cal_label_roc_binary(self, raw_label_name: str, df_inst: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC for binary class.

        Args:
            raw_label_name (str): raw label name
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
        df_label = df_inst[required_columns]

        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]   # [pred_label_discharge, pred_label_decease] -> ['discharge', 'decease']
        POSITIVE = 1
        pred_positive_name = pred_name_list[POSITIVE]
        positive_class = class_list[POSITIVE]

        label_metrics = LabelMetrics()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_true = df_split[raw_label_name]
            y_score = df_split[pred_positive_name]
            _fpr, _tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=positive_class)
            self._set_roc(label_metrics, split, _fpr, _tpr)
        return label_metrics

    def _cal_label_roc_multi(self, raw_label_name: str, df_inst: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC for multi-class by macro average.

        Args:
            raw_label_name (str): raw label name
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
        df_label = df_inst[required_columns]

        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [column_name.rsplit('_', 1)[-1] for column_name in pred_name_list]
        num_classes = len(class_list)

        label_metrics = LabelMetrics()
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
            self._set_roc(label_metrics, split, _fpr['macro'], _tpr['macro'])
        return label_metrics

    def cal_label_metrics(self, raw_label_name: str, df_inst: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC and AUC for label depending on the number of classes of raw label.

        Args:
            raw_label_name (str): raw label name
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        pred_name_list = df_inst.columns[df_inst.columns.str.contains(raw_label_name) & df_inst.columns.str.startswith('pred')]
        isMultiClass = (len(pred_name_list) > 2)
        if isMultiClass:
            label_metrics = self._cal_label_roc_multi(raw_label_name, df_inst)
        else:
            label_metrics = self._cal_label_roc_binary(raw_label_name, df_inst)
        return label_metrics


class YYMixin:
    """
    Class for calculating YY and R2.
    """
    def _set_yy(self, label_metrics: LabelMetrics, split: str, y_obs: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Set ground truth, prediction, and R2.

        Args:
            label_metrics (LabelMetrics): metrics of 'val' and 'test'
            split (str): 'val' or 'test'
            y_obs (np.ndarray): ground truth
            y_pred (np.ndarray): prediction

        self.metrics_kind = 'r2' is defined in class RegEval below.
        """
        label_metrics.set_label_metrics(split, 'y_obs', y_obs.values)
        label_metrics.set_label_metrics(split, 'y_pred', y_pred.values)
        label_metrics.set_label_metrics(split, self.metrics_kind, metrics.r2_score(y_obs, y_pred))

    def cal_label_metrics(self, raw_label_name: str, df_inst: pd.DataFrame) -> LabelMetrics:
        """
        Calculate YY and R2 for raw label.

        Args:
            raw_label_name (str): raw label name
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['split']
        df_label = df_inst[required_columns]
        label_metrics = LabelMetrics()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            y_obs = df_split[raw_label_name]
            y_pred = df_split['pred_' + raw_label_name]
            self._set_yy(label_metrics, split, y_obs, y_pred)
        return label_metrics


class C_IndexMixin:
    """
    Class for calculating C-Index.
    """
    def _set_c_index(
                    self,
                    label_metrics: LabelMetrics,
                    split: str,
                    periods: pd.Series,
                    preds: pd.Series,
                    internal_labels: pd.Series
                    ) -> None:
        """
        Set C-Index.

        Args:
            label_metrics (LabelMetrics): metrics of 'val' and 'test'
            split (str): 'val' or 'test'
            periods (pd.Series): periods
            preds (pd.Series): prediction
            internal_labels (pd.Series): internal label

        self.metrics_kind = 'c_index' is defined in class DeepSurvEval below.
        """
        value_c_index = concordance_index(periods, (-1)*preds, internal_labels)
        label_metrics.set_label_metrics(split, self.metrics_kind, value_c_index)

    def cal_label_metrics(self, raw_label_name: str, df_inst: pd.DataFrame) -> LabelMetrics:
        """
        Calculate C-Index for raw label.

        Args:
            raw_label_name (str): raw label name
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['period', 'split']
        df_label = df_inst[required_columns]
        label_metrics = LabelMetrics()
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            periods = df_split['period']
            preds = df_split['pred_' + raw_label_name]
            internal_labels = df_split['internal_' + raw_label_name]
            self._set_c_index(label_metrics, split, periods, preds, internal_labels)
        return label_metrics


class MetricsMixin:
    """
    Class which has common methods to calculating metrics and making summary.
    """
    def _cal_inst_metrics(self, df_inst: pd.DataFrame) -> Dict[str, LabelMetrics]:
        """
        Calculate metrics for each institution.

        Args:
            df_inst (pd.DataFrame): likelihood for institution

        Returns:
            Dict[str, LabelMetrics]: dictionary of label and its LabelMetrics
            eg. {{label_1: LabelMetrics(), label_2: LabelMetrics(), ...}
        """
        raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])
        inst_metrics = dict()
        for raw_label_name in raw_label_list:
            label_metrics = self.cal_label_metrics(raw_label_name, df_inst)
            inst_metrics[raw_label_name] = label_metrics
        return inst_metrics

    def cal_whole_metrics(self, likelihood_path: Path) -> Dict[str, Dict[str, LabelMetrics]]:
        """
        Calculate metrics for all institutions.

        Args:
            likelihood_path (Path): path to likelihood

        Returns:
            Dict[str, Dict[str, LabelMetrics]]: dictionary of institution and dictionary of label and its LabelMetrics
            eg. {
                instA: {label_1: LabelMetrics(), label_2: LabelMetrics(), ...}, 
                instB: {label_1: LabelMetrics(), label_2: LabelMetrics()}, ...},
                ...}
        """
        df_likelihood = pd.read_csv(likelihood_path)
        whole_metrics = dict()
        for inst in df_likelihood['Institution'].unique():
            df_inst = df_likelihood.query('Institution == @inst')
            whole_metrics[inst] = self._cal_inst_metrics(df_inst)
        return whole_metrics

    def make_summary(
                    self,
                    whole_metrics: Dict[str, Dict[str, LabelMetrics]],
                    datetime: str,
                    likelihood_path: Path,
                    metrics_kind: str
                    ) -> pd.DataFrame:
        """
        Make summary.

        Args:
            whole_metrics (Dict[str, Dict[str, LabelMetrics]]): metrics for all institutions
            datetime (str): date time
            likelihood_path (Path): path to likelihood
            metrics_kind (str): kind of metrics, ie, 'auc', 'r2', or 'c_index'

        Returns:
            pd.DataFrame: summary
        """
        df_summary = pd.DataFrame()
        for inst, inst_metrics in whole_metrics.items():
            _new = dict()
            _new['datetime'] = [datetime]
            _new['weight'] = [likelihood_path.stem.replace('likelihood_', '') + '.pt']
            _new['Institution'] = [inst]
            for raw_label_name, label_metrics in inst_metrics.items():
                _val_metrics = label_metrics.get_label_metrics('val', metrics_kind)
                _test_metrics = label_metrics.get_label_metrics('test', metrics_kind)
                _new[raw_label_name + '_val_' + metrics_kind] = [f"{_val_metrics:.2f}"]
                _new[raw_label_name + '_test_' + metrics_kind] = [f"{_test_metrics:.2f}"]
            df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

        df_summary = df_summary.sort_values('Institution')
        return df_summary

    def print_metrics(self, df_summary: pd.DataFrame, metrics_kind: str) -> None:
        """
        Print metrics.

        Args:
            df_summary (pd.DataFrame): summary
            metrics_kind (str): kind of metrics, ie. 'auc', 'r2', or 'c_index'
        """
        label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])  # [label_1_val, label_1_test, label_2_val, label_2_test, ...]
        num_splits = len(['val', 'test'])
        _column_val_test_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]  # [[label_1_val, label_1_test], [label_2_val, label_2_test], ...]
        for _, row in df_summary.iterrows():
            logger.logger.info(row['Institution'])
            for _column_val_test in _column_val_test_list:
                _label_name = _column_val_test[0].replace('_val', '')
                _label_name_val = _column_val_test[0]
                _label_name_test = _column_val_test[1]
                logger.logger.info(f"{_label_name:<25} val_{metrics_kind}: {row[_label_name_val]:>7}, test_{metrics_kind}: {row[_label_name_test]:>7}")

    def update_summary(self, df_summary: pd.DataFrame) -> None:
        """
        Update summary.

        Args:
            df_summary (pd.DataFrame): summary to be added to the previous summary
        """
        summary_dir = Path('./results/summary')
        summary_path = Path(summary_dir, 'summary.csv')
        if summary_path.exists():
            df_prev = pd.read_csv(summary_path)
            df_updated = pd.concat([df_prev, df_summary], axis=0)
        else:
            summary_dir.mkdir(parents=True, exist_ok=True)
            df_updated = df_summary
        df_updated.to_csv(summary_path, index=False)

    def make_metrics(self, eval_datetime: str, likelihood_path: Path) -> None:
        """
        Make metrics, substantially this method handles everthing all.

        Args:
            eval_datetime (str): date time
            likelihood_path (Path): path to likelihood
        """
        whole_metrics = self.cal_whole_metrics(likelihood_path)
        self.make_save_fig(whole_metrics, eval_datetime, likelihood_path, self.fig_kind)
        df_summary = self.make_summary(whole_metrics, eval_datetime, likelihood_path, self.metrics_kind)
        self.print_metrics(df_summary, self.metrics_kind)
        self.update_summary(df_summary)


class FigROCMixin:
    """
    Class to plot ROC.
    """
    def _plot_fig_inst_metrics(self, inst: str, inst_metrics: Dict[str, LabelMetrics]) -> plt:
        """
        Plot ROC.

        Args:
            inst (str): institution
            inst_metrics (Dict[str, LabelMetrics]): dictionary of label and its LabelMetrics

        Returns:
            plt: ROC
        """
        raw_label_list = inst_metrics.keys()
        num_rows = 1
        num_cols = len(raw_label_list)
        base_size = 7
        height = num_rows * base_size
        width = num_cols * height
        fig = plt.figure(figsize=(width, height))

        for i, raw_label_name in enumerate(raw_label_list):
            label_metrics = inst_metrics[raw_label_name]
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
            ax_i.plot(label_metrics.val.fpr, label_metrics.val.tpr, label=f"AUC_val = {label_metrics.val.auc:.2f}", marker='x')
            ax_i.plot(label_metrics.test.fpr, label_metrics.test.tpr, label=f"AUC_test = {label_metrics.test.auc:.2f}", marker='o')
            ax_i.grid()
            ax_i.legend()
            fig.tight_layout()
        return fig


class FigYYMixin:
    """
    Class to plot YY-graph.
    """
    def _plot_fig_inst_metrics(self, inst: str, inst_metrics: Dict[str, LabelMetrics]) -> plt:
        """
        Plot yy.

        Args:
            inst (str): institution
            inst_metrics (Dict[str, LabelMetrics]): dictionary of label and its LabelMetrics

        Returns:
            plt: YY-graph
        """
        raw_label_list = inst_metrics.keys()
        num_splits = len(['val', 'test'])
        num_rows = 1
        num_cols = len(raw_label_list) * num_splits
        base_size = 7
        height = num_rows * base_size
        width = num_cols * height
        fig = plt.figure(figsize=(width, height))

        for i, raw_label_name in enumerate(raw_label_list):
            label_metrics = inst_metrics[raw_label_name]
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

            y_obs_val = label_metrics.val.y_obs
            y_pred_val = label_metrics.val.y_pred

            y_obs_test = label_metrics.test.y_obs
            y_pred_test = label_metrics.test.y_pred

            # Plot
            color = mcolors.TABLEAU_COLORS
            val_ax.scatter(y_obs_val, y_pred_val, color=color['tab:blue'], label='val')
            test_ax.scatter(y_obs_test, y_pred_test, color=color['tab:orange'], label='test')

            # Draw diagonal line
            y_values_val = np.concatenate([y_obs_val.flatten(), y_pred_val.flatten()])
            y_values_test = np.concatenate([y_obs_test.flatten(), y_pred_test.flatten()])

            y_values_val_min, y_values_val_max, y_values_val_range = np.amin(y_values_val), np.amax(y_values_val), np.ptp(y_values_val)
            y_values_test_min, y_values_test_max, y_values_test_range = np.amin(y_values_test), np.amax(y_values_test), np.ptp(y_values_test)

            val_ax.plot([y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)],
                        [y_values_val_min - (y_values_val_range * 0.01), y_values_val_max + (y_values_val_range * 0.01)], color='red')

            test_ax.plot([y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)],
                         [y_values_test_min - (y_values_test_range * 0.01), y_values_test_max + (y_values_test_range * 0.01)], color='red')

        fig.tight_layout()
        return fig


class FigMixin:
    """
    Class for make and save figure
    This class is for ROC and YY-graph.
    """
    def make_save_fig(self, whole_metrics: Dict[str, Dict[str, LabelMetrics]], datetime: str, likelihood_path: Path, fig_kind: str) -> None:
        """
        Make and save figure.

        Args:
            whole_metrics (Dict[str, Dict[str, LabelMetrics]]): metrics for all institutions
            datetime (str): date time
            likelihood_path (Path): path to likelihood
            fig_kind (str): kind of figure, ie. 'roc' or 'yy'
        """
        for inst, inst_metrics in whole_metrics.items():
            fig = self._plot_fig_inst_metrics(inst, inst_metrics)
            save_dir = Path('./results/sets', datetime, fig_kind)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = Path(save_dir, inst + '_' + fig_kind + '_' + likelihood_path.stem.replace('likelihood_', '') + '.png')  # 'likelihood_weight_epoch-010_best.csv'  -> inst_roc_weight_epoch-010_best.png
            fig.savefig(save_path)
            plt.close()


class ClsEval(ROCMixin, MetricsMixin, FigROCMixin, FigMixin):
    """
    Class for calculation metrics for classification.
    """
    def __init__(self) -> None:
        self.fig_kind = 'roc'
        self.metrics_kind = 'auc'


class RegEval(YYMixin, MetricsMixin, FigYYMixin, FigMixin):
    """
    Class for calculation metrics for regression.
    """
    def __init__(self) -> None:
        self.fig_kind = 'yy'
        self.metrics_kind = 'r2'


class DeepSurvEval(C_IndexMixin, MetricsMixin):
    """
    Class for calculation metrics for DeepSurv.
    """
    def __init__(self) -> None:
        self.fig_kind = None
        self.metrics_kind = 'c_index'

    def make_metrics(self, eval_datetime: str, likelihood_path: Path) -> None:
        """
        Make metrics, substantially this method handles everthing all.

        Args:
            eval_datetime (str): date time
            likelihood_path (Path): path to likelihood

        Orverwrite def make_metrics() in class MetricsMixin by deleteing self.make_save_fig(),
        ie. no need to plot and save figure.
        """
        whole_metrics = self.cal_whole_metrics(likelihood_path)
        df_summary = self.make_summary(whole_metrics, eval_datetime, likelihood_path, self.metrics_kind)
        self.print_metrics(df_summary, self.metrics_kind)
        self.update_summary(df_summary)


def set_eval(task: str) -> Union[ClsEval, RegEval, DeepSurvEval]:
    """
    Set class for evaluation depending on task depending on task.

    Args:
        task (str): task

    Returns:
        Union[ClsEval, RegEval, DeepSurvEval]: class for evaluation
    """
    if task == 'classification':
        return ClsEval()
    elif task == 'regression':
        return RegEval()
    elif task == 'deepsurv':
        return DeepSurvEval()
    else:
        raise ValueError(f"Invalid task: {task}.")
