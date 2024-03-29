#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from .logger import BaseLogger
from typing import Dict, Union


logger = BaseLogger.get_logger(__name__)


class MetricsData:
    """
    Class for storing metrics.
    Metrics are defined depending on task.

    For ROC
        self.fpr: np.ndarray
        self.tpr: np.ndarray
        self.auc: float

    For Regression
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
        Metrics of each split

        eg.
        self.train = MetricsData()
        self.val = MetricsData()
        self.test = MetricsData()
        """
        pass

    def set_label_metrics(self, split: str, attr: str, value: Union[np.ndarray, float]) -> None:
        """
        Set value as appropriate metrics of split.

        Args:
            split (str): split
            attr (str): attribute name as follows:
                        classification: 'fpr', 'tpr', or 'auc',
                        regression:     'y_obs'(ground truth), 'y_pred'(prediction) or 'r2'
                        deepsurv:       'c_index'
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

    def _cal_label_roc_binary(self, label_name: str, df_group: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC for binary class.

        Args:
            label_name (str): label name
            df_group (pd.DataFrame): likelihood for group

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = [column_name for column_name in df_group.columns if label_name in column_name] + ['split']
        df_label = df_group[required_columns]
        POSITIVE = 1
        positive_pred_name = 'pred_' + label_name + '_' + str(POSITIVE)

        label_metrics = LabelMetrics()
        for split in df_group['split'].unique():
            setattr(label_metrics, split, MetricsData())
            df_split = df_label.query('split == @split')
            y_true = df_split[label_name]
            y_score = df_split[positive_pred_name]
            _fpr, _tpr, _ = metrics.roc_curve(y_true, y_score)
            self._set_roc(label_metrics, split, _fpr, _tpr)
        return label_metrics

    def _cal_label_roc_multi(self, label_name: str, df_group: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC for multi-class by macro average.

        Args:
            label_name (str): label name
            df_group (pd.DataFrame): likelihood for group

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = [column_name for column_name in df_group.columns if label_name in column_name] + ['split']
        df_label = df_group[required_columns]

        pred_name_list = list(df_label.columns[df_label.columns.str.startswith('pred')])
        class_list = [int(pred_name.rsplit('_', 1)[-1]) for pred_name in pred_name_list]  # [pred_label_0, pred_label_1, pred_label_2] -> [0, 1, 2]
        num_classes = len(class_list)

        label_metrics = LabelMetrics()
        for split in df_group['split'].unique():
            setattr(label_metrics, split, MetricsData())
            df_split = df_label.query('split == @split')
            y_true = df_split[label_name]
            y_true_bin = label_binarize(y_true, classes=class_list)  # Since y_true: List[int], should be class_list: List[int]

            # Compute ROC for each class by OneVsRest
            _fpr = dict()
            _tpr = dict()
            for i, class_name in enumerate(class_list):
                pred_name = 'pred_' + label_name + '_' + str(class_name)
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

    def cal_label_metrics(self, label_name: str, df_group: pd.DataFrame) -> LabelMetrics:
        """
        Calculate ROC and AUC for label depending on binary or multi-class.

        Args:
            label_name (str):label name
            df_group (pd.DataFrame): likelihood for group

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        pred_name_list = df_group.columns[df_group.columns.str.startswith('pred_' + label_name)]
        isMultiClass = (len(pred_name_list) > 2)
        if isMultiClass:
            label_metrics = self._cal_label_roc_multi(label_name, df_group)
        else:
            label_metrics = self._cal_label_roc_binary(label_name, df_group)
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

    def cal_label_metrics(self, label_name: str, df_group: pd.DataFrame) -> LabelMetrics:
        """
        Calculate YY and R2 for label.

        Args:
            label_name (str): label name
            df_group (pd.DataFrame): likelihood for group

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = [column_name for column_name in df_group.columns if label_name in column_name] + ['split']
        df_label = df_group[required_columns]
        label_metrics = LabelMetrics()
        for split in df_group['split'].unique():
            setattr(label_metrics, split, MetricsData())
            df_split = df_label.query('split == @split')
            y_obs = df_split[label_name]
            y_pred = df_split['pred_' + label_name]
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
                    labels: pd.Series
                    ) -> None:
        """
        Set C-Index.

        Args:
            label_metrics (LabelMetrics): metrics of 'val' and 'test'
            split (str): 'val' or 'test'
            periods (pd.Series): periods
            preds (pd.Series): prediction
            labels (pd.Series): label

        self.metrics_kind = 'c_index' is defined in class DeepSurvEval below.
        """
        from lifelines.utils import concordance_index
        value_c_index = concordance_index(periods, (-1)*preds, labels)
        label_metrics.set_label_metrics(split, self.metrics_kind, value_c_index)

    def cal_label_metrics(self, label_name: str, df_group: pd.DataFrame) -> LabelMetrics:
        """
        Calculate C-Index for label.

        Args:
            label_name (str): label name
            df_group (pd.DataFrame): likelihood for group

        Returns:
            LabelMetrics: metrics of 'val' and 'test'
        """
        required_columns = [column_name for column_name in df_group.columns if label_name in column_name] + ['periods', 'split']
        df_label = df_group[required_columns]
        label_metrics = LabelMetrics()
        for split in df_group['split'].unique():
            setattr(label_metrics, split, MetricsData())
            df_split = df_label.query('split == @split')
            periods = df_split['periods']
            preds = df_split['pred_' + label_name]
            labels = df_split[label_name]
            self._set_c_index(label_metrics, split, periods, preds, labels)
        return label_metrics


class MetricsMixin:
    """
    Class to calculate metrics and make summary.
    """
    def _cal_group_metrics(self, df_group: pd.DataFrame) -> Dict[str, LabelMetrics]:
        """
        Calculate metrics for each group.

        Args:
            df_group (pd.DataFrame): likelihood for group

        Returns:
            Dict[str, LabelMetrics]: dictionary of label and its LabelMetrics
            eg. {{label_1: LabelMetrics(), label_2: LabelMetrics(), ...}
        """
        label_list = list(df_group.columns[df_group.columns.str.startswith('label')])
        group_metrics = dict()
        for label_name in label_list:
            label_metrics = self.cal_label_metrics(label_name, df_group)
            group_metrics[label_name] = label_metrics
        return group_metrics

    def cal_whole_metrics(self, df_likelihood: pd.DataFrame) -> Dict[str, Dict[str, LabelMetrics]]:
        """
        Calculate metrics for all groups.

        Args:
            df_likelihood (pd.DataFrame) : DataFrame of likelihood

        Returns:
            Dict[str, Dict[str, LabelMetrics]]: dictionary of group and dictionary of label and its LabelMetrics
            eg. {
                groupA: {label_1: LabelMetrics(), label_2: LabelMetrics(), ...},
                groupB: {label_1: LabelMetrics(), label_2: LabelMetrics()}, ...},
                ...}
        """
        whole_metrics = dict()
        for group in df_likelihood['group'].unique():
            df_group = df_likelihood.query('group == @group')
            whole_metrics[group] = self._cal_group_metrics(df_group)
        return whole_metrics

    def make_summary(
                    self,
                    whole_metrics: Dict[str, Dict[str, LabelMetrics]],
                    likelihood_path: Path,
                    metrics_kind: str
                    ) -> pd.DataFrame:
        """
        Make summary.

        Args:
            whole_metrics (Dict[str, Dict[str, LabelMetrics]]): metrics for all groups
            likelihood_path (Path): path to likelihood
            metrics_kind (str): kind of metrics, ie, 'auc', 'r2', or 'c_index'

        Returns:
            pd.DataFrame: summary
        """
        _datetime = likelihood_path.parents[1].name
        _weight = likelihood_path.stem.replace('likelihood_', '') + '.pt'
        df_summary = pd.DataFrame()
        for group, group_metrics in whole_metrics.items():
            _df_new_group = pd.DataFrame({
                                        'datetime': [_datetime],
                                        'weight': [ _weight],
                                        'group': [group]
                                        })

            for label_name, label_metrics in group_metrics.items():
                # val
                if hasattr(label_metrics, 'val'):
                    _val_metrics = label_metrics.get_label_metrics('val', metrics_kind)
                    _str_val_metrics = f"{_val_metrics:.2f}"
                else:
                    _str_val_metrics = 'No data'

                # test
                if hasattr(label_metrics, 'test'):
                    _test_metrics = label_metrics.get_label_metrics('test', metrics_kind)
                    _str_test_metrics = f"{_test_metrics:.2f}"
                else:
                    _str_test_metrics = 'No data'

                _df_new_group[label_name + '_val_' + metrics_kind] = _str_val_metrics
                _df_new_group[label_name + '_test_' + metrics_kind] = _str_test_metrics

            df_summary = pd.concat([df_summary, _df_new_group], ignore_index=True)
        return df_summary

    def print_metrics(self, df_summary: pd.DataFrame, metrics_kind: str) -> None:
        """
        Print metrics.

        Args:
            df_summary (pd.DataFrame): summary
            metrics_kind (str): kind of metrics, ie. 'auc', 'r2', or 'c_index'
        """
        label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])  # [label_1_val_auc, label_1_test_auc, label_2_val_auc, label_2_test_auc, ...]
        num_splits = len(['val', 'test'])
        _column_val_test_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]  # [[label_1_val, label_1_test], [label_2_val, label_2_test], ...]
        for _, row in df_summary.iterrows():
            logger.info(row['group'])
            for _column_val_test in _column_val_test_list:
                _label_name = _column_val_test[0].replace('_val', '')
                _label_name_val = _column_val_test[0]
                _label_name_test = _column_val_test[1]
                logger.info(f"{_label_name:<25} val_{metrics_kind}: {row[_label_name_val]:>7}, test_{metrics_kind}: {row[_label_name_test]:>7}")

    def update_summary(self, df_summary: pd.DataFrame, likelihood_path: Path) -> None:
        """
        Update summary.

        Args:
            df_summary (pd.DataFrame): summary to be added to the previous summary
            likelihood_path (Path): path to likelihood
        """
        _project_dir = likelihood_path.parents[3]
        summary_dir = Path(_project_dir, 'summary')
        summary_path = Path(summary_dir, 'summary.csv')
        if summary_path.exists():
            df_prev = pd.read_csv(summary_path)
            df_updated = pd.concat([df_prev, df_summary], axis=0)
        else:
            summary_dir.mkdir(parents=True, exist_ok=True)
            df_updated = df_summary
        df_updated.to_csv(summary_path, index=False)

    def make_metrics(self, likelihood_path: Path) -> None:
        """
        Make metrics.

        Args:
            likelihood_path (Path): path to likelihood
        """
        df_likelihood = pd.read_csv(likelihood_path)
        whole_metrics = self.cal_whole_metrics(df_likelihood)
        self.make_save_fig(whole_metrics, likelihood_path, self.fig_kind)
        df_summary = self.make_summary(whole_metrics, likelihood_path, self.metrics_kind)
        self.print_metrics(df_summary, self.metrics_kind)
        self.update_summary(df_summary, likelihood_path)


class FigROCMixin:
    """
    Class to plot ROC.
    """
    def _plot_fig_group_metrics(self, group: str, group_metrics: Dict[str, LabelMetrics]) -> plt:
        """
        Plot ROC.

        Args:
            group (str): group
            group_metrics (Dict[str, LabelMetrics]): dictionary of label and its LabelMetrics

        Returns:
            plt: ROC
        """
        label_list = group_metrics.keys()
        num_rows = 1
        num_cols = len(label_list)
        base_size = 7
        height = num_rows * base_size
        width = num_cols * height
        fig = plt.figure(figsize=(width, height))

        for i, label_name in enumerate(label_list):
            label_metrics = group_metrics[label_name]
            offset = i + 1
            ax_i = fig.add_subplot(
                                    num_rows,
                                    num_cols,
                                    offset,
                                    title=group + ': ' + label_name,
                                    xlabel='1 - Specificity',
                                    ylabel='Sensitivity',
                                    xmargin=0,
                                    ymargin=0
                                    )

            # val
            if hasattr(label_metrics, 'val'):
                _fpr = label_metrics.get_label_metrics('val', 'fpr')
                _tpr = label_metrics.get_label_metrics('val', 'tpr')
                _auc = label_metrics.get_label_metrics('val', 'auc')
                ax_i.plot(_fpr, _tpr, label=f"AUC_val = {_auc:.2f}", marker='x')

            # test
            if hasattr(label_metrics, 'test'):
                _fpr = label_metrics.get_label_metrics('test', 'fpr')
                _tpr = label_metrics.get_label_metrics('test', 'tpr')
                _auc = label_metrics.get_label_metrics('test', 'auc')
                ax_i.plot(_fpr, _tpr, label=f"AUC_test = {_auc:.2f}", marker='o')

            ax_i.grid()
            ax_i.legend()
            fig.tight_layout()
        return fig


class FigYYMixin:
    """
    Class to plot YY-graph.
    """
    def _plot_fig_split(
                        self,
                        group: str,
                        label_metrics: LabelMetrics,
                        label_name: str,
                        split: str,
                        fig: plt,
                        color: Dict[str, str],
                        num_rows: int,
                        num_cols: int,
                        offset: int
                        ) -> plt:
        """
        Plot yy of split.

        Args:
            group (str): group
            label_metrics (LabelMetrics): label metrics
            label_name (str): label name
            split (str): split
            fig (plt): figure
            color (Dict[str, str]): color palette
            num_rows (int): number of rows
            num_cols (int): number of columns
            offset (int): offset

        Returns:
            plt: yy of split
        """
        split_ax = fig.add_subplot(
                            num_rows,
                            num_cols,
                            offset,
                            title=group + ': ' + label_name + '\n' + split + ': Observed-Predicted Plot',
                            xlabel='Observed',
                            ylabel='Predicted',
                            xmargin=0,
                            ymargin=0
                            )

        split_y_obs = label_metrics.get_label_metrics(split, 'y_obs')
        split_y_pred = label_metrics.get_label_metrics(split, 'y_pred')

        # Plot
        split_ax.scatter(split_y_obs, split_y_pred, color=color['tab:blue'], label=split)

        # Draw diagonal line
        split_y_values = np.concatenate([split_y_obs.flatten(), split_y_pred.flatten()])
        split_y_values_min = np.amin(split_y_values)
        split_y_values_max = np.amax(split_y_values)
        split_y_values_range = np.ptp(split_y_values)
        split_ax.plot(
                    [split_y_values_min - (split_y_values_range * 0.01), split_y_values_max + (split_y_values_range * 0.01)],
                    [split_y_values_min - (split_y_values_range * 0.01), split_y_values_max + (split_y_values_range * 0.01)],
                    color='red'
                    )
        return fig

    def _plot_fig_group_metrics(self, group: str, group_metrics: Dict[str, LabelMetrics]) -> plt:
        """
        Plot yy.

        Args:
            group (str): group
            group_metrics (Dict[str, LabelMetrics]): dictionary of label and its LabelMetrics

        Returns:
            plt: YY-graph
        """
        label_list = group_metrics.keys()
        num_rows = 1
        base_size = 7
        height = num_rows * base_size
        color = mcolors.TABLEAU_COLORS
        fig = None

        for i, label_name in enumerate(label_list):
            label_metrics = group_metrics[label_name]
            num_splits = len(vars(label_metrics).keys())
            num_cols = len(label_list) * num_splits
            width = height * num_cols

            if fig is None:
                # create new figure
                fig = plt.figure(figsize=(width, height))
            else:
                # No need to create new figure, ie. overwrite figure
                pass

            _base_offset = (i * num_splits) + 1

            # val
            if hasattr(label_metrics, 'val'):
                val_offset =  _base_offset
                fig = self._plot_fig_split(
                                    group,
                                    label_metrics,
                                    label_name,
                                    'val',
                                    fig,
                                    color,
                                    num_rows,
                                    num_cols,
                                    val_offset
                                    )

            # test
            if hasattr(label_metrics, 'test'):
                test_offset = _base_offset + 1 if hasattr(label_metrics, 'val') else _base_offset
                fig = self._plot_fig_split(
                                            group,
                                            label_metrics,
                                            label_name,
                                            'test',
                                            fig,
                                            color,
                                            num_rows,
                                            num_cols,
                                            test_offset
                                            )

        fig.tight_layout()
        return fig


class FigMixin:
    """
    Class for make and save figure
    This class is for ROC and YY-graph.
    """
    def make_save_fig(self, whole_metrics: Dict[str, Dict[str, LabelMetrics]], likelihood_path: Path, fig_kind: str) -> None:
        """
        Make and save figure.

        Args:
            whole_metrics (Dict[str, Dict[str, LabelMetrics]]): metrics for all groups
            likelihood_path (Path): path to likelihood
            fig_kind (str): kind of figure, ie. 'roc' or 'yy'
        """
        _datetime_dir = likelihood_path.parents[1]
        save_dir = Path(_datetime_dir, fig_kind)
        save_dir.mkdir(parents=True, exist_ok=True)
        _fig_name = fig_kind + '_' + likelihood_path.stem.replace('likelihood_', '')
        for group, group_metrics in whole_metrics.items():
            fig = self._plot_fig_group_metrics(group, group_metrics)
            save_path = Path(save_dir, group + '_' + _fig_name + '.png')
            fig.savefig(save_path)
            plt.close()


class ClsEval(MetricsMixin, ROCMixin, FigMixin, FigROCMixin):
    """
    Class for calculation metrics for classification.
    """
    def __init__(self) -> None:
        self.fig_kind = 'roc'
        self.metrics_kind = 'auc'


class RegEval(MetricsMixin, YYMixin, FigMixin, FigYYMixin):
    """
    Class for calculation metrics for regression.
    """
    def __init__(self) -> None:
        self.fig_kind = 'yy'
        self.metrics_kind = 'r2'


class DeepSurvEval(MetricsMixin, C_IndexMixin):
    """
    Class for calculation metrics for DeepSurv.
    """
    def __init__(self) -> None:
        self.fig_kind = None
        self.metrics_kind = 'c_index'

    def make_metrics(self, likelihood_path: Path) -> None:
        """
        Make metrics, substantially this method handles everything all.

        Args:
            likelihood_path (Path): path to likelihood

        Overwrite def make_metrics() in class MetricsMixin by deleting self.make_save_fig(),
        because of no need to plot and save figure.
        """
        df_likelihood = pd.read_csv(likelihood_path)
        whole_metrics = self.cal_whole_metrics(df_likelihood)
        df_summary = self.make_summary(whole_metrics, likelihood_path, self.metrics_kind)
        self.print_metrics(df_summary, self.metrics_kind)
        self.update_summary(df_summary, likelihood_path)


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
