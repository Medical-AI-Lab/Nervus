#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from typing import Dict, Tuple


class SplitProvider:
    """
    Class to make label for each class and cast tabular data
    """
    def __init__(self, split_path: Path, task: str) -> None:
        """
        Args:
            split_path (Path): path to csv
            task (str): task
        """
        super().__init__()

        self.split_path = split_path
        self.task = task

        _df_source = pd.read_csv(self.split_path)
        _df_source_excluded = _df_source[_df_source['split'] != 'exclude'].copy()
        _df_source_labeled, _class_name_in_raw_label = self._make_labelling(_df_source_excluded, self.task)
        self.df_source = self._cast_csv(_df_source_labeled, self.task)

        self.raw_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith('label')])
        self.internal_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith('internal_label')])
        self.class_name_in_raw_label = _class_name_in_raw_label
        self.num_classes_in_internal_label = self._define_num_classes_in_internal_label(self.df_source, self.task)
        self.input_list = list(self.df_source.columns[self.df_source.columns.str.startswith('input')])
        # self.id_column = list(self.df_source.columns[self.df_source.columns.str.startswith('id')])[0]

        if self.task == 'deepsurv':
            self.period_column = list(self.df_source.columns[self.df_source.columns.str.startswith('period')])[0]
        else:
            self.period_column = None

    # Labeling
    def _make_labelling(self, df_source_excluded: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        Assign a number to the class name within each label

        Args:
            df_source_excluded (DataFrame): DataFrame of csv without 'exlucde'
            task (str): task

        Returns:
            pd.DataFrame: DataFrame with columns assigned a number to the class name within each label,
            Dict[str, int]: Dictionary with numbers assigned to class names within each label
        """
        # Make dict for labeling
        # _class_name_in_raw_label =
        # {'label_1':{'A':0, 'B':1}, 'label_2':{'C':0, 'D':1, 'E':2}, ...}   classification
        # {'label_1':{}, 'label_2':{}, ...}                                  regression
        # {'label_1':{'A':0, 'B':1}}                                         deepsurv, should be 2 classes only
        _df_tmp = df_source_excluded.copy()
        _raw_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('label')])
        _class_name_in_raw_label = {}
        for raw_label_name in _raw_label_list:
            class_list = _df_tmp[raw_label_name].value_counts().index.tolist()  # {'A', 'B', ... } DECENDING ORDER
            _class_name_in_raw_label[raw_label_name] = {}
            if (task == 'classification') or (task == 'deepsurv'):
                for i, ith_class in enumerate(class_list):
                    _class_name_in_raw_label[raw_label_name][ith_class] = i
            else:
                _class_name_in_raw_label[raw_label_name] = {}  # No need of labeling

        # Labeling
        for raw_label_name, class_name_in_raw_label in _class_name_in_raw_label.items():
            _internal_label = raw_label_name.replace('label', 'internal_label')  # label_XXX -> internal_label_XXX
            if (task == 'classification') or (task == 'deepsurv'):
                for class_name, ground_truth in class_name_in_raw_label.items():
                    _df_tmp.loc[_df_tmp[raw_label_name] == class_name, _internal_label] = ground_truth
            else:
                # When regression
                _df_tmp[_internal_label] = _df_tmp[raw_label_name].copy()    # Just copy. Needed because internal_label_XXX will be cast later.

        _df_source_labeled = _df_tmp.copy()
        return _df_source_labeled, _class_name_in_raw_label

    def _define_num_classes_in_internal_label(self, df_source: pd.DataFrame, task: str) -> Dict[str, int]:
        """
        Find thr number of classes for each internal label

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            task (str): task

        Returns:
            Dict[str, int]: Number of classes for each internal label
        """
        # _num_classes_in_internal_label =
        # {internal_label_output_1: 2, internal_label_output_2: 3, ...}   classification
        # {internal_label_output_1: 1, internal_label_output_2: 1, ...}   regression,  should be 1
        # {internal_label_output_1: 1}                                    deepsurv,    should be 1
        _num_classes_in_internal_label = {}
        _internal_label_list = list(df_source.columns[df_source.columns.str.startswith('internal_label')])
        for internal_label_name in _internal_label_list:
            if task == 'classification':
                # Actually _num_classes_in_internal_label can be made from self.class_name_in_raw_label, however
                # it might be natural to count the number of classes in each internal label.
                _num_classes_in_internal_label[internal_label_name] = df_source[internal_label_name].nunique()
            else:
                # When regression or deepsurv
                _num_classes_in_internal_label[internal_label_name] = 1

        return _num_classes_in_internal_label

    # Cast
    def _cast_csv(self, df_source_labeled: pd.DataFrame, task: str) -> pd.DataFrame:
        """
        Cast columns as required by the task

        Args:
            df_source_labeled (pd.DataFrame): DataFrame of labeled csv
            task (str): task

        Returns:
            pd.DataFrame: cast DataFrame od cvs with labeling
        """
        # label_* : int
        # input_* : float
        _df_tmp = df_source_labeled.copy()
        _input_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('input')])
        _internal_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('internal_label')])

        _cast_input_dict = {input: float for input in _input_list}

        if task == 'classification':
            _cast_internal_label_dict = {internal_label: int for internal_label in _internal_label_list}
        else:
            # When regression or deepsurv
            _cast_internal_label_dict = {internal_label: float for internal_label in _internal_label_list}

        _df_tmp = _df_tmp.astype(_cast_input_dict)
        _df_tmp = _df_tmp.astype(_cast_internal_label_dict)
        _df_casted = _df_tmp.copy()
        return _df_casted


def make_split_provider(split_path: Path, task: str) -> SplitProvider:
    """
    Format by making label

    Args:
        split_path (Path): path to csv
        task (str): task

    Returns:
        SplitProvider: Object to DataFrame of labeled csv
    """
    sp = SplitProvider(split_path, task)
    return sp
