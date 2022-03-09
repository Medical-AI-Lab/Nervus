#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import dataclasses


@dataclasses.dataclass
class NervusEnv:
    dataroot: str = '../materials'
    splits_dir: str = os.path.join(dataroot, 'splits')
    images_dir: str = os.path.join(dataroot, 'images')
    reslts_dir: str = './results'
    sets_dir: str = os.path.join(reslts_dir, 'sets')
    weight: str = 'weight.pt'
    csv_parameters: str = 'perparameter.csv'
    csv_learning_curve: str = 'learning_curve.csv'
    csv_likelihood: str = 'likelihood.csv'
    roc: str = 'roc.png'
    yy: str = 'yy.png'
    csv_c_index: str = 'c_index.csv'
    visualization_dir: str = 'visualization'
    summary_dir: str = os.path.join(reslts_dir, 'summary')
    csv_summary: str = 'summary.csv'


class SplitProvider:
    def __init__(self, split_path, task):
        super().__init__()

        self.split_path = split_path
        self.task = task

        self.prefix_id = 'id'
        self.prefix_input = 'input'
        self.prefix_label = 'label'
        self.prefix_internal_label = 'internal'
        self.prefix_period = 'periods'

        self.filepath_column = 'filepath'
        self.split_column = 'split'

        _df_source = pd.read_csv(self.split_path)
        _df_source = _df_source[_df_source[self.split_column] != 'exclude'].copy()
        _df_source_labeled, self.class_name_in_label = self._make_labelling(_df_source, self.task)   # Labeling

        # cast
        self.df_source = self._cast_csv(_df_source_labeled, task)
        self.label_num_classes = self._define_label_num_classes(self.df_source, self.task)    # After labeling, column names are fixed.

        self.label_list = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_label)])
        self.internal_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_internal_label)])
        self.input_list = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_input)])
        self.num_inputs = len(self.input_list)

        if self.task == 'deepsurv':
            self.period_column = list(self.df_source.columns[self.df_source.columns.str.startswith(self.prefix_period)])[0]  # should be one
        else:
            self.period_column = None


    # Labeling
    def _make_labelling(self, df_source, task):
        # Make dict for labeling
        # _class_name_in_label =
        # {'label_1':{'A':0, 'B':1}, 'label_2':{'C':0, 'D':1, 'E':2}, ...}   classification
        # {'label_1':{}, 'label_2':{}, ...}                                  regression
        # {'label_1':{'A':0, 'B':1}}                                         deepsurv
        _df_tmp = df_source.copy()
        _label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith(self.prefix_label)])
        _class_name_in_label = {}
        for label_name in _label_list:
            class_list = _df_tmp[label_name].value_counts().index.tolist()    # {'A', 'B', ... } decending order
            _class_name_in_label[label_name] = {}
            if (task == 'classification') or (task == 'deepsurv'):
                 for i in range(len(class_list)):
                    _class_name_in_label[label_name][class_list[i]] = i
            else:
                _class_name_in_label[label_name] = {}                         # No need of labeling

        # Labeling
        for label_name, class_name_in_label in _class_name_in_label.items():
            _internal_label = self.prefix_internal_label + '_' + label_name      # label_XXX -> internal_label_XXX
            if (task == 'classification') or (task == 'deepsurv'):
                for class_name, ground_truth in class_name_in_label.items():
                    _df_tmp.loc[_df_tmp[label_name]==class_name, _internal_label] = ground_truth
            else:
                _df_tmp[_internal_label] = _df_tmp[label_name]    # Just copy. Needed because internal_label_XXX will be cast later.

        return _df_tmp, _class_name_in_label


    def _define_label_num_classes(self, df_source, task):
        # _label_num_classes
        # {label_output_1: 2, label_output_2: 3, ...}   classification
        # {label_output_1: 1, label_output_2: 1, ...}   regression,  should be 1
        # {label_output_1: 1}                           deepsurv,    should be 1
        _label_num_classes = {}
        _label_list = list(df_source.columns[df_source.columns.str.startswith(self.prefix_label)])
        for label_name in _label_list:
            if task == 'classification':
                _label_num_classes[label_name] = df_source[label_name].nunique()
            else:
                _label_num_classes[label_name] = 1

        return _label_num_classes


    # Cast
    def _cast_csv(self, df_source, task):
        # label_* : int
        # input_* : float
        _df_tmp = df_source.copy()
        _input_list = list(df_source.columns[df_source.columns.str.startswith(self.prefix_input)])
        _internal_label_list = list(df_source.columns[df_source.columns.str.startswith(self.prefix_internal_label)])

        _cast_input_dict = {input: float for input in _input_list}
        
        if task == 'classification':
            _cast_internal_label_dict = {label: int for label in _internal_label_list}
        else:
            # When regression or deepsurv
            _cast_internal_label_dict = {label: float for label in _internal_label_list}

        _df_tmp = _df_tmp.astype(_cast_input_dict)
        _df_tmp = _df_tmp.astype(_cast_internal_label_dict)

        return _df_tmp

