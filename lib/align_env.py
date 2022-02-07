#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import dataclasses


@dataclasses.dataclass
class NervusEnv:
    dataroot: str = '../materials'
    csvs_dir: str = os.path.join(dataroot, 'csvs')
    images_dir: str = os.path.join(dataroot, 'images')
    reslts_dir: str = './results'
    sets_dir: str = os.path.join(reslts_dir, 'sets')
    weight: str = 'weight.pt'
    csv_hyperparameters: str = 'hyperparameters.csv'
    csv_learning_curve: str = 'learning_curve.csv'
    csv_likelihood: str = 'likelihood.csv'
    roc: str = 'roc.png'
    yy: str = 'yy.png'
    csv_c_index: str = 'c_index.csv'
    visualization_dir: str = 'visualization'
    summary_dir: str = os.path.join(reslts_dir, 'summary')
    csv_summary: str = 'summary.csv'


def parse_csv(csv_path, task):
    csv_dict = {} 
    prefix_id = 'id'  # 'id'
    prefix_input = 'input'
    prefix_output = 'output'
    prefix_label = 'label'
    prefix_period = 'periods'

    csv_dict['filepath_column'] = 'filepath'
    csv_dict['split_column'] = 'split'

    # Exclude
    df_source = pd.read_csv(csv_path)
    df_source = df_source[df_source[csv_dict['split_column']] != 'exclude']

    # Make dict for labeling
    # csv_dict['output_class_label'] =
    # {'output_1':{'A':0, 'B':1}, 'output_2':{'C':0, 'D':1, 'E':2}, ...}   classification
    # {'output_1':{}, 'output_2':{}, ...}                                  regression 
    # {'output_1':{'A':0, 'B':1}}                                          deepsurv
    csv_dict['output_list'] = [column_name for column_name in list(df_source.columns) if column_name.startswith(prefix_output)]
    output_class_label_dict = {}
    for output_name in csv_dict['output_list']:
        class_list = df_source[output_name].value_counts().index.tolist()
        output_class_label_dict[output_name] = {}
        if (task == 'classification') or (task == 'deepsurv'):
             for i in range(len(class_list)):
                output_class_label_dict[output_name][class_list[i]] = i
        else:
            output_class_label_dict[output_name] = {}   # No need of labeling
    csv_dict['output_class_label'] = output_class_label_dict

    # Labeling
    for output_name, class_label_dict in output_class_label_dict.items():
        if (task == 'classification') or (task == 'deepsurv'):
            for class_name, label in class_label_dict.items():
                df_source.loc[df_source[output_name]==class_name, (prefix_label + '_' + output_name)] = label
        else:
            df_source[(prefix_label + '_' + output_name)] = df_source[output_name]   # Just copy. Needed because labeL_XXX will be cast later.

    # After labeling, column names are fixed.
    column_names = list(df_source.columns)

    # Define the number of outputs of mode
    # csv_dict['label_num_classes'] =
    # {label_output_1: 2, label_output_2: 3, ...}   classification
    # {label_output_1: 1, label_output_2: 1, ...}   regression,  should be 1
    # {label_output_1: 1}                           deepsurv,    should be 1
    csv_dict['label_list'] = [column_name for column_name in column_names if column_name.startswith(prefix_label)]
    csv_dict['label_num_classes'] = {}
    for label_name in csv_dict['label_list']:
        if task == 'classification':
            csv_dict['label_num_classes'][label_name] = df_source[label_name].nunique()
        else:
            # When regression or deepsurv
            csv_dict['label_num_classes'][label_name] = 1

    csv_dict['id_column'] = [column_name for column_name in column_names if column_name.startswith(prefix_id)][0]   # should be one
    csv_dict['input_list'] = [column_name for column_name in column_names if column_name.startswith(prefix_input)]
    csv_dict['num_inputs'] = len(csv_dict['input_list'])

    if task == 'deepsurv':
        csv_dict['period_column'] = [column_name for column_name in column_names if column_name.startswith(prefix_period)][0]   # should be one
    else:
        csv_dict['period_column'] = None

    # Cast
    # label_* : int
    # input_* : float
    cast_input_dict = {input: float for input in csv_dict['input_list']}
    if task == 'classification':
        cast_label_dict = {label: int for label in csv_dict['label_list']}
    else:
        # When regression or deepsurv
        cast_label_dict = {label: float for label in csv_dict['label_list']}
    df_source = df_source.astype(cast_input_dict)
    df_source = df_source.astype(cast_label_dict)

    csv_dict['source'] = df_source
    return csv_dict

# ----- EOF -----
