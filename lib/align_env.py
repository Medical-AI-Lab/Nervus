#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd


def set_dirs():
    dirs_dict = {}

    data_root = '../materials'
    dirs_dict['csvs_dir'] = os.path.join(data_root, 'csvs')
    dirs_dict['images_dir'] = os.path.join(data_root, 'images')

    dirs_dict['train_opt_log'] = './train_opt_logs'
    dirs_dict['weight'] = './weights'

    results_dir = './results'
    dirs_dict['learning_curve'] = os.path.join(results_dir, 'learning_curve')
    dirs_dict['likelihood'] = os.path.join(results_dir, 'likelihood')
    dirs_dict['roc'] = os.path.join(results_dir, 'roc')
    dirs_dict['roc_summary'] = os.path.join(dirs_dict['roc'], 'summary')
    dirs_dict['yy'] = os.path.join(results_dir, 'yy')
    dirs_dict['yy_summary'] = os.path.join(dirs_dict['yy'], 'summary')
    dirs_dict['visualization'] = os.path.join(results_dir, 'visualization')
    return dirs_dict


def parse_csv(csv_path, task):
    csv_dict = {}
    prefix_id = 'id_'
    prefix_label = 'label_'
    prefix_input = 'input_'
    prefix_period = 'periods_'

    csv_dict['filename_column'] = 'filename'
    csv_dict['dir_to_img_column'] = 'dir_to_img'
    csv_dict['split_column'] = 'split'

    df_source = pd.read_csv(csv_path)
    df_source = df_source[df_source[csv_dict['split_column']] != 'exclude']
    column_names = list(df_source.columns)

    csv_dict['id_column'] = [column_name for column_name in column_names if column_name.startswith(prefix_id)][0]   # should be one
    csv_dict['label_list'] = [column_name for column_name in column_names if column_name.startswith(prefix_label)]
    csv_dict['input_list'] = [column_name for column_name in column_names if column_name.startswith(prefix_input)]
    csv_dict['num_inputs'] = len(csv_dict['input_list'])

    if task == 'classification':
        csv_dict['num_classes'] = 2
    elif task == 'regression':
        csv_dict['num_classes'] = 1
    else:
        # When deepsurv
        csv_dict['period_column'] = [column_name for column_name in column_names if column_name.startswith(prefix_period)][0]   # should be one
        csv_dict['num_classes'] = 1

    # Cast
    # label_* : int
    # input_* : float
    cast_input_dict = {input: float for input in csv_dict['input_list']}
    if task == 'classification':
        cast_label_dict = {label: int for label in csv_dict['label_list']}
    else:
        cast_label_dict = {label: float for label in csv_dict['label_list']}
    df_source = df_source.astype(cast_input_dict)
    df_source = df_source.astype(cast_label_dict)
    csv_dict['source'] = df_source
    return csv_dict

# ----- EOF -----
