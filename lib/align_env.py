#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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
    dirs_dict['yy'] = os.path.join(results_dir, 'yy')
    dirs_dict['visualization'] = os.path.join(results_dir, 'visualization')

    return dirs_dict



def parse_csv(csv_path):
    csv_dict = {}
    prefix_id = 'id_'
    prefix_label = 'label_'
    prefix_input = 'input_'

    csv_dict['filename_column'] = 'finename'
    csv_dict['dir_to_img_column'] = 'dir_to_img'
    csv_dict['split_column'] = 'split'

    df_source = pd.read_csv(csv_path)
    column_names = list(df_source.columns)

    csv_dict['id_column'] = [ column_name for column_name in column_names if column_name.startswith(prefix_id) ][0]
    
    csv_dict['label_list'] = [ column_name for column_name in column_names if column_name.startswith(prefix_label) ]
    csv_dict['label_name'] = csv_dict['label_list'][0]
    csv_dict['class_names'] = [ str(i) for i in df_source[csv_dict['label_name']].unique() ]  # must be ['0', '1']
    csv_dict['num_classes'] = len(csv_dict['class_names'])                                    # must be 2, when classification

    csv_dict['input_list'] = [ column_name for column_name in column_names if column_name.startswith(prefix_input) ]
    csv_dict['num_inputs'] = len(csv_dict['input_list'])

    # Cast
    # label_* : int
    # input_* : float
    cast_input_dict = { input: float for input in csv_dict['input_list'] }
    cast_label_dict = { label: int for label in csv_dict['label_list'] }
    df_source = df_source.astype(cast_input_dict)
    df_source = df_source.astype(cast_label_dict)
    csv_dict['source'] = df_source

    return csv_dict

# ----- EOF -----
