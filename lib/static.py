#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd


# Project name
project_name = 'CovidPrognosis'

# project_task must be specified.
# project_task = 'classification' | 'regression' | 'multi_classification' | 'multi_regression'
project_task = 'classification'


project_root = '.'


data_root = '../../../Implement/covid_prognosis'


# Source CSV
csv_path = os.path.join(data_root, 'COVID', 'csv', 'covid_last_status_fillna_split.csv')


# Images
image_dirs = {
               'covid': os.path.join(data_root, 'COVID')
             }



# Directory for save
train_opt_log_dir = os.path.join(project_root, 'train_opt_logs')
weight_dir = os.path.join(project_root, 'weights')
learning_curve_dir = os.path.join(project_root, 'results', 'learning_curve')
likelihood_dir = os.path.join(project_root, 'results', 'likelihood')
roc_dir = os.path.join(project_root, 'results', 'roc')
yy_dir = os.path.join(project_root, 'results', 'yy')
visualization_dir = os.path.join(project_root, 'results', 'visualization')


# Auxiliary
df_source = pd.read_csv(csv_path)       # {PID: str, ....}  # mixed both int and float
column_names = list(df_source.columns)

# Objective variable
label_list = [ column_name for column_name in column_names if column_name.startswith('label_') ]

# Explanatory variables
input_list = [ column_name for column_name in column_names if column_name.startswith('input_') ]
input_list_normed = [ 'normed_' +  input for input in input_list ]


# NOTE
# label_* : int
# input_* : float
cast_input_dict = { input: float for input in input_list }
cast_label_dict = { label: int for label in label_list }
df_source = df_source.astype(cast_input_dict)
df_source = df_source.astype(cast_label_dict)


label_name = label_list[0]                                           # should be one.  ['label_1']
class_names = [ str(i) for i in df_source[label_name].unique() ]     # must be ['0', '1']
num_classes = len(class_names)                                       # 2
input_list = input_list



# ----- EOF -----
