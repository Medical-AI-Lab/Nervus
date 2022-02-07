#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
import torch


# Even if GPU available, set CPU as device when specifying CPU.
def set_device(gpu_ids):
    if gpu_ids:
        assert torch.cuda.is_available(), 'No avalibale GPU on this machine. Use CPU.'
        primary_gpu_id = gpu_ids[0]
        device_name = f'cuda:{primary_gpu_id}'
        device = torch.device(device_name)
    else:
        device = torch.device('cpu')
    return device


def get_column_value(df, column_name:str, value_list:list):
    assert (value_list!=[]), 'The list of values is empty list.' #  ie. When value_list==[], raise AssertionError.
    df_result = pd.DataFrame([])
    for value in value_list:
        df_tmp = df[df[column_name]==value]
        df_result = df_result.append(df_tmp, ignore_index=True)
    return df_result


def get_target(source_dir, target):
    if (target is None):
        dirs = glob.glob(source_dir + '/*')
        if dirs:
            target_dir = sorted(dirs, key=lambda f: os.stat(f).st_mtime, reverse=True)[0]   # latest
        else:
            target_dir = None
            print(f"No directory in {source_dir}") 
    else:
        target_dir = os.path.join(source_dir, target)
        if os.path.isdir(target_dir):
            target_dir = target_dir
        else:
            target_dir = None
            print(f"No such a directory: {target_dir}")
    return target_dir


def read_train_hyperparameters(hyperparameters_path):
    df_hyperparameters = pd.read_csv(hyperparameters_path, index_col=0)
    df_hyperparameters = df_hyperparameters.fillna(np.nan).replace([np.nan],[None])
    hyperparameters_dict = df_hyperparameters.to_dict()['value']
    return hyperparameters_dict


def str2int(gpu_ids_str:str):
    gpu_ids_str = gpu_ids_str.replace('[', '').replace(']', '')
    if gpu_ids_str == '':
        gpu_ids = []
    else:
        gpu_ids = gpu_ids_str.split(',')
        gpu_ids = [ int(i) for i in gpu_ids ]
    return gpu_ids


def update_summary(summary_dir, summary,  df_summary_new):
    summary_path = os.path.join(summary_dir, summary)
    if os.path.isfile(summary_path):
        df_summary = pd.read_csv(summary_path, dtype=str)
        df_summary_updated = pd.concat([df_summary, df_summary_new], axis=0)
    else:
        os.makedirs(summary_dir, exist_ok=True)
        df_summary_updated = df_summary_new
    df_summary_updated.to_csv(summary_path, index=False)


# ----- EOF -----
