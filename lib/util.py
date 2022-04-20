#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import sys
import os
import glob

import numpy as np
import pandas as pd
import torch

class NervusLogger:
    _unexecuted_configure = True

    @classmethod
    def get_logger(cls, filename):
        if cls._unexecuted_configure:
            cls._init_logger()

        return logging.getLogger('nervus.{}'.format(filename))

    @classmethod
    def set_level(cls, level):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(level)

    @classmethod
    def _init_logger(cls):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(logging.INFO)

        ## uppper warining
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        format = logging.Formatter('%(levelname)-8s %(message)s')
        ch.setFormatter(format)
        ch.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
        _nervus_root_logger.addHandler(ch)

        ## lower warning
        ch_info = logging.StreamHandler()
        ch_info.setLevel(logging.DEBUG)
        ch_info.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
        _nervus_root_logger.addHandler(ch_info)

        cls._unexecuted_configure = False

logger = NervusLogger.get_logger('lib.util')

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
        df_tmp = df[df[column_name] == value]
        df_result = pd.concat([df_result, df_tmp], ignore_index=True)
    return df_result


def get_target(source_dir, target):
    if (target is None):
        dirs = glob.glob(source_dir + '/*')
        if dirs:
            target_dir = sorted(dirs, key=lambda f: os.stat(f).st_mtime, reverse=True)[0]   # latest
        else:
            target_dir = None
            logger.error(f"No directory in {source_dir}")
    else:
        target_dir = os.path.join(source_dir, target)
        if os.path.isdir(target_dir):
            target_dir = target_dir
        else:
            target_dir = None
            logger.error(f"No such a directory: {target_dir}")
    return target_dir


def read_train_parameters(parameters_path):
    df_parameters = pd.read_csv(parameters_path, index_col=0)
    df_parameters = df_parameters.fillna(np.nan).replace([np.nan],[None])
    parameters_dict = df_parameters.to_dict()['value']
    
    # Cast
    parameters_dict['lr'] = float(parameters_dict['lr'])
    parameters_dict['epochs'] = int(parameters_dict['epochs'])
    parameters_dict['batch_size'] = int(parameters_dict['batch_size'])
    parameters_dict['input_channel'] = int(parameters_dict['input_channel'])
    return parameters_dict


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

