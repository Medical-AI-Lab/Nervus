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
        if torch.cuda.is_available():
            primary_gpu_id = gpu_ids[0]
            device_name = f'cuda:{primary_gpu_id}'
            device = torch.device(device_name)
        else:
            print('Error from set_device: No avalibale GPU on this machine. Use CPU.')
            exit()
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


def get_image_dir(image_dirs:dict, image_set):
    image_dir = image_dirs.get(image_set)

    if image_dir is None:
        print('No specified image set.')
        exit()
    else:
        pass

    return image_dir


def get_basename(path):
    return (os.path.splitext(os.path.basename(path)))[0]  # Delete suffix


def get_latest(target_dir):
    target_paths = glob.glob(target_dir + '/*')

    if target_paths:
        target_sorted = sorted(target_paths, key=lambda f: os.stat(f).st_mtime, reverse=True)
        latest = target_sorted[0]
    else:
        latest = None

    return latest


def get_target(target_dir, target_name):
    if not(target_name is None):
        target_path = os.path.join(target_dir, target_name)

        if os.path.exists(target_path):
            valid_path = target_path
        else:
            print('No specified target.')
            exit()
    else:
        valid_path = get_latest(target_dir)  # Dafult: the latest weight
        if valid_path is None:
            print('No target in {}.'.format(target_dir))
            exit()
        else:
            pass

    return valid_path



def strcat(delim, *strs):
    joined = delim.join(strs)
    return joined


"""
# Notation of basename
# basename = '[project_name]_[project_task]+ResNet18_CrossEntropyLoss_Adam_batch-size-64_image-set-covid_epochs-3_val-best-epoch-3_val-loss-0.0019_GPU-0-1_2021-05-21-15-04'
"""
def make_basename(args, val_best_epoch, val_best_loss, dt_name):

    gpu_ids = args['gpu_ids']

    if gpu_ids:
        gpu_ids = [str(id) for id in gpu_ids]
        gpu_ids.insert(0, 'GPU')
        device_name = '-'.join(gpu_ids)
    else:
        device_name = 'CPU'

    prefix = strcat('_', args['project_name'], args['project_task'])

    conf = strcat('_', args['model'],
                       args['criterion'],
                       args['optimizer'],
                       'batch-size-' + str(args['batch_size']),
                       'image-set-' + str(args['image_set']),
                       'resize-size-' + str(args['resize_size']),
                       'epochs-' + str(args['epochs']),
                       'val-best-epoch-' + str(val_best_epoch),
                       'val-best-loss-' + f'{val_best_loss:.4f}',
                       device_name,
                       dt_name)

    basename = strcat('+', prefix, conf)

    return basename



def save_train_options(args, train_opt_log_dir, dt_name):
    save_path = os.path.join(train_opt_log_dir, dt_name + '.csv')

    df_opt = pd.DataFrame(list(args.items()), columns=['option', 'value'])
    df_opt.to_csv(save_path, index=False)



def read_train_options(path, train_opt_log_dir):
    basename = get_basename(path)
    datetime = basename.rsplit('_', 1)[-1]
    opt_path = os.path.join(train_opt_log_dir, datetime + '.csv')

    df_opt = pd.read_csv(opt_path, index_col=0)
    df_opt = df_opt.fillna(np.nan).replace([np.nan],[None])
    opt_dict = df_opt.to_dict()['value']

    return opt_dict



def str2int(gpu_ids_str:str):
    gpu_ids_str = gpu_ids_str.replace('[', '').replace(']', '')

    if gpu_ids_str == '':
        gpu_ids = []
    else:
        gpu_ids = gpu_ids_str.split(',')
        gpu_ids = [ int(i) for i in gpu_ids ]

    return gpu_ids


# ----- EOF -----
