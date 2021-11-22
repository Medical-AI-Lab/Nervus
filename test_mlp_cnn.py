#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from lib.util import *
from lib.align_env import *
from options.test_options import TestOptions

from dataloader.dataloader_mlp_cnn import *
from config.mlp_cnn import CreateModel_MLPCNN


args = TestOptions().parse()

dirs_dict = set_dirs()
train_opt_log_dir = dirs_dict['train_opt_log']
weight_dir = dirs_dict['weight']
likelilhood_dir = dirs_dict['likelihood']

# Retrieve training options
path_train_opt = get_target(dirs_dict['train_opt_log'], args['test_datetime'])  # the latest train_opt if test_datatime is None
dt_name = get_dt_name(path_train_opt)
train_opt = read_train_options(path_train_opt)   # Retrieve options in training
mlp = train_opt['mlp']
cnn = train_opt['cnn']
gpu_ids = str2int(train_opt['gpu_ids'])
device = set_device(gpu_ids)

image_dir = os.path.join(dirs_dict['images_dir'], train_opt['image_dir'])

csv_dict = parse_csv(os.path.join(dirs_dict['csvs_dir'], train_opt['csv_name']))
class_names = csv_dict['class_names']
num_classes = csv_dict['num_classes']
num_inputs = csv_dict['num_inputs']
id_column = csv_dict['id_column']
label_name = csv_dict['label_name']
split_column = csv_dict['split_column']


# Align option for test only
test_weight = get_target(weight_dir, dt_name)
test_batch_size = args['test_batch_size']        # Default: 64  No exixt in train_opt
train_opt['preprocess'] = 'no'                   # No need of preprocess for image when test, Define no in test_options.py, 


# Data Loader
val_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(train_opt, csv_dict, image_dir, split_list=['val'], batch_size=test_batch_size, sampler='no')    # Fixed 'no'
test_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(train_opt, csv_dict, image_dir, split_list=['test'], batch_size=test_batch_size, sampler='no')  # Fixed 'no'


# Configure of model
model = CreateModel_MLPCNN(mlp, cnn, num_inputs, num_classes, gpu_ids)
weight = torch.load(test_weight)
model.load_state_dict(weight)


# Classification
#def test_classification():
print ('Inference for clasification started...')

val_total = len(val_loader.dataset)
test_total = len(test_loader.dataset)
print(' val_data = {num_val_data}'.format(num_val_data=val_total))
print('test_data = {num_test_data}'.format(num_test_data=test_total))


model.eval()
with torch.no_grad():
    val_acc = 0
    test_acc = 0
    df_result = pd.DataFrame([])

    for split in ['val', 'test']:
        if split == 'val':
            dataloader = val_loader
        elif split == 'test':
            dataloader = test_loader

        for i, (ids, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
            if not(mlp is None) and (cnn is None):
                # When MLP only
                inputs_values_normed = inputs_values_normed.to(device)
                labels = labels.to(device)
                outputs = model(inputs_values_normed)

            elif (mlp is None) and not(cnn is None):
                # When CNN only
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

            elif not(mlp is None) and not(cnn is None):
                # When MLP+CNN
                inputs_values_normed = inputs_values_normed.to(device)
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(inputs_values_normed, images)


            likelihood_ratio = outputs   # No softmax

            _, preds = torch.max(outputs, 1)


            if split == 'val':
                val_acc += (torch.sum(preds == labels.data)).item()

            elif split == 'test':
                test_acc += (torch.sum(preds == labels.data)).item()


            labels = labels.to('cpu').detach().numpy().copy()
            likelihood_ratio = likelihood_ratio.to('cpu').detach().numpy().copy()

            df_id = pd.DataFrame({id_column: ids})
            df_label = pd.DataFrame({label_name :labels})
            df_likelihood_ratio = pd.DataFrame(likelihood_ratio, columns=class_names)
            df_split = pd.DataFrame({split_column: splits})

            df_tmp = pd.concat([df_id, df_label, df_likelihood_ratio, df_split], axis=1)
            df_result = df_result.append(df_tmp, ignore_index=True)

print(' val: Inference_accuracy: {:.4f} %'.format((val_acc / val_total)*100))
print('test: Inference_accuracy: {:.4f} %'.format((test_acc / test_total)*100))
print('Inference finished!')


# Save inference result
os.makedirs(likelilhood_dir, exist_ok=True)
basename = get_basename(test_weight)
likelihood_path = os.path.join(likelilhood_dir, basename) + '.csv'
df_result.to_csv(likelihood_path, index=False)


# ----- EOF -----
