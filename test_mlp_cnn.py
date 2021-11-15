#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn


from lib import static
from lib.options import Options
from lib.util import *

from dataloader.dataloader_mlp_cnn import *
from config.mlp_cnn import CreateModel_MLPCNN


args = Options().parse()


csv_path = static.csv_path
train_opt_log_dir = static.train_opt_log_dir
weight_dir = static.weight_dir
likelilhood_dir = static.likelihood_dir
num_classes = static.num_classes
class_names = static.class_names
label_name = static.label_name
input_list = static.input_list
input_list_normed = static.input_list_normed
num_inputs = len(input_list_normed)


# Dafault: the latest weight
test_weight = get_target(weight_dir, args['test_weight'])


# Configure of training
train_opt = read_train_options(test_weight, train_opt_log_dir)
mlp = train_opt['mlp']
cnn = train_opt['cnn']
load_image = 'yes' if not(cnn is None) else 'no'

image_set = train_opt['image_set']
image_dir = get_image_dir(static.image_dirs, image_set)
resize_size = int(train_opt['resize_size'])
normalize_image = train_opt['normalize_image']

gpu_ids = str2int(train_opt['gpu_ids'])
device = set_device(gpu_ids)


# Data Loader
val_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(
                                    image_dir,
                                    label_name,
                                    input_list,
                                    split_list=['val'],
                                    resize_size=resize_size,
                                    normalize_image=normalize_image,
                                    load_image=load_image,
                                    batch_size=args['test_batch_size'],
                                    is_sampler='no'   # Fixed
                                )


test_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(
                                    image_dir,
                                    label_name,
                                    input_list,
                                    split_list=['test'],
                                    resize_size=resize_size,
                                    normalize_image=normalize_image,
                                    load_image=load_image,
                                    batch_size=args['test_batch_size'],
                                    is_sampler='no'   # Fixed
                                )


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

        for i, (pids, labels, inputs_values, inputs_values_normed, images, image_path_names, splits) in enumerate(dataloader):
            if not(mlp is None) and (cnn is None):
                # When MLP only
                inputs_values_normed = inputs_values_normed.to(device)  # Pass to GPU
                images = images                                         # No pass to GPU
                labels = labels.to(device)
                outputs = model(inputs_values_normed)

            elif (mlp is None) and not(cnn is None):
                # When CNN only
                inputs_values_normed = inputs_values_normed             # No pass to GPU
                images = images.to(device)                              # Pass to GPU
                labels = labels.to(device)
                outputs = model(images)

            elif not(mlp is None) and not(cnn is None):
                # When MLP+CNN
                inputs_values_normed = inputs_values_normed.to(device)  # Pass to GPU
                images = images.to(device)                              # Pass to GPU
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

            df_misc = pd.DataFrame({
                                    **{'PID': pids},
                                    **{label_name: labels},
                                    **{'path_to_img': image_path_names},
                                    **{'split': splits}
                                  })

            # Original values
            inputs_values = inputs_values.to('cpu').detach().numpy().copy()
            df_inputs_values = pd.DataFrame(inputs_values, columns=input_list)
            df_likelihood_ratio = pd.DataFrame(likelihood_ratio, columns=class_names)
            df_tmp = pd.concat([df_misc, df_inputs_values, df_likelihood_ratio], axis=1)
            df_result = df_result.append(df_tmp, ignore_index=True)



# Sort columns
sorted_columns = ['PID', label_name] + input_list + class_names + ['path_to_img', 'split']
df_result = df_result.reindex(columns=sorted_columns)

print(' val: Inference_accuracy: {:.4f} %'.format((val_acc / val_total)*100))
print('test: Inference_accuracy: {:.4f} %'.format((test_acc / test_total)*100))
print('Inference finished!')


# Save inference result
os.makedirs(likelilhood_dir, exist_ok=True)
basename = get_basename(test_weight)
likelihood_path = os.path.join(likelilhood_dir, basename) + '.csv'
df_result.to_csv(likelihood_path, index=False)



# ----- EOF -----
