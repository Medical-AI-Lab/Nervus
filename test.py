#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import torch

from lib.util import *
from lib.align_env import *
from options.test_options import TestOptions
from config.model import *


args = TestOptions().parse()
nervusenv = NervusEnv()
datetime_dir = get_target(nervusenv.sets_dir, args['test_datetime'])   # args['test_datetime'] if exists or the latest
hyperparameters_path = os.path.join(datetime_dir, nervusenv.csv_hyperparameters)
train_hyperparameters = read_train_hyperparameters(hyperparameters_path)
task = train_hyperparameters['task']
mlp = train_hyperparameters['mlp']
cnn = train_hyperparameters['cnn']
gpu_ids = str2int(train_hyperparameters['gpu_ids'])
device, _ = set_device(gpu_ids)

image_dir = os.path.join(nervusenv.images_dir, train_hyperparameters['image_dir'])
csv_dict = parse_csv(os.path.join(nervusenv.csvs_dir, train_hyperparameters['csv_name']), task)

output_class_label = csv_dict['output_class_label']
label_num_classes = csv_dict['label_num_classes']
label_list = csv_dict['label_list']
output_list = csv_dict['output_list']

num_inputs = csv_dict['num_inputs']
id_column = csv_dict['id_column']
split_column = csv_dict['split_column']
period_column = csv_dict['period_column']   # When classification or regression, None

# Align option for test only
test_weight = os.path.join(datetime_dir, nervusenv.weight)
test_batch_size = args['test_batch_size']                            # Default: 64  No exixt in train_opt
train_hyperparameters['preprocess'] = 'no'                           # No need of preprocess for image when test, Define no in test_options.py
train_hyperparameters['normalize_image'] = args['normalize_image']   # Default: 'yes'


# Data Loadar
if task == 'deepsurv':
    from dataloader.dataloader_deepsurv import *
else:
    # when classification or regression
    if len(label_list) > 1:
        from dataloader.dataloader_multi import *
    else:
        from dataloader.dataloader import *
val_loader = dalaloader_mlp_cnn(train_hyperparameters, csv_dict, image_dir, split_list=['val'], batch_size=test_batch_size, sampler='no')
test_loader = dalaloader_mlp_cnn(train_hyperparameters, csv_dict, image_dir, split_list=['test'], batch_size=test_batch_size, sampler='no')

# Configure of model
model = create_mlp_cnn(mlp, cnn, label_num_classes, num_inputs, gpu_ids=gpu_ids)
weight = torch.load(test_weight)
model.load_state_dict(weight)

# Make column name of 
# {output_XXX: {A:1, B:2, C:3}, ...} -> {output_XXX: [pred_XXX_A, pred_XXX_B, pred_XXX_C], ...}
column_output_class_names_dict = {}
for output_name, class_label_dict in output_class_label.items():
    column_class_label_names = []
    if task == 'classification':
        for class_name in class_label_dict.keys():
            column_class_label_names.append('pred_' + output_name + '_' + class_name)
    else:
    # When regression or deepsurv
        column_class_label_names.append('pred_' + output_name)
    column_output_class_names_dict[output_name] = column_class_label_names


def execute_test_single_label(task, mlp, cnn, device, id_column, split_column, val_loader, test_loader, model, column_output_class_names_dict):
    model.eval()
    with torch.no_grad():
        val_acc = 0.0
        test_acc = 0.0
        df_result = pd.DataFrame([])

        for split in ['val', 'test']:
            if split == 'val':
                dataloader = val_loader
            else: # elif split == 'test':
                dataloader = test_loader

            for i, (ids, raw_outputs, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
                if (mlp is not None) and (cnn is None):
                # When MLP only
                    inputs_values_normed = inputs_values_normed.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs_values_normed)

                elif (mlp is None) and (cnn is not None):
                # When CNN only
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)

                else: # elif not(mlp is None) and not(cnn is None):
                # When MLP+CNN
                    inputs_values_normed = inputs_values_normed.to(device)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs_values_normed, images)

                likelihood_ratio = outputs   # No softmax

                if task == 'classification':
                    _, preds = torch.max(outputs, 1)
                    split_acc = (torch.sum(preds == labels.data)).item()
                    if split == 'val':
                        val_acc += split_acc
                    else: #elif split == 'test':
                        test_acc += split_acc
                else:
                    pass

                labels = labels.to('cpu').detach().numpy().copy()
                likelihood_ratio = likelihood_ratio.to('cpu').detach().numpy().copy()

                output_name = output_list[0]
                df_id = pd.DataFrame({id_column: ids})
                df_split = pd.DataFrame({split_column: splits})
                df_raw_output = pd.DataFrame({output_name: raw_outputs})
                df_likelihood = pd.DataFrame(likelihood_ratio, columns=column_output_class_names_dict[output_name])
                df_tmp = pd.concat([df_id, df_raw_output, df_likelihood, df_split], axis=1)
                df_result = df_result.append(df_tmp, ignore_index=True)

    return val_acc, test_acc, df_result

def execute_test_multi_label(task, mlp, cnn, device, id_column, split_column, val_loader, test_loader, model, column_output_class_names_dict):
    model.eval()
    with torch.no_grad():
        val_acc = 0
        test_acc = 0
        df_result = pd.DataFrame([])

        for split in ['val', 'test']:
            if split == 'val':
                dataloader = val_loader
            else: # elif split == 'test':
                dataloader = test_loader

            for i, (ids, raw_outputs_dict, labels_dict, inputs_values_normed, images, splits) in enumerate(dataloader):
                if (mlp is not None) and (cnn is None):
                # When MLP only
                    inputs_values_normed = inputs_values_normed.to(device)
                    labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}
                    outputs = model(inputs_values_normed)

                elif (mlp is None) and (cnn is not None):
                # When CNN only
                    images = images.to(device)
                    labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}
                    outputs = model(images)

                else: # elif not(mlp is None) and not(cnn is None):
                # When MLP+CNN
                    inputs_values_normed = inputs_values_normed.to(device)
                    images = images.to(device)
                    labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}
                    outputs = model(inputs_values_normed, images)

                likelihood_multi = {}
                preds_multi = {}
                for label_name, labels in labels_multi.items():
                    likelihood_multi[label_name] = get_layer_output(outputs, label_name)   # No softmax
                    if task == 'classification':
                        preds_multi[label_name] = torch.max(likelihood_multi[label_name], 1)[1]
                        split_acc_label_name = torch.sum(preds_multi[label_name] == labels.data).item()
                        if split == 'val':
                            val_acc += split_acc_label_name
                        else: #elif split == 'test':
                            test_acc += split_acc_label_name
                    else:
                        pass

                labels_multi = {label_name: label.to('cpu').detach().numpy().copy() for label_name, label in labels_multi.items()}
                likelihood_multi = {label_name: likelihood.to('cpu').detach().numpy().copy() for label_name, likelihood in likelihood_multi.items()}

                df_id = pd.DataFrame({id_column: ids})
                df_split = pd.DataFrame({split_column: splits})
                df_likelihood_tmp = pd.DataFrame([])
                for output_name, class_label_names in column_output_class_names_dict.items():
                    label_name = 'label_' + output_name
                    df_raw_output = pd.DataFrame(raw_outputs_dict[output_name], columns=[output_name])
                    df_likelihood = pd.DataFrame(likelihood_multi[label_name], columns=class_label_names)
                    df_likelihood_tmp = pd.concat([df_likelihood_tmp, df_raw_output, df_likelihood], axis=1)

                df_tmp = pd.concat([df_id, df_likelihood_tmp, df_split], axis=1)
                df_result = df_result.append(df_tmp, ignore_index=True)
    return val_acc, test_acc, df_result

def execute_test_deepsurv(mlp, cnn, device, id_column, split_column, period_column, val_loader, test_loader, model, column_output_class_names_dict):
    model.eval()
    with torch.no_grad():
        df_result = pd.DataFrame([])

        for split in ['val', 'test']:
            if split == 'val':
                dataloader = val_loader
            else: #elif split == 'test':
                dataloader = test_loader

            for i, (ids, raw_outputs, labels, periods, inputs_values_normed, images, splits) in enumerate(dataloader):
                if (mlp is not None) and (cnn is None):
                # When MLP only
                    inputs_values_normed = inputs_values_normed.to(device)
                    outputs = model(inputs_values_normed)

                elif (mlp is None) and (cnn is not None):
                # When CNN only
                    images = images.to(device)
                    outputs = model(images)

                else: # elif not(mlp is None) and not(cnn is None):
                # When MLP+CNN
                    inputs_values_normed = inputs_values_normed.to(device)
                    images = images.to(device)
                    outputs = model(inputs_values_normed, images)

                likelihood_ratio = outputs   # No softmax
                labels = labels.to('cpu').detach().numpy().copy()
                periods = periods.to('cpu').detach().numpy().copy()
                likelihood_ratio = likelihood_ratio.to('cpu').detach().numpy().copy()

                output_name = output_list[0]
                label_name = label_list[0]
                df_id = pd.DataFrame({id_column: ids})
                df_split = pd.DataFrame({split_column: splits})
                df_raw_output = pd.DataFrame({output_name: raw_outputs})
                df_label = pd.DataFrame({label_name: labels}, dtype=int)
                df_likelihood = pd.DataFrame(likelihood_ratio, columns=column_output_class_names_dict[output_name])
                df_period = pd.DataFrame({period_column: periods})
                df_tmp = pd.concat([df_id, df_raw_output, df_label, df_period, df_likelihood, df_split], axis=1)
                df_result = df_result.append(df_tmp, ignore_index=True)
    return df_result


# Inference
print ('Inference started...')
val_total = len(val_loader.dataset)
test_total = len(test_loader.dataset)
print(f" val_data = {val_total}")
print(f"test_data = {test_total}")
if task == 'deepsurv':
    df_result = execute_test_deepsurv(mlp, cnn, device, id_column, split_column, period_column, val_loader, test_loader, model, column_output_class_names_dict)
else:
# When classification or regression
    if len(label_list) > 1:
        # Multi-label outputs
        val_acc, test_acc, df_result = execute_test_multi_label(task, mlp, cnn, device, id_column, split_column, val_loader, test_loader, model, column_output_class_names_dict)
    else:
        # Single-label output
        val_acc, test_acc, df_result = execute_test_single_label(task, mlp, cnn, device, id_column, split_column, val_loader, test_loader, model, column_output_class_names_dict)

if task == 'classification':
    val_acc = (val_acc / (val_total * len(label_list))) * 100
    test_acc = (test_acc / (test_total * len(label_list))) * 100
    print(f" val: Inference_accuracy: {val_acc:.4f} %")
    print(f"test: Inference_accuracy: {test_acc:.4f} %")
else:
    # When regresson or deepsurv
    pass
print('Inference finished!')

# Save likelohood
likelihood_path = os.path.join(datetime_dir, nervusenv.csv_likelihood)
df_result.to_csv(likelihood_path, index=False)

# ----- EOF -----
