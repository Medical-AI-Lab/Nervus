#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Tuple

import pandas as pd
import torch
from dataloader.dataloader_deepsurv import DeepSurvDataSet
from dataloader.dataloader_multi import MultiLabelDataSet
from dataloader.dataloader_single import SingleLabelDataSet

from lib.util import *
from lib.align_env import *
from options.test_options import TestOptions
from config.model import *


nervusenv = NervusEnv()
args = TestOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['test_datetime'])   # args['test_datetime'] if exists or the latest
parameters_path = os.path.join(datetime_dir, nervusenv.csv_parameters)
train_parameters = read_train_parameters(parameters_path)
task = train_parameters['task']
mlp = train_parameters['mlp']
cnn = train_parameters['cnn']
gpu_ids = str2int(train_parameters['gpu_ids'])
device = set_device(gpu_ids)

image_dir = os.path.join(nervusenv.images_dir, train_parameters['image_dir'])
sp = SplitProvider(os.path.join(nervusenv.splits_dir, train_parameters['csv_name']), task)
label_list = sp.internal_label_list   # Regard internal label as label

# Align option for test only
test_weight = os.path.join(datetime_dir, nervusenv.weight)
test_batch_size = args['test_batch_size']                       # Default: 64  No exixt in train_opt
train_parameters['preprocess'] = 'no'                           # MUST: Stop augmentaion, No need when test
train_parameters['normalize_image'] = args['normalize_image']   # Default: 'yes'

## bool of using neural network
hasMLP = mlp is not None
hasCNN = cnn is not None

## choose dataloader and function to execute test
if task == 'deepsurv':
    dataset_handler = DeepSurvDataSet
    def _execute_test(*args):
        return _execute_test_deepsurv(*args)
else:
    if len(label_list) > 1: #multi
        dataset_handler = MultiLabelDataSet
        def _execute_test(*args):
            return _execute_test_multi_label(*args)
    else: #single
        dataset_handler = SingleLabelDataSet
        def _execute_test(*args):
            return _execute_test_single_label(*args)

train_loader = dataset_handler.create_dataloader(train_parameters, sp, image_dir, split_list=['train'], batch_size=test_batch_size, sampler='no')
val_loader = dataset_handler.create_dataloader(train_parameters, sp, image_dir, split_list=['val'], batch_size=test_batch_size, sampler='no')
test_loader = dataset_handler.create_dataloader(train_parameters, sp, image_dir, split_list=['test'], batch_size=test_batch_size, sampler='no')

# Configure of model
model = create_mlp_cnn(mlp, cnn, sp.num_inputs, sp.num_classes_in_internal_label, gpu_ids=gpu_ids)
weight = torch.load(test_weight)
model.load_state_dict(weight)

# Make column name of
# {label_XXX: {A:1, B:2, C:3}, ...} -> {label_XXX: [pred_label_XXX_A, pred_label_XXX_B, pred_label_XXX_C], ...}
column_pred_names_in_label_dict = {}
for raw_label_name, class_dict in sp.class_name_in_raw_label.items():
    pred_names = []
    if task == 'classification':
        for class_name in class_dict.keys():
            pred_names.append('pred_' + raw_label_name + '_' + str(class_name))
    else:
    # When regression or deepsurv
        pred_names.append('pred_' + raw_label_name)
    column_pred_names_in_label_dict[raw_label_name] = pred_names

def execute():
    model.eval()
    with torch.no_grad():
        train_acc = 0.0
        val_acc = 0.0
        test_acc = 0.0
        df_result = pd.DataFrame([])

        for _split in ['train', 'val', 'test']:
            if _split == 'train':
                _dataloader = train_loader
            elif _split == 'val':
                _dataloader = val_loader
            elif _split == 'test':
                _dataloader = test_loader
            else: #
                print('Split in dataloader error.')

            # execute test: _execute_test_single_label, _execute_test_multi_label, _execute_test_deepsurv
            _train_acc, _val_acc, _test_acc, _df_result = _execute_test(_split, _dataloader)
            # merge result of test
            train_acc += _train_acc
            val_acc += _val_acc
            test_acc += _test_acc
            df_result = pd.concat([df_result, _df_result], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result

def _execute_test_single_label(split:str, dataloader:Dataset) -> Tuple[float, float, pd.DataFrame]:
    train_acc = 0.0
    val_acc = 0.0
    test_acc = 0.0
    df_result = pd.DataFrame([])

    # Regard internal label as label
    for i, (ids, raw_labels, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
        outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

        likelihood = outputs   # No softmax

        if task == 'classification':
            _, preds = torch.max(outputs, 1)
            split_acc = (torch.sum(preds == labels.data)).item()
            if split == 'train':
                train_acc += split_acc
            elif split == 'val':
                val_acc += split_acc
            elif split == 'test':
                test_acc += split_acc
        else:
            pass

        labels = labels.to('cpu').detach().numpy().copy()
        likelihood = likelihood.to('cpu').detach().numpy().copy()

        raw_label_name = list(column_pred_names_in_label_dict.keys())[0]
        df_id = pd.DataFrame({sp.id_column: ids})
        df_split = pd.DataFrame({sp.split_column: splits})
        df_raw_label = pd.DataFrame({raw_label_name: raw_labels})
        df_likelihood = pd.DataFrame(likelihood, columns=column_pred_names_in_label_dict[raw_label_name])
        df_tmp = pd.concat([df_id, df_raw_label, df_likelihood, df_split], axis=1)
        df_result = pd.concat([df_result, df_tmp], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result

def _execute_test_multi_label(split:str, dataloader:Dataset) -> Tuple[float, float, pd.DataFrame]:
    train_acc = 0.0
    val_acc = 0.0
    test_acc = 0.0
    df_result = pd.DataFrame([])

    # Regard internal label as label
    for i, (ids, raw_labels_dict, labels_dict, inputs_values_normed, images, splits) in enumerate(dataloader):
        outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

        likelihood_multi = {}
        preds_multi = {}
        labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}
        for label_name, labels in labels_multi.items():
            likelihood_multi[label_name] = get_layer_output(outputs, label_name)   # No softmax
            if task == 'classification':
                preds_multi[label_name] = torch.max(likelihood_multi[label_name], 1)[1]
                split_acc_label_name = torch.sum(preds_multi[label_name] == labels.data).item()
                if split == 'train':
                    train_acc += split_acc_label_name
                elif split == 'val':
                    val_acc += split_acc_label_name
                elif split == 'test':
                    test_acc += split_acc_label_name
            else:
                pass

        labels_multi = {label_name: label.to('cpu').detach().numpy().copy() for label_name, label in labels_multi.items()}
        likelihood_multi = {label_name: likelihood.to('cpu').detach().numpy().copy() for label_name, likelihood in likelihood_multi.items()}

        df_id = pd.DataFrame({sp.id_column: ids})
        df_split = pd.DataFrame({sp.split_column: splits})
        df_likelihood_tmp = pd.DataFrame([])
        for label_name in likelihood_multi.keys():
            raw_label_name = label_name.split('_', 1)[-1]                                                                         # internal_label_XXX -> label_XXX
            df_raw_label = pd.DataFrame(raw_labels_dict[raw_label_name], columns=[raw_label_name])
            df_likelihood = pd.DataFrame(likelihood_multi[label_name], columns=column_pred_names_in_label_dict[raw_label_name])   # pred_names = [pred_XXX_A, pred_XXX_B, pred_XXX_C]
            df_likelihood_tmp = pd.concat([df_likelihood_tmp, df_raw_label, df_likelihood], axis=1)

        df_tmp = pd.concat([df_id, df_likelihood_tmp, df_split], axis=1)
        df_result = pd.concat([df_result, df_tmp], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result

def _execute_test_deepsurv(_, dataloader:Dataset) -> Tuple[float, float, pd.DataFrame]:
    train_acc = 0.0 # dummy value: not use in deepsurv
    val_acc = 0.0 # dummy value: not use in deepsurv
    test_acc = 0.0 # dummy value: not use in deepsurv
    df_result = pd.DataFrame([])

    # Regard internal label as label
    for i, (ids, raw_labels, labels, periods, inputs_values_normed, images, splits) in enumerate(dataloader):
        outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

        likelihood = outputs   # No softmax
        labels = labels.to('cpu').detach().numpy().copy()
        periods = periods.to('cpu').detach().numpy().copy()
        likelihood = likelihood.to('cpu').detach().numpy().copy()

        raw_label_name = list(column_pred_names_in_label_dict.keys())[0]
        df_id = pd.DataFrame({sp.id_column: ids})
        df_split = pd.DataFrame({sp.split_column: splits})
        df_raw_label = pd.DataFrame({raw_label_name: raw_labels})
        label_name = sp.prefix_internal_label + raw_label_name.replace(sp.prefix_raw_label, '')           # label_XXX -> internal_label_XXX
        df_label = pd.DataFrame({label_name: labels}, dtype=int)                                          # Needed to calculate c-index
        df_likelihood = pd.DataFrame(likelihood, columns=column_pred_names_in_label_dict[raw_label_name])
        df_period = pd.DataFrame({sp.period_column: periods})
        df_tmp = pd.concat([df_id, df_raw_label, df_label, df_period, df_likelihood, df_split], axis=1)
        df_result = pd.concat([df_result, df_tmp], ignore_index=True)

    return train_acc, val_acc, test_acc, df_result

if __name__=="__main__":
    # Inference
    print('Inference started...')
    train_total = len(train_loader.dataset)
    val_total = len(val_loader.dataset)
    test_total = len(test_loader.dataset)
    print(f"train_data = {train_total}")
    print(f"  val_data = {val_total}")
    print(f" test_data = {test_total}")

    train_acc, val_acc, test_acc, df_result = execute()

    if task == 'classification':
        train_acc = (train_acc / (train_total * len(label_list))) * 100
        val_acc = (val_acc / (val_total * len(label_list))) * 100
        test_acc = (test_acc / (test_total * len(label_list))) * 100
        print(f"train: Inference_accuracy: {train_acc:.4f} %")
        print(f"  val: Inference_accuracy: {val_acc:.4f} %")
        print(f" test: Inference_accuracy: {test_acc:.4f} %")
    else:
        # When regresson or deepsurv
        pass
    print('Inference finished!')

    # Save likelohood
    likelihood_path = os.path.join(datetime_dir, nervusenv.csv_likelihood)
    df_result.to_csv(likelihood_path, index=False)
