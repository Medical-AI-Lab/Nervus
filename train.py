#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import copy

from lib.util import *
from lib.align_env import *
from options.train_options import TrainOptions
from config.criterion import set_criterion
from config.optimizer import set_optimizer
from config.model import *


train_option_parser = TrainOptions()
args = train_option_parser.parse()
#TrainOptions().is_option_valid(args)
train_option_parser.print_options()

task = args['task']
mlp = args['mlp']
cnn = args['cnn']
criterion = args['criterion']
optimizer = args['optimizer']
lr = args['lr']
num_epochs = args['epochs']
batch_size = args['batch_size']
sampler = args['sampler']
gpu_ids = args['gpu_ids']
device = set_device(gpu_ids)

nervusenv = NervusEnv()
image_dir = os.path.join(nervusenv.images_dir , args['image_dir'])
csv_dict = parse_csv(os.path.join(nervusenv.splits_dir, args['csv_name']), task)
label_num_classes = csv_dict['label_num_classes']
label_list = csv_dict['label_list']
num_inputs = csv_dict['num_inputs']

## bool of using neural network
hasMLP = mlp is not None
hasCNN = cnn is not None

## choice dataloader and function to execute
if task == 'deepsurv':
    from dataloader.dataloader_deepsurv import *
    def _execute_task(*args):
        return _execute_deepsurv(*args)
else: # classification or regression
    if len(label_list) > 1:
        # Multi-label outputs
        from dataloader.dataloader_multi import *
        def _execute_task(*args):
            return _execute_multi_label(*args)
    else:
        # Single-label output
        from dataloader.dataloader import *
        def _execute_task(*args):
            return _execute_single_label(*args)

train_loader = dataloader_mlp_cnn(args, csv_dict, image_dir, split_list=['train'], batch_size=batch_size, sampler=sampler)
val_loader = dataloader_mlp_cnn(args, csv_dict, image_dir, split_list=['val'], batch_size=batch_size, sampler=sampler)

# Configure of training
model = create_mlp_cnn(mlp, cnn, num_inputs, label_num_classes, gpu_ids=gpu_ids)
criterion = set_criterion(criterion, device)
optimizer = set_optimizer(optimizer, model, lr)

best_weight = None
val_best_loss = None
val_best_epoch = None

loss_acc_dict = {}
if task == 'classification':
    loss_acc_dict = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
else:
    # When regression or deepsurv
    loss_acc_dict = {'train_loss': [], 'val_loss': []}

def execute(best_weight, val_best_loss, val_best_epoch, loss_acc_dict):
    for _epoch in range(num_epochs):
        for _phase in ['train', 'val']:
            if _phase == 'train':
                model.train()
                _dataloader = train_loader
            else: # elif phase == 'val':
                model.eval()
                _dataloader = val_loader

            _running_loss = 0.0
            _running_acc = 0.0
            # execute task: execute_single_label, execute_multi_label, execute_deepsurv
            _running_loss, _running_acc = _execute_task(_phase, _dataloader)

            _update_flag = None
            loss_acc_dict, val_best_loss, val_best_epoch, _update_flag = _update_loss_acc_dict(task, num_epochs, loss_acc_dict, _epoch, _phase, _running_loss, _running_acc, len(_dataloader.dataset), len(label_list), val_best_loss, val_best_epoch)

            # Keep the best weight when epoch_loss is the lowest.
            if (_phase == 'val' and _update_flag):
                best_weight = copy.deepcopy(model.state_dict())

    return best_weight, val_best_loss, val_best_epoch, loss_acc_dict

def _execute_single_label(phase:str, dataloader:Dataset) -> Tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0

    for i, (ids, raw_outputs, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

            labels = labels.to(device)

            if task == 'classification':
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs.squeeze(), labels.float())

            if phase == 'train':
                loss.backward()
                optimizer.step()

        if task == 'classification':
            running_loss += loss.item() * labels.size(0)
            running_acc += (torch.sum(preds == labels.data)).item()
        else:
            running_loss += loss.item() * labels.size(0)

    return running_loss, running_acc

def _execute_multi_label(phase:str, dataloader:Dataset) -> Tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0

    for i, (ids, raw_outputs_dict, labels_dict, inputs_values_normed, images, splits) in enumerate(dataloader):
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

            labels_multi = {label_name: labels.to(device) for label_name, labels in labels_dict.items()}

            # Initialize every iteration
            preds_multi = {}
            loss_multi = {}

            for label_name, labels in labels_multi.items():
                layer_outputs = get_layer_output(outputs, label_name)

                if task == 'classification':
                    preds_multi[label_name] = torch.max(layer_outputs, 1)[1]
                    loss_multi[label_name] = criterion(layer_outputs, labels)
                else:
                    loss_multi[label_name] = criterion(layer_outputs.squeeze(), labels.float())

            # Zero Reset
            # Without this, cannot backward every iteration,
            # bacause loss is kept all the time in spite of that computaion graph is freed after the first iteration.
            # This means that the backword after the second iteration cannot be done absolutely.
            # After each backward, no need to keep any loss from the previous iteration.
            ##loss = torch.tensor([0.0], requires_grad = True).to(device)
            loss = torch.tensor([0.0]).to(device)

            # Total of loss for each label_i
            for loss_i in loss_multi.values():
                loss = torch.add(loss, loss_i)

                # Backward and Update weight
            if phase == 'train':
                loss.backward()
                optimizer.step()

        if task == 'classification':
            running_loss += loss.item() * labels.size(0)

            for label_name, labels in labels_multi.items():
                running_acc_i = (torch.sum(preds_multi[label_name] == labels.data).item())
                running_acc += running_acc_i
        else:
            running_loss += loss.item() * labels.size(0)

    return running_loss, running_acc

def _execute_deepsurv(phase:str, dataloader:Dataset) -> Tuple[float, float]:
    running_loss = 0.0
    running_acc = 0.0

    for i, (ids, raw_outputs, labels, periods, inputs_values_normed, images, splits) in enumerate(dataloader):
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            outputs = predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images)

            periods = periods.float().to(device)
            if hasMLP:
                labels = labels.float().to(device)
            else:
                labels = labels.to(device)

            risk_preds = outputs   # Just rename for clarity
            loss = criterion(risk_preds, periods.reshape(-1,1), labels.reshape(-1,1), model)

            if (phase == 'train') and (torch.sum(labels).item() > 0):
            # No backward when all labels are 0.
            # To be specofic, loss(NegativeLogLikelihood) cannot be defined in this case.
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss, running_acc

def _update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len_dataloader, len_label_list, val_best_loss, val_best_epoch):
    update_comment = None
    update_flag = None

    if task == 'classification':
        epoch_loss = running_loss / (len_dataloader * len_label_list)
        epoch_acc = running_acc / (len_dataloader * len_label_list)
        if phase == 'train':
            loss_acc_dict['train_loss'].append(epoch_loss)
            loss_acc_dict['train_acc'].append(epoch_acc)
        else: # elif phase == 'val':
            loss_acc_dict['val_loss'].append(epoch_loss)
            loss_acc_dict['val_acc'].append(epoch_acc)
    else:
        # When regression or deepsurv
        epoch_loss = running_loss / (len_dataloader * len_label_list)
        if phase == 'train':
            loss_acc_dict['train_loss'].append(epoch_loss)
        else: # elif phase == 'val':
            loss_acc_dict['val_loss'].append(epoch_loss)

    # Check if val_best_loss
    if (phase == 'val') and ((val_best_loss is None) or (epoch_loss < val_best_loss)):
        val_best_loss = epoch_loss
        val_best_epoch = epoch + 1
        update_comment = ' Updated val_best_loss!'
        update_flag = True
    else:
        update_comment = ''
        update_flag = False

    # Print loss and acc at lats epoch
    if phase == 'val':
        if task == 'classification':
            print(f"epoch [{epoch+1:>3}/{num_epochs:<3}], train_loss: {loss_acc_dict['train_loss'][-1]:.4f}, val_loss: {loss_acc_dict['val_loss'][-1]:.4f}, val_acc: {loss_acc_dict['val_acc'][-1]:.4f}" + update_comment)
        else:
            # When regression or deepsurv
            print(f"epoch [{epoch+1:>3}/{num_epochs:<3}], train_loss: {loss_acc_dict['train_loss'][-1]:.4f}, val_loss: {loss_acc_dict['val_loss'][-1]:.4f}" + update_comment)

    return loss_acc_dict, val_best_loss, val_best_epoch, update_flag

def save_result(best_weight, val_best_loss, val_best_epoch, loss_acc_dict):
    # Save
    date_now = datetime.datetime.now()
    date_name = date_now.strftime('%Y-%m-%d-%H-%M-%S')
    save_dir = os.path.join(nervusenv.sets_dir, date_name)
    os.makedirs(save_dir, exist_ok=True)

    # Parameters
    df_opt = pd.DataFrame(list(args.items()), columns=['option', 'value'])
    parameters_path = os.path.join(save_dir, nervusenv.csv_parameters)
    df_opt.to_csv(parameters_path, index=False)

    # Weight
    weight_path = os.path.join(save_dir, nervusenv.weight)
    torch.save(best_weight, weight_path)

    # Learning curve
    csv_learning_curve = nervusenv.csv_learning_curve.replace('.csv', '') + '_val-best-epoch-' + str(val_best_epoch) + '_val-best-loss-' + f"{val_best_loss:.4f}" + '.csv'
    learning_curve_path = os.path.join(save_dir, csv_learning_curve)
    df_learning_curve = pd.DataFrame(loss_acc_dict)
    df_learning_curve.to_csv(learning_curve_path, index=False)

if __name__=="__main__":
    # Training
    print ('Training started...')
    print(f"train_data = {len(train_loader.dataset)}")
    print(f"  val_data = {len(val_loader.dataset)}")

    best_weight, val_best_loss, val_best_epoch, loss_acc_dict = execute(best_weight, val_best_loss, val_best_epoch, loss_acc_dict)

    print('Training finished!')

    save_result(best_weight, val_best_loss, val_best_epoch, loss_acc_dict)
