#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

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


args = TrainOptions().parse()

#TrainOptions().is_option_valid(args)
TrainOptions().print_options(args)

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

dirs_dict = set_dirs()
train_opt_log_dir = dirs_dict['train_opt_log']
weight_dir = dirs_dict['weight']
learning_curve_dir = dirs_dict['learning_curve']

image_dir = os.path.join(dirs_dict['images_dir'], args['image_dir'])

csv_dict = parse_csv(os.path.join(dirs_dict['csvs_dir'], args['csv_name']), task)
label_num_classes = csv_dict['label_num_classes']
label_list = csv_dict['label_list']
num_inputs = csv_dict['num_inputs']


# Data Loadar
if task == 'deepsurv':
    from dataloader.dataloader_deepsurv import *
else:
    # when classification or regression
    if len(label_list) > 1:
        from dataloader.dataloader_multi import *
    else:
        from dataloader.dataloader import *
train_loader = dalaloader_mlp_cnn(args, csv_dict, image_dir, split_list=['train'], batch_size=batch_size, sampler=sampler)
val_loader = dalaloader_mlp_cnn(args, csv_dict, image_dir, split_list=['val'], batch_size=batch_size, sampler=sampler)

# Configure of training
#model = CreateModel_MLPCNN(mlp, cnn, label_list, num_inputs, num_classes, device=gpu_ids)
model = create_mlp_cnn(mlp, cnn, label_num_classes, num_inputs, device=gpu_ids)
criterion = set_criterion(criterion, device)
optimizer = set_optimizer(optimizer, model, lr)

best_weight = None
val_best_loss = None
val_best_epoch = None

if task == 'classification':
    loss_acc_dict = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
else:
    # When regression or deepsurv
    loss_acc_dict = {'train_loss': [], 'val_loss': []}

def update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len_dataloader, len_label_list, val_best_loss, val_best_epoch):
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


def execute_epoch_single_label(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else: # elif phase == 'val':
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_acc = 0.0

            for i, (ids, raw_outputs, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
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

                    else: # elif not(mlp is None) and not(cnn is None):
                    # When MLP+CNN
                        inputs_values_normed = inputs_values_normed.to(device)
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs_values_normed, images)

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

            loss_acc_dict, val_best_loss, val_best_epoch, update_flag = update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len(dataloader.dataset), len(label_list), val_best_loss, val_best_epoch)

            # Keep the best weight when epoch_loss is the lowest.
            if (phase == 'val' and update_flag):
                best_weight = copy.deepcopy(model.state_dict())
    return best_weight, val_best_loss, val_best_epoch, loss_acc_dict


def execute_epoch_multi_label(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else: # elif phase == 'val':
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_acc = 0

            for i, (ids, raw_outputs_dict, labels_dict, inputs_values_normed, images, splits) in enumerate(dataloader):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if not(mlp is None) and (cnn is None):
                        # When MLP only
                        inputs_values_normed = inputs_values_normed.to(device)
                        labels_multi = { label_name: labels.to(device) for label_name, labels in labels_dict.items() }
                        outputs = model(inputs_values_normed)

                    elif (mlp is None) and not(cnn is None):
                        # When CNN only
                        images = images.to(device)
                        labels_multi = { label_name: labels.to(device) for label_name, labels in labels_dict.items() }
                        outputs = model(images)

                    else: # elif not(mlp is None) and not(cnn is None):
                        # When MLP+CNN
                        inputs_values_normed = inputs_values_normed.to(device)
                        images = images.to(device)
                        labels_multi = { label_name: labels.to(device) for label_name, labels in labels_dict.items() }
                        outputs = model(inputs_values_normed, images)

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

            loss_acc_dict, val_best_loss, val_best_epoch, update_flag = update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len(dataloader.dataset), len(label_list), val_best_loss, val_best_epoch)

            # Keep the best weight when epoch_loss is the lowest.
            if (phase == 'val' and update_flag):
                best_weight = copy.deepcopy(model.state_dict())
    return best_weight, val_best_loss, val_best_epoch, loss_acc_dict


def execute_epoch_deepsurv(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict):
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else: #elif phase == 'val':
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_acc = 0

            for i, (ids, raw_outputs, labels, periods, inputs_values_normed, images, splits) in enumerate(dataloader):
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if not(mlp is None) and (cnn is None):
                    # When MLP only
                        inputs_values_normed = inputs_values_normed.to(device)
                        labels = labels.float().to(device)
                        periods = periods.float().to(device)
                        outputs = model(inputs_values_normed)

                    elif (mlp is None) and not(cnn is None):
                    # When CNN only
                        images = images.to(device)
                        labels = labels.to(device)
                        periods = periods.float().to(device)
                        outputs = model(images)

                    else: # elif not(mlp is None) and not(cnn is None):
                    # When MLP+CNN
                        inputs_values_normed = inputs_values_normed.to(device)
                        images = images.to(device)
                        labels = labels.float().to(device)
                        periods = periods.float().to(device)
                        outputs = model(inputs_values_normed, images)

                    risk_preds = outputs   # Just rename for clarity
                    loss = criterion(risk_preds, periods.reshape(-1,1), labels.reshape(-1,1), model)
                    
                    if (phase == 'train') and (torch.sum(labels).item() > 0):
                    # No backward when all labels are 0.
                    # To be specofic, loss(NegativeLogLikelihood) cannot be defined in this case.
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * labels.size(0)

            loss_acc_dict, val_best_loss, val_best_epoch, update_flag = update_loss_acc_dict(task, num_epochs, loss_acc_dict, epoch, phase, running_loss, running_acc, len(dataloader.dataset), len(label_list), val_best_loss, val_best_epoch)

            # Keep the best weight when epoch_loss is the lowest.
            if (phase == 'val' and update_flag):
                best_weight = copy.deepcopy(model.state_dict())
    return best_weight,val_best_loss,val_best_epoch,loss_acc_dict

# Training
print ('Training started...')
print(f"train_data = {len(train_loader.dataset)}")
print(f"  val_data = {len(val_loader.dataset)}")

if task == 'deepsurv':
    best_weight, val_best_loss, val_best_epoch, loss_acc_dict = execute_epoch_deepsurv(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict)
else:    
    # When classification or regression
    if len(label_list) > 1:
        # Multi-label outputs
        best_weight, val_best_loss, val_best_epoch, loss_acc_dict = execute_epoch_multi_label(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict)
    else:
        # Single-label output
        best_weight, val_best_loss, val_best_epoch, loss_acc_dict = execute_epoch_single_label(task, mlp, cnn, criterion, optimizer, num_epochs, device, label_list, train_loader, val_loader, model, val_best_loss, val_best_epoch, loss_acc_dict)
print('Training finished!')

# Save misc
date_now = datetime.datetime.now()
date_name = date_now.strftime('%Y-%m-%d-%H-%M-%S')
basename = make_basename(args, val_best_epoch, val_best_loss, date_name)

# Options
os.makedirs(train_opt_log_dir, exist_ok=True)
save_train_options(args, train_opt_log_dir, date_name)

# Weight
os.makedirs(weight_dir, exist_ok=True)
weight_path = os.path.join(weight_dir, basename) + '.pt'
torch.save(best_weight, weight_path)

# Learning curve
os.makedirs(learning_curve_dir, exist_ok=True)
learning_curve_path = os.path.join(learning_curve_dir, basename) + '.csv'
df_learning_curve = pd.DataFrame(loss_acc_dict)
df_learning_curve.to_csv(learning_curve_path, index=False)


# ---- EOF --------
