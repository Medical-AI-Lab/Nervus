#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import copy

from lib.util import *
from lib.align_env import *
from options.train_options import TrainOptions
from config.criterion import Criterion
from config.optimizer import Optimizer
from dataloader.dataloader_multi import *
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
label_list = csv_dict['label_list']
num_classes = csv_dict['num_classes']
num_inputs = csv_dict['num_inputs']



# Data Loadar
train_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(args, csv_dict, image_dir, split_list=['train'], batch_size=batch_size, sampler=sampler)
val_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(args, csv_dict, image_dir, split_list=['val'], batch_size=batch_size, sampler=sampler)

# Configure of training
model = CreateModel_MLPCNN(mlp, cnn, label_list, num_inputs, num_classes, device=gpu_ids)
criterion = Criterion(criterion)
optimizer = Optimizer(optimizer, model, lr)



# Classification
#def train_classification():
print ('Training started...')
print('train_data = {num_train_data}'.format(num_train_data=len(train_loader.dataset)))
print('val_data = {num_val_data}'.format(num_val_data=len(val_loader.dataset)))

best_weight = None
val_best_loss = None
val_best_epoch = None
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
update_comment = ''


for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        elif phase == 'val':
            model.eval()
            dataloader = val_loader

        running_loss = 0.0
        running_acc = 0

        for i, (ids, labels_dict, inputs_values_normed, images, splits) in enumerate(dataloader):
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

            running_loss += loss.item() * labels.size(0)
 
            if task == 'classification':
                for label_name, labels in labels_multi.items():
                    running_acc_i = (torch.sum(preds_multi[label_name] == labels.data).item())
                    running_acc += running_acc_i
            else:
                pass


        if task == 'classification':
            epoch_loss = running_loss / (len(dataloader.dataset) * len(label_list))
            epoch_acc = running_acc / (len(dataloader.dataset) * len(label_list))
        else:
            epoch_loss = running_loss / (len(dataloader.dataset) * len(label_list))


        if task == 'classification':
            if phase == 'train':
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            elif phase == 'val':
                val_loss_list.append(epoch_loss)
                val_acc_list.append(epoch_acc)
        else:
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            elif phase == 'val':
                val_loss_list.append(epoch_loss)


        # Keep the best weight when epoch_loss is the lowest.
        if (phase == 'val' and (val_best_loss is None or (epoch_loss < val_best_loss))):
            val_best_loss = epoch_loss
            val_best_epoch = epoch + 1
            best_weight = copy.deepcopy(model.state_dict())
            update_comment = ' Updated val_best_loss!'
        else:
            update_comment = ''

    if task == 'classification':
        print(('epoch [{ith_epoch:>3}/{num_epochs:<3}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' + update_comment)
        .format(ith_epoch=epoch+1, num_epochs=num_epochs, train_loss=train_loss_list[-1], val_loss=val_loss_list[-1], val_acc=val_acc_list[-1]))
    else:
        print(('epoch [{ith_epoch:>3}/{num_epochs:<3}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}' + update_comment)
        .format(ith_epoch=epoch+1, num_epochs=num_epochs, train_loss=train_loss_list[-1], val_loss=val_loss_list[-1]))
    

print('Training finished!')


# Save misc
dt_now = datetime.datetime.now()
dt_name = dt_now.strftime('%Y-%m-%d-%H-%M-%S')

# Options
os.makedirs(train_opt_log_dir, exist_ok=True)
save_train_options(args, train_opt_log_dir, dt_name)

basename = make_basename(args, val_best_epoch, val_best_loss, dt_name)

# Weight
os.makedirs(weight_dir, exist_ok=True)
weight_path = os.path.join(weight_dir, basename) + '.pt'
torch.save(best_weight, weight_path)

# Learning curve
os.makedirs(learning_curve_dir, exist_ok=True)
learning_curve_path = os.path.join(learning_curve_dir, basename) + '.csv'

if task == 'classification':
    df_learning_curve = pd.DataFrame({'train_loss': train_loss_list,
                                       'train_acc':  train_acc_list,
                                       'val_loss':   val_loss_list,
                                       'val_acc':    val_acc_list
                                    })
else:
    df_learning_curve = pd.DataFrame({'train_loss': train_loss_list,
                                      'val_loss':   val_loss_list,
                                    })

df_learning_curve.to_csv(learning_curve_path, index=False)



# ---- EOF --------
