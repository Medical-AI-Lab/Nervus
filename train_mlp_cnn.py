#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import copy

from lib.options import Options
from lib.static import Static
from lib.util import *

from config.criterion import Criterion
from config.optimizer import Optimizer

from dataloader.dataloader_mlp_cnn import *
from config.mlp_cnn import CreateModel_MLPCNN


args = Options().parse() # dict

# Align args by adding additional infomration
args = Static(args, 'train').parse()


# Check validity of options
# Options().is_option_valid(args)
# Static().is_option_valid(args) # ???

# Print options
# Options().print_options(args)
# Static().print_options(args)  # ???


gpu_ids = args['gpu_ids']
device = set_device(gpu_ids)

mlp = args['mlp']
cnn = args['cnn']

num_classes = args['num_classes']
num_inputs = args['num_inputs']

criterion = args['criterion']
optimizer = args['optimizer']
lr = args['lr']
num_epochs = args['epochs']

batch_size = args['batch_size']
sampler = args['sampler']

train_opt_log_dir = args['train_opt_log_dir']
weight_dir = args['weight_dir']
learning_curve_dir = args['learning_curve_dir']


# Data Loadar
train_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(args, split_list=['train'], batch_size=batch_size, sampler=sampler)
val_loader = MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(args, split_list=['val'], batch_size=batch_size, sampler=sampler)

# Configure of training
model = CreateModel_MLPCNN(mlp, cnn, num_inputs, num_classes, device=gpu_ids)
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

        for i, (ids, labels, inputs_values_normed, images, splits) in enumerate(dataloader):
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

                elif not(mlp is None) and not(cnn is None):
                    # When MLP+CNN
                    inputs_values_normed = inputs_values_normed.to(device)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs_values_normed, images)


                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_acc += (torch.sum(preds == labels.data)).item()

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc = running_acc / len(dataloader.dataset)


        if phase == 'train':
            train_loss_list.append(epoch_loss)
            train_acc_list.append(epoch_acc)
        else:
            val_loss_list.append(epoch_loss)
            val_acc_list.append(epoch_acc)

        # Keep the best weight when epoch_loss is the lowest.
        if (phase == 'val' and (val_best_loss is None or (epoch_loss < val_best_loss))):
            val_best_loss = epoch_loss
            val_best_epoch = epoch + 1
            best_weight = copy.deepcopy(model.state_dict())
            update_comment = ' Updated val_best_loss!'
        else:
            update_comment = ''

    print(('epoch [{ith_epoch:>3}/{num_epochs:<3}], train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' + update_comment)
    .format(ith_epoch=epoch+1, num_epochs=num_epochs, train_loss=train_loss_list[-1], val_loss=val_loss_list[-1], val_acc=val_acc_list[-1]))

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
df_learning_curve = pd.DataFrame({'train_loss': train_loss_list,
                                  'train_acc':  train_acc_list,
                                  'val_loss':   val_loss_list,
                                  'val_acc':    val_acc_list})
df_learning_curve.to_csv(learning_curve_path, index=False)



# ---- EOF --------
