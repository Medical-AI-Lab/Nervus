#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *


# Hidden layers of MLP
hidden_layers_size = [256, 256, 256]


# Configure GPU or CPU
def ConfigDevice(model, gpu_ids):
    if gpu_ids:
        if torch.cuda.is_available():
            primary_gpu_id = gpu_ids[0]
            device_name = f'cuda:{primary_gpu_id}'

            torch.cuda.set_device(device_name)  # Set primary GPU

            model.to(device_name)
            model = torch.nn.DataParallel(model, gpu_ids)

        else:
            print('Error from ConfigDevice: No avalibale GPU on this machine. Use CPU.')
            exit()

    else:
        device = torch.device('cpu')
        model = model.to(device)

    return model



# MLP
class MLPNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        #super(MLPNet, self).__init__()
        super().__init__()

        self.dropout = 0.2
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers_size = [ self.num_inputs ] + hidden_layers_size + [ self.num_outputs ]

        self.mlp = self._build()


    # Build MLP
    def _build(self):
        layers = OrderedDict()

        for i in range(len(self.layers_size)-1):
            input_size = self.layers_size.pop(0)
            output_size = self.layers_size[0]

            if len(self.layers_size) >=2:
                layers['linear_' + str(i)] = nn.Linear(input_size, output_size)
                layers['relu_' + str(i)] = nn.ReLU()
                layers['dropout_' + str(i)] = nn.Dropout(self.dropout)
            else:
                # Output layer
                layers['linear_'+str(i)] = nn.Linear(input_size, output_size)

        return nn.Sequential(layers)


    def forward(self, inputs):
        return self.mlp(inputs)



# CNN
def CNN(cnn_name, num_classes):
    if cnn_name == 'B0':
        #cnn = EfficientNet.from_name('efficientnet-b0')
        cnn = models.efficientnet_b0(num_classes=num_classes)

    elif cnn_name == 'B2':
        #cnn = EfficientNet.from_name('efficientnet-b2')
        cnn = models.efficientnet_b2(num_classes=num_classes)

    elif cnn_name == 'B4':
        #cnn = EfficientNet.from_name('efficientnet-b4')
        cnn = models.efficientnet_b4(num_classes=num_classes)

    elif cnn_name == 'B6':
        #cnn = EfficientNet.from_name('efficientnet-b6')
        cnn = models.efficientnet_b6(num_classes=num_classes)

    elif cnn_name == 'ResNet18':
        cnn = models.resnet18(num_classes=num_classes)

    elif cnn_name == 'ResNet':
        cnn = models.resnet50(num_classes=num_classes)

    elif cnn_name == 'DenseNet':
        cnn = models.densenet161(num_classes=num_classes)

    else:
        print('Cannot specify such a CNN: {}.'.format(cnn_name))
        exit()

    return cnn



# MLP+CNN
class MLPCNN_Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, cnn_name, cnn_num_outputs):
        """
        # Memo
        num_inputs:     number of 'input_*' in csv
        num_outputs:    number of output of MLP+CNN
        cnn_num_outpus: number of output size to be passed to MLP
        """

        super().__init__()

        # CNN
        self.cnn_name = cnn_name
        self.cnn = CNN(self.cnn_name, len(['0', '1']))

        # Build MLP+CNN
        self.num_inputs = num_inputs     # Not include image
        self.cnn_num_outputs = cnn_num_outputs
        self.mlp_cnn_num_inputs = self.num_inputs + self.cnn_num_outputs
        self.mlp_cnn_num_outputs = num_outputs
        self.mlp = MLPNet(self.mlp_cnn_num_inputs, self.mlp_cnn_num_outputs)


    # Normalize tensor to [0, 1]
    def normalize_cnn_output(self, outputs_cnn):
        max = outputs_cnn.max()
        min = outputs_cnn.min()

        outputs_cnn_normed = (outputs_cnn - min) / (max - min)

        return outputs_cnn_normed



    def forward(self, inputs, images):
        outputs_cnn = self.cnn(images)                                  # Tensor [64, 2],   images.shape Tesor [64, 48]

        # Select likelihood of '1'
        outputs_cnn = outputs_cnn[:, 1]                                 # Numpy [64, 2] -> Numpy [64]
        outputs_cnn = outputs_cnn.reshape(len(outputs_cnn), 1)          # Numpy [64]    -> Numpy [64, 1]

        # Normalize
        outputs_cnn_normed = self.normalize_cnn_output(outputs_cnn)     # Numpy [64, 1] -> Tensor [64, 1]   Normalize bach_size-wise

        # Merge inputs with output from CNN
        inputs_images = torch.cat((inputs, outputs_cnn_normed), dim=1)  # Tensor [64, 48], Tensor [64, 1] -> Tensor [64, 48+1]


        outputs = self.mlp(inputs_images)

        return outputs



def CreateModel_MLPCNN(mlp, cnn, num_inputs, num_classes, device=[]):
    """
    num_input:    number of inputs of MLP or MLP+CNN
    num_classes:  number of outputs of MLP, CNN, or MLP+CNN
    """
    if not(mlp is None) and (cnn is None):
        # When MLP only
        model = MLPNet(num_inputs, num_classes)

    elif (mlp is None) and not(cnn is None):
        # When CNN only
        model = CNN(cnn, num_classes)

    elif not(mlp is None) and not(cnn is None):
        # When MLP+CNN
        CNN_NUM_OUTPUTS = 1        # number of outputs from CNN, whose shape is (batch_size, 1), to MLP
        model = MLPCNN_Net(num_inputs, num_classes, cnn, CNN_NUM_OUTPUTS)

    else:
        print('Invalid model: ' + mlp_or_cnn)
        exit()

    model = ConfigDevice(model, device)

    return model


# ----- EOF -----
