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


DUMMY_LAYER = nn.Identity()  # No paramaters


# MLP
class MLPNet(nn.Module):
    def __init__(self, label_list, num_inputs, num_outputs):
        #super(MLPNet, self).__init__()
        super().__init__()

        self.dropout = 0.2

        self.label_list = label_list
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers_size = [ self.num_inputs ] + hidden_layers_size + [ self.num_outputs ]
        self.fc_names = [ ('fc_' + label) for label in self.label_list ]
        self.mlp = self._build()

    # Build MLP
    def _build(self):
        layers = OrderedDict()

        for i in range(len(self.layers_size)-1):
            input_size = self.layers_size.pop(0)
            output_size = self.layers_size[0]

            if len(self.layers_size) >=2:
                layers['linear_' + str(i)] = nn.Linear(input_size, output_size)
                layers['relu_' + str(i)] = nn.ReLU()                    # Default: inplace=False
                layers['dropout_' + str(i)] = nn.Dropout(self.dropout)  # Default: inplace=False
            else:
                # layers['linear_'+str(i)] = nn.Linear(input_size, output_size)
                layers['dummy'] = DUMMY_LAYER   # It is OK without this, but add fir 

        self.extractor = nn.Sequential(layers)

        # Add multi-fc
        self.fc_multi = nn.ModuleDict({
                            fc_name : nn.Linear(input_size, self.num_outputs)
                            for fc_name in self.fc_names
                        })

        _mlp = OrderedDict([
                            ('extractor', self.extractor),
                            ('fc_multi', self.fc_multi)
                        ])

        return _mlp


    def forward(self, inputs):
        inputs = self.mlp['extractor'](inputs)
        output_multi = {
                        fc_name: self.fc_multi[fc_name](inputs)
                        for fc_name in self.fc_names
                       }

        return output_multi



# For ResNet family
class ResNet_Multi(nn.Module):
    def __init__(self, base_model, label_list, num_outputs):
        super().__init__()

        self.extractor = base_model 
        self.label_list = label_list
        self.num_outputs = num_outputs

        # Construct fc layres
        self.input_size_fc = self.extractor.fc.in_features
        self.fc_names = [ ('fc_' + label) for label in self.label_list ]
        self.fc_multi = nn.ModuleDict({
                            fc_name : nn.Linear(self.input_size_fc, self.num_outputs)
                            for fc_name in self.fc_names
                        })

        # Replace the original fc layer
        self.extractor.fc = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)

        # Fork forwarding
        output_multi =  {
                            fc_name : self.fc_multi[fc_name](x)
                            for fc_name in self.fc_names
                        }

        return output_multi



# For EfficientNet family
class EfficientNet_Multi(nn.Module):
    def __init__(self, base_model, label_list, num_outputs):
        super().__init__()

        self.extractor = base_model
        self.label_list = label_list
        self.num_outputs = num_outputs

        # Construct fc layres
        self.input_size_fc = base_model.classifier[1].in_features
        _dropout = base_model.classifier[0]     # Originally, Dropout(p=0.2, inplace=True)
        # Note:
        # Unless inplace=False, cannot backword
        # because gradient is deleted bue to inplace
        _dropout.inplace = False
        
        
        _linear = nn.Linear(self.input_size_fc, self.num_outputs)

        self.fc_multi = nn.ModuleDict({
                                        ('block_'+label_name) : 
                                                nn.Sequential(
                                                        OrderedDict([
                                                            ('0_'+label_name, _dropout),  
                                                            ('1_'+label_name, _linear)
                                                        ]))
                                        for label_name in self.label_list
                                    })

        # Replace the original classifier with nn.Identity()
        self.extractor.classifier = DUMMY_LAYER

    def forward(self, x):
       x = self.extractor(x)

       # Fork forwarding
       output_multi =  {
                        ('block_'+label_name) : self.fc_multi['block_'+label_name](x)
                        for label_name in self.label_list
                       }

       return output_multi


# CNN
def CNN(cnn_name, label_list, num_outputs):
    if cnn_name == 'B0':
        cnn = models.efficientnet_b0

    elif cnn_name == 'B2':
        cnn = models.efficientnet_b2

    elif cnn_name == 'B4':
        cnn = models.efficientnet_b4

    elif cnn_name == 'B6':
        cnn = models.efficientnet_b6

    elif cnn_name == 'ResNet':
        cnn = models.resnet50

    elif cnn_name == 'ResNet18':
        cnn = models.resnet18

    elif cnn_name == 'ResNext':
        cnn = models.resnext50_32x4d

    elif cnn_name == 'DenseNet':
        cnn = models.densenet161

    else:
        print('Cannot specify such a CNN: {}.'.format(cnn_name))
        exit()


    if not(label_list is None):
        # When CNN only -> make multi
        if cnn_name.startswith('ResNet'):
            cnn = ResNet_Multi(cnn(), label_list, num_outputs)

        elif cnn_name.startswith('B'):
            cnn = EfficientNet_Multi(cnn(), label_list, num_outputs)
            
        elif cnn_name.startswith('ResNext'):
            print('To be defined.')

        elif cnn_name.startswith('DenseNet'):
            print('To be defined.')

    else:
        # When MLP+CNN
        cnn = cnn(num_classes=num_outputs)


    return cnn



# MLP+CNN
class MLPCNN_Net(nn.Module):
    def __init__(self, label_list, num_inputs, num_outputs, cnn_name, cnn_num_outputs):
        """
        # Memo
        num_inputs:     number of 'input_*' in csv
        num_outputs:    number of output of MLP+CNN
        cnn_num_outpus: number of output size to be passed to MLP
        """

        super().__init__()

        self.label_list = label_list
        self.num_inputs = num_inputs             # Not include image
        self.cnn_num_outputs = cnn_num_outputs
        self.cnn_name = cnn_name
        self.mlp_cnn_num_outputs = num_outputs

        self.mlp_cnn_num_inputs = self.num_inputs + self.cnn_num_outputs
        
        self.mlp = MLPNet(self.label_list, self.mlp_cnn_num_inputs, self.mlp_cnn_num_outputs)
        self.cnn = CNN(self.cnn_name, None, len(['pred_n_label_x', 'pred_p_label_x']))         # No multi-oputput


    # Normalize tensor to [0, 1]
    def normalize_cnn_output(self, outputs_cnn):
        max = outputs_cnn.max()
        min = outputs_cnn.min()

        outputs_cnn_normed = (outputs_cnn - min) / (max - min)

        return outputs_cnn_normed



    def forward(self, inputs, images):
        outputs_cnn = self.cnn(images)                                  # [64, 256, 256]  -> [64, 2]

        # Select likelihood of '1'
        outputs_cnn = outputs_cnn[:, 1]                                 # [64, 2] -> Numpy [64]
        outputs_cnn = outputs_cnn.reshape(len(outputs_cnn), 1)          # Numpy [64] -> Numpy [64, 1]

        # Normalize
        outputs_cnn_normed = self.normalize_cnn_output(outputs_cnn)     # Numpy [64, 1] -> Tensor [64, 1]   Normalize bach_size-wise

        # Merge inputs with output from CNN
        inputs_images = torch.cat((inputs, outputs_cnn_normed), dim=1)  # Tensor [64, 24] + Tensor [64, 1] -> Tensor [64, 24+1]


        outputs = self.mlp(inputs_images)

        return outputs



def CreateModel_MLPCNN(mlp, cnn, label_list, num_inputs, num_outputs, device=[]):
    """
    num_input:    number of inputs of MLP or MLP+CNN
    num_classes:  number of outputs of MLP, CNN, or MLP+CNN
    """
    if not(mlp is None) and (cnn is None):
        # When MLP only
        model = MLPNet(label_list, num_inputs, num_outputs)

    elif (mlp is None) and not(cnn is None):
        # When CNN only
        model = CNN(cnn, label_list, num_outputs)

    elif not(mlp is None) and not(cnn is None):
        # When MLP+CNN
        CNN_NUM_OUTPUTS = 1        # number of outputs from CNN, whose shape is (batch_size, 1), to MLP
        model = MLPCNN_Net(label_list, num_inputs, num_outputs, cnn, CNN_NUM_OUTPUTS)

    else:
        print('\nInvalid model.\n')
        exit()

    model = ConfigDevice(model, device)

    return model




# Get outputs besed on label_name or value_name.
def get_layer_output(outputs_multi, label_name):
    output_layer_names = outputs_multi.keys()

    # output_layer_name should be uniq
    # Eg.
    # Label_1 -> fc_Label_1  |  Label_1 -> block_Label_1
    layer_name = [ output_layer_name
                   for output_layer_name in output_layer_names if  output_layer_name.endswith(label_name)
                 ][0]

    outputs = outputs_multi[layer_name]

    return outputs


# ----- EOF -----
