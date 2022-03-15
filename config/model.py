#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import *

logger = NervusLogger.get_logger('config.model')

#model = mlp_net(label_num_classes, num_inputs)
# MLP
class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.hidden_layers_size = [256, 256, 256]   # Hidden layers of MLP
        self.dropout = 0.2
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.layers_size = [self.num_inputs] + self.hidden_layers_size + [self.num_outputs]
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
                layers['fc'] = nn.Linear(input_size, output_size)   # Output layer
        return nn.Sequential(layers)

    def forward(self, inputs):
        return self.mlp(inputs)


DUMMY_LAYER = nn.Identity()  # No paramaters


# For MLP family
class MLP_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.mlp = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(self.label_num_classes.keys())
        self.fc_names = [('fc_' + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = self.mlp.fc.in_features
        self.fc_multi = nn.ModuleDict({
                            ('fc_' + label_name) : nn.Linear(self.input_size_fc, num_outputs)
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original fc layer
        self.mlp.fc = DUMMY_LAYER

    def forward(self, inputs):
        inputs = self.mlp(inputs)
        # Fork forwarding
        output_multi = {fc_name: self.fc_multi[fc_name](inputs) for fc_name in self.fc_names}
        return output_multi


# For ResNet family
class ResNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        self.fc_names = [('fc_' + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = self.extractor.fc.in_features
        self.fc_multi = nn.ModuleDict({
                            ('fc_' + label_name) : nn.Linear(self.input_size_fc, num_outputs)
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original fc layer
        self.extractor.fc = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)
        # Fork forwarding
        output_multi = {fc_name : self.fc_multi[fc_name](x) for fc_name in self.fc_names}
        return output_multi


# For DenseNet family
class DenseNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        self.fc_names = [('fc_' + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = self.extractor.classifier.in_features
        self.fc_multi = nn.ModuleDict({
                            ('fc_' + label_name) : nn.Linear(self.input_size_fc, num_outputs)
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original fc layer
        self.extractor.classifier = DUMMY_LAYER

    def forward(self, x):
        x = self.extractor(x)
        # Fork forwarding
        output_multi =  {fc_name : self.fc_multi[fc_name](x) for fc_name in self.fc_names}
        return output_multi


# For EfficientNet family
class EfficientNet_Multi(nn.Module):
    def __init__(self, base_model, label_num_classes):
        super().__init__()
        self.extractor = base_model
        self.label_num_classes = label_num_classes
        self.label_list = list(label_num_classes.keys())
        self.block_names = [('block_' + label_name) for label_name in self.label_list]

        # Construct fc layres
        self.input_size_fc = base_model.classifier[1].in_features
        self._dropout = base_model.classifier[0]    # Originally, Dropout(p=0.2, inplace=True)
        # Note: If inplace=True, cannot backword, because gradients are deleted bue to inplace
        self._dropout.inplace = False
        self.fc_multi = nn.ModuleDict({
                            ('block_'+label_name) : nn.Sequential(
                                                        OrderedDict([
                                                        ('0_'+label_name, self._dropout),
                                                        ('1_'+label_name, nn.Linear(self.input_size_fc, num_outputs))
                                                    ]))
                            for label_name, num_outputs in self.label_num_classes.items()
                        })

        # Replace the original classifier
        self.extractor.classifier = DUMMY_LAYER

    def forward(self, x):
       x = self.extractor(x)
       # Fork forwarding
       output_multi = {block_name: self.fc_multi[block_name](x) for block_name in self.block_names}
       return output_multi


def mlp_net(num_inputs, label_num_classes):
    label_list = list(label_num_classes.keys())
    num_outputs_first_label = label_num_classes[label_list[0]]
    mlp_base = MLP(num_inputs, num_outputs_first_label)        # Once make MLP for the first label only
    if len(label_list) > 1:
        mlp = MLP_Multi(mlp_base.mlp, label_num_classes)
    else:
        mlp = mlp_base
    return mlp


def conv_net(cnn_name, label_num_classes):
    if cnn_name == 'B0':
        cnn = models.efficientnet_b0

    elif cnn_name == 'B2':
        cnn = models.efficientnet_b2

    elif cnn_name == 'B4':
        cnn = models.efficientnet_b4

    elif cnn_name == 'B6':
        cnn = models.efficientnet_b6

    elif cnn_name == 'ResNet18':
        cnn = models.resnet18

    elif cnn_name == 'ResNet':
        cnn = models.resnet50

    elif cnn_name == 'DenseNet':
        cnn = models.densenet161

    else:
        logger.error(f"No specified CNN: {cnn_name}.")

    label_list = list(label_num_classes.keys())
    if len(label_list) > 1 :
        # When CNN only -> make multi
        if cnn_name.startswith('ResNet'):
            cnn = ResNet_Multi(cnn(), label_num_classes)

        elif cnn_name.startswith('B'):
            cnn = EfficientNet_Multi(cnn(), label_num_classes)

        else:
            cnn = DenseNet_Multi(cnn(), label_num_classes)
    else:
        # When Single-label output or MLP+CNN
        num_outputs_first_label = label_num_classes[label_list[0]]
        cnn = cnn(num_classes=num_outputs_first_label)
    return cnn


# MLP+CNN
class MLPCNN_Net(nn.Module):
    def __init__(self, cnn_name, num_inputs, label_num_classes):
        super().__init__()
        self.num_inputs = num_inputs           # Not include image
        self.label_num_classes = label_num_classes
        self.label_list = list(self.label_num_classes.keys())
        self.cnn_name = cnn_name
        self.cnn_num_outputs_pass_to_mlp = 1   # POSITIVE
        self.mlp_cnn_num_inputs = self.num_inputs + self.cnn_num_outputs_pass_to_mlp
        self.dummy_label_num_classes = {'dummy_label': len(['pred_n_label_x', 'pred_p_label_x'])}  # Before passing to MLP, do binary classification
        self.cnn = conv_net(self.cnn_name, self.dummy_label_num_classes)          # Non multi-label output
        self.mlp = mlp_net(self.mlp_cnn_num_inputs, self.label_num_classes)

    def normalize_cnn_output(self, outputs_cnn):
        # Note: Cannot use sklearn.preprocessing.MinMaxScaler() because sklearn does not support GPU.
        max = outputs_cnn.max()
        min = outputs_cnn.min()
        if min == max:
            outputs_cnn_normed = outputs_cnn - min   # ie. 0
        else:
            outputs_cnn_normed = (outputs_cnn - min) / (max - min)
        return outputs_cnn_normed

    def forward(self, inputs, images):
        outputs_cnn = self.cnn(images)                                  # [64, 256, 256]  -> [64, 2]

        # Select likelihood of '1'
        outputs_cnn = outputs_cnn[:, self.cnn_num_outputs_pass_to_mlp]  # [64, 2] -> Numpy [64]
        outputs_cnn = outputs_cnn.reshape(len(outputs_cnn), 1)          # Numpy [64] -> Numpy [64, 1]

        # Normalize
        outputs_cnn_normed = self.normalize_cnn_output(outputs_cnn)     # Numpy [64, 1] -> Tensor [64, 1]   Normalize bach_size-wise

        # Merge inputs with output from CNN
        inputs_images = torch.cat((inputs, outputs_cnn_normed), dim=1)  # Tensor [64, 24] + Tensor [64, 1] -> Tensor [64, 24+1]
        outputs = self.mlp(inputs_images)
        return outputs

def create_mlp_cnn(mlp, cnn, num_inputs, num_classes_in_label, gpu_ids=[]):
    """
    num_input: number of inputs of MLP or MLP+CNN
    """
    if (mlp is not None) and (cnn is None):
        # When MLP only
        model = mlp_net(num_inputs, num_classes_in_label)
    elif (mlp is None) and (cnn is not None):
        # When CNN only
        model = conv_net(cnn, num_classes_in_label)
    else:
        # When MLP+CNN
        # Set the number of outputs from CNN to MLP as 1, then
        # the shape of outputs from CNN is [batgch_size,1]
        model = MLPCNN_Net(cnn, num_inputs, num_classes_in_label)

    #model = config_device(model, gpu_ids)
    device = set_device(gpu_ids)
    model.to(device)
    if gpu_ids:
        model = torch.nn.DataParallel(model, gpu_ids)
    else:
        pass
    return model


# Extract outputs of label_name
def get_layer_output(outputs_multi, label_name):
    output_layer_names = outputs_multi.keys()
    layer_name = [output_layer_name for output_layer_name in output_layer_names if output_layer_name.endswith(label_name)][0]
    output_layer = outputs_multi[layer_name]
    return output_layer


def predict_by_model(model, hasMLP, hasCNN, device, inputs_values_normed, images) -> torch.Tensor:
    if hasMLP and hasCNN: # elif not(mlp is None) and not(cnn is None):
    # When MLP+CNN
        inputs_values_normed = inputs_values_normed.to(device)
        images = images.to(device)
        outputs = model(inputs_values_normed, images)
    elif hasMLP:
    # When MLP only
        inputs_values_normed = inputs_values_normed.to(device)
        outputs = model(inputs_values_normed)
    elif hasCNN:
    # When CNN only
        images = images.to(device)
        outputs = model(images)

    return outputs
