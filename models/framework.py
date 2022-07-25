#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchinfo import summary

from net import BaseNet

from abc import ABC, abstractmethod

import sys
from pathlib import Path
sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('models.framework')


class MultiMixin:
    def multi_forward(self, out_features):
        output = dict()
        for internal_label_name, unit in self.multi_classifier.items():
            output[internal_label_name] = unit(out_features)
        return output

    def get_output(self, multi_output, internal_label_name):
        return multi_output[internal_label_name]


class MultiWidget(nn.Module, BaseNet, MultiMixin):
    """
    Class for a widght to inherit multiple classes simultaneously
    """
    pass


class MultiNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channels=None, vit_image_size=None):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channels = in_channels
        self.vit_image_size = vit_image_size

        self.extractor = self.constuct_extractor(self.net_name, mlp_num_inputs=self.mlp_num_inputs, in_channels=self.in_channels, vit_image_size=self.vit_image_size)
        self.multi_classifier = self.construct_multi_classifier(self.net_name, self.num_classes_in_internal_label)

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiNetFusion(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channels=None, vit_image_size=None):
        assert (net_name != 'MLP'), "net_name should not be MLP."

        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channels = in_channels
        self.vit_image_size = vit_image_size

        # Extractor of MLP and Net
        self.extractor_mlp = self.constuct_extractor('MLP', mlp_num_inputs=self.mlp_num_inputs)
        self.extractor_net = self.constuct_extractor(self.net_name, in_channels=self.in_channels, vit_image_size=self.vit_image_size)
        self.aux_module = self.construct_aux_module(self.net_name)

        # Intermediate MLP
        self.in_featues_from_mlp = self.get_classifier_in_features('MLP')
        self.in_features_from_net = self.get_classifier_in_features(self.net_name)
        self.inter_mlp_in_feature = self.in_featues_from_mlp + self.in_features_from_net
        self.inter_mlp = self.MLPNet(self.inter_mlp_in_feature, inplace=False)  # ! If inplace==True, cannot backweard  Check!

        # Multi classifier
        self.multi_classifier = self.construct_multi_classifier('MLP', num_classes_in_internal_label)

    def forward(self, x_mlp, x_net):
        out_mlp = self.extractor_mlp(x_mlp)
        out_net = self.extractor_net(x_net)
        out_net = self.aux_module(out_net)

        out_features = torch.cat([out_mlp, out_net], dim=1)
        out_features = self.inter_mlp(out_features)
        output = self.multi_forward(out_features)
        return output


def create_model(args, split_provider, device=None, gpu_ids=None):
    # 1ch, 3ch
    # MLP, CNN, or MLP+CNN
    # MultiNet or MultiNetFusion
    mlp = args.mlp
    net = args.net
    in_channels = args.in_channels
    vit_image_size = args.vit_image_size
    mlp_num_inputs = len(split_provider.input_list)
    num_classes_in_internal_label = split_provider.num_classes_in_internal_label

    if (mlp is not None) and (net is None):
        multi_net = MultiNet('MLP', num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channels=None, vit_image_size=None)
    elif (mlp is None) and (net is not None):
        multi_net = MultiNet(net, num_classes_in_internal_label, mlp_num_inputs=None, in_channels=in_channels, vit_image_size=vit_image_size)
    elif (mlp is not None) and (net is not None):
        multi_net = MultiNetFusion(net, num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channels=in_channels, vit_image_size=vit_image_size)
    else:
        logger.error("Cannot identify net type.")
    return multi_net

"""
class BaseModel(ABC):
    def __init__(self, args, split_provider, gpu_ids=None):
        self.args = args
        self.sp = split_provider
        self.mlp = args.mlp
        self.net = args.mlp
        self.criterion_name = args.criterion
        self.optimizer_name = args.optimizer
        self.gpu_ids = gpu_ids
        self.device = self.gpu_ids[0]

    @property
    def train(self):
        self.net.train()

    @property
    def eval(self):
        self.net.eval()

    @abstractmethod
    def set_data(self, data):
        pass
        # data = {
        #        'Filename': filename,
        #        'ExamID': examid,
        #        'Institution': institution,
        #        'raw_labels': raw_label_dict,
        #        'internal_labels': internal_label_dict,
        #        'inputs': inputs_value,
        #        'image': image,
        #        'split': split
        #        }

    @abstractmethod
    def forward(self):
        pass

    def backward(self):
        pass

    def optimize_paramters(self):
        pass


class ModelMLP(BaseModel):
    def __init__(self, args, split_provider):
        super().__init__()

        self.net = ...

    def set_data(self, data):
        inputs = data['inputs'].to(self.device)
        return inputs

    def forward(self, inputs):
        output = self.net(inputs)


class ModelNet(BaseModel):
    def __init__(self, args, split_provider):
        super().__init__()

    def set_data(self, data):
        image = data['image'].to(self.device)
        return image

    def forward(self, image):
        output = self.net(image)


class ModelFusion(BaseModel):
    def __init__(self, args, split_provider):
        super().__init__()

    def set_data(self, data):
        image = data['image'].to(self.device)
        inputs = data['inputs'].to(self.device)
        return inputs, image

    def forward(self, inputs, image):
        output = self.net(inputs, image)


class Loss:
    def __init__(self, label_list):
        self.label_list = label_list
        train_loss = []
        val_loss = []
"""
