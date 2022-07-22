#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchinfo import summary

from net import BaseNet


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
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, vit_image_size=None):
        super().__init__()

        self.net_name = net_name
        self.mlp_num_inputs = mlp_num_inputs
        self.vit_image_size = vit_image_size
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self.constuct_extractor(self.net_name, mlp_num_inputs=self.mlp_num_inputs, vit_image_siz=self.vit_image_size)
        self.multi_classifier = self.construct_multi_classifier(self.net_name, self.num_classes_in_internal_label)

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


# MLP+(CNN or ViT)
class MultiNetFusion(MultiWidget):
    def __init__(self, num_inputs, net_name, num_classes_in_internal_label):
        super().__init__()

        self.num_inputs = num_inputs
        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label

        self.mlp = self.MLPNet(self.num_inputs)
        self.cnn_vit = self._create_extracrtor(self.net_name)
        self.num_inputs_all = 1
        self.multi_classifier = self.MLPNet(self.num_inputs)

    def _create_feature_extratror(self):
        # To be defined
        pass

    def forward(self, inputs, x):
        # To be defined
        pass


def create_model(args, split_provider, gpu_ids=None):
    pass
    # 1ch, 3ch
    # MLP, CNN, or MLP+CNN
