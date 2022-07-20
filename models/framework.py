#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary

from net import FeatureExtractor, MLPNet


class MultiMixin:
    # Only for MLP, ResNet, DenseNet
    def build_multi_classifier(self):
        if self.net_name.startswith('ResNet'):
            in_features = self.set_net(self.net_name).fc.in_features
        elif self.net_name.startswith('DenseNet'):
            in_features = self.set_net(self.net_name).classifier.in_features
        else:
            in_features = MLPNet(self.num_inputs)[-2].out_features

        classifier = nn.ModuleDict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifier[internal_label_name] = nn.Linear(in_features, num_classes)
        return classifier

    def multi_forward(self, out_features):
        output = dict()
        for internal_label_name, layer in self.classifier.items():
            output[internal_label_name] = layer(out_features)
        return output

    def get_output(self, multi_output, internal_label_name):
        return multi_output[internal_label_name]


class MultiWidget(nn.Module, FeatureExtractor, MultiMixin):
    """
    Class for a widght to inherit multiple classes simultaneously
    """
    pass


class MultiMLPNet(MultiWidget):
    def __init__(self, num_inputs, num_classes_in_internal_label):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = MLPNet(self.num_inputs)
        self.classifier = self.build_multi_classifier()

    def forward(self, x):
        out_feature = self.extractor(x)
        output = self.multi_forward(out_feature)
        return output


class MultiResNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self.create_extractor(net_name)
        self.classifier = self.build_multi_classifier()

    def forward(self, x):
        x = self.extractor(x)
        out_features = self.get_feature(x)
        output = self.multi_forward(out_features)
        return output


"""
# class MultiDenseNet(MultiWidget):
"""


class MultiEfficientNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self.create_extractor(net_name)
        self.classifier = self.build_multi_classifier()

    def build_multi_classifier(self):
        base_net = self.set_net(self.net_name)
        dropout = base_net.classifier[0].p
        in_features = base_net.classifier[1].in_features
        classifier = nn.ModuleDict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifier[internal_label_name] = nn.Sequential(
                                                nn.Dropout(p=dropout, inplace=False),
                                                nn.Linear(in_features, num_classes)
                                                )
        return classifier

    def forward(self, x):
        x = self.extractor(x)
        out_features = self.get_feature(x)
        output = self.multi_forward(out_features)
        return output



class MultiConvNeXt(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label

        self.extractor = self.create_extractor(net_name)
        self.classifier = self.build_multi_classifier()

    def build_multi_classifier(self):
        base_net = self.set_net(self.net_name)
        layer_norm = base_net.classifier[0]
        flatten = base_net.classifier[1]
        in_features = base_net.classifier[2].in_features

        classifier = nn.ModuleDict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifier[internal_label_name] = nn.Sequential(
                                                layer_norm,
                                                flatten,
                                                nn.Linear(in_features, num_classes)
                                                )
        return classifier

    def forward(self, x):
        x = self.extractor(x)
        out_features = self.get_feature(x)
        breakpoint()
        output = self.multi_forward(out_features)
        return output



# ! Dense, ConveNext
# ! extractor と classifier の次元が合わない



# MLP+CNN
# class MLPCNN(MultiWidget):




def create_model(args, split_provider, gpu_ids=None):
    pass
    # 1ch, 3ch
    # multi
    # MLP+CNN
