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

class MultiMLPNet(MultiWidget):
    def __init__(self, num_inputs, num_classes_in_internal_label):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self.MLPNet(self.num_inputs)
        self.multi_classifier = self._construct_multi_classifier()

    def _construct_multi_classifier(self):
        in_features = self.extractor[-2].out_features
        classifiers = dict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifiers[internal_label_name] = nn.Linear(in_features, num_classes)

        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiResNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self._create_feature_extractor()
        self.multi_classifier = self._construct_multi_classifier()

    def _create_feature_extractor(self):
        extractor = self.set_net(self.net_name)
        extractor.fc = self.DUMMY
        return extractor

    def _construct_multi_classifier(self):
        in_features = self.set_net(self.net_name).fc.in_features
        classifiers = dict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifiers[internal_label_name] = nn.Linear(in_features, num_classes)

        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiDenseNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self._create_feature_extractor()
        self.multi_classifier = self._construct_multi_classifier()

    def _create_feature_extractor(self):
        extractor = self.set_net(self.net_name)
        extractor.classifier = self.DUMMY
        return extractor

    def _construct_multi_classifier(self):
        in_features = self.set_net(self.net_name).classifier.in_features
        classifiers = dict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifiers[internal_label_name] = nn.Linear(in_features, num_classes)

        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiEfficientNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self._create_feature_extractor()
        self.multi_classifier = self._construct_multi_classifier()

    def _create_feature_extractor(self):
        extractor = self.set_net(self.net_name)
        extractor.classifier = self.DUMMY
        return extractor

    def _construct_multi_classifier(self):
        base_classifier = self.set_net(self.net_name).classifier
        dropout = base_classifier[0].p
        in_features = base_classifier[1].in_features
        classifiers = dict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifiers[internal_label_name] = nn.Sequential(
                                                    nn.Dropout(p=dropout, inplace=False),
                                                    nn.Linear(in_features, num_classes)
                                                )
        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiConvNeXt(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.extractor = self._create_feature_extractor()
        self.multi_classifier = self._construct_multi_classifier()

    def _create_feature_extractor(self):
        extractor = self.set_net(self.net_name)
        extractor.classifier = self.DUMMY
        return extractor

    def _construct_multi_classifier(self):
        base_classifier = self.set_net(self.net_name).classifier
        layer_norm = base_classifier[0]
        flatten = base_classifier[1]
        in_features = base_classifier[2].in_features
        classifiers = dict()
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            classifiers[internal_label_name] = nn.Sequential(
                                                    layer_norm,
                                                    flatten,
                                                    nn.Linear(in_features, num_classes)
                                                )
        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


# MLP+CNN
class MultiMLPCNN(MultiWidget):
    def __init__(self, num_inputs, net_name, num_classes_in_internal_label):
        super().__init__()

        self.num_inputs = num_inputs
        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label

        self.mlp = self.MLPNet(self.num_inputs)
        self.cnn = self._create_extracrtor(self.net_name)
        self.num_inputs_all = 1
        self.multi_classifier = self.MLPNet(self.num_inputs)

    def _create_feature_extratror(self):
        pass

    def forward(self, inputs, x):
        pass


def create_model(args, split_provider, gpu_ids=None):
    pass
    # 1ch, 3ch
    # MLP, CNN, or MLP+CNN
