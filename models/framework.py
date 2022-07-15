#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch.nn as nn
from torchinfo import summary

from net import FeatureExtractorMixin


class MultiLabelMixin:
    def fork_forward(self, x):
        multi_output = dict()
        for fc_name, fc in self.multi_fc.items():
            multi_output[fc_name] = fc(x)
        return multi_output

    def get_output_of_label(self, internal_label_name):
        return self.multi_outputs['fc_' + internal_label_name]


class MultiLabelWidget(nn.Module, FeatureExtractorMixin, MultiLabelMixin):
    """
    Class for a widght to inherit multiple base classes simultaneously
    """
    pass


class MultiMLPNet(MultiLabelWidget):
    def __init__(self, num_inputs, num_classes_in_internal_label):
        super().__init__()

        self.num_inputs = num_inputs
        self.num_classes_in_internal_label = num_classes_in_internal_label

        _mlp = self.set_net('MLPNet')(self.num_inputs)
        self.extractor = self.create_extractor(_mlp, 'MLPNet')   # * _mlp should be in nn.Module.
        self.multi_fc = self._build_multi_fc()

    def _build_multi_fc(self):
        _multi_fc = nn.ModuleDict()
        _fc_in_features = self.extractor.mlp[-2].out_features
        for internal_label_name, num_classes in self.num_classes_in_internal_label.items():
            _multi_fc['fc_' + internal_label_name] = nn.Linear(_fc_in_features, num_classes)
        return _multi_fc

    def forward(self, x):
        x = self.extractor(x)['feature']  # * reference key should be specified to extract output from extractor.
        multi_output = self.fork_forward(x)
        return multi_output


def create_model(args, split_provider, gpu_ids=None):
    pass
    # 1ch, 3ch
    # multi
    # MLP+CNN
