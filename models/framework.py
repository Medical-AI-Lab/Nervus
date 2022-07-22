#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
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
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.vit_image_size = vit_image_size
        self.extractor = self.constuct_extractor(self.net_name, mlp_num_inputs=self.mlp_num_inputs, vit_image_size=self.vit_image_size)
        self.multi_classifier = self.construct_multi_classifier(self.net_name, self.num_classes_in_internal_label)

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiNetFusion(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, vit_image_size=None):
        assert (net_name != 'MLP'), f"net_name should not be MLP."

        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.vit_image_size = vit_image_size

        # Extractor of MLP and Net
        self.extractor_mlp = self.constuct_extractor('MLP', mlp_num_inputs=self.mlp_num_inputs)
        self.extractor_net = self.constuct_extractor(self.net_name, vit_image_size=self.vit_image_size)

        # * Needs to align shape when ConvNeXt
        #  >>> BaseNet.get_classifier('ConvNeXtTiny')
        # Sequential(
        # (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=768, out_features=1000, bias=True)
        # )
        self.addtional_module = nn.Identity()  # This is for aligning shape.
        if net_name.startswith('ConvNeXt'):
            base_classifier = self.get_classifier(net_name)
            layer_norm = base_classifier[0]
            flatten = base_classifier[1]
            self.addtional_module = nn.Sequential(
                                            layer_norm,
                                            flatten
                                        )

        # Intermidiate MLP
        self.in_featues_from_mlp = self.get_classifier_in_features('MLP')
        self.in_features_from_net = self.get_classifier_in_features(self.net_name)
        self.inter_mlp_in_feature = self.in_featues_from_mlp + self.in_features_from_net
        self.inter_mlp = self.MLPNet(self.inter_mlp_in_feature, inplace=False)  # ! If inplace==True, cannot backweard  Check!

        # Multi classifier
        self.multi_classifier = self.construct_multi_classifier('MLP', num_classes_in_internal_label)

    def forward(self, x_mlp, x_net):
        out_mlp = self.extractor_mlp(x_mlp)                  #            [5, 10] -> [5, 256]
        out_net = self.extractor_net(x_net)                  #   [5, 3, 256, 256] -> [5, 512]

        out_net = self.addtional_module(out_net)             # For alignning shape

        out_features = torch.cat([out_mlp, out_net], dim=1)  # [5, 256] + [5, 512] -> [5, 768]
        out_features = self.inter_mlp(out_features)          #            [5, 768] -> [5, 256]
        output = self.multi_forward(out_features)
        return output


def create_model(args, split_provider, gpu_ids=None):
    pass
    mlp = args
    # 1ch, 3ch
    # MLP, CNN, or MLP+CNN
    # MultiNet or MultiNetFusion