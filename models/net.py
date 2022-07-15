#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
from torchvision.ops import MLP
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor


# * Once num_classis is supposed to be 1000 as defined in models.resnet18
class MLPNet(nn.Module):
    def __init__(self, num_inputs, num_classes=1000):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_classes = num_classes

        # Configure of MLP
        self.hidden_channels = [256, 256, 256]
        self.dropout = 0.2

        self.mlp = MLP(in_channels=self.num_inputs, hidden_channels=self.hidden_channels, dropout=self.dropout)
        self.fc_in_features = self.hidden_channels[-1]
        self.fc = nn.Linear(self.fc_in_features, self.num_classes)

    def forward(self, x):
        x = self.mlp(x)
        output = self.fc(x)
        return output


cnns = {
    'ResNet18': models.resnet18,
    'ResNet': models.resnet50,
    'DenseNet': models.densenet161,
    'B0': models.efficientnet_b0,
    'B2': models.efficientnet_b2,
    'B4': models.efficientnet_b4,
    'B6': models.efficientnet_b6,
    'ConvNeXtTiny': models.convnext_tiny,
    'ConvNeXtSmall': models.convnext_small,
    'ConvNeXtBase': models.convnext_base,
    'ConvNeXtLarge': models.convnext_large
    }

vits = {
    'ViTb16': models.vit_b_16,
    'ViTb32': models.vit_b_32,
    'ViTl16': models.vit_l_16,
    'ViTl32': models.vit_l_32
    }

# * {'参照したい層': '参照したい名前'}
ref_feature = 'feature'
feature_extractors = {
    'ResNet18': {'avgpool': ref_feature},
    'ResNet': {'avgpool': ref_feature},
    'DenseNet': {'features': ref_feature},
    'B0': {'avgpool': ref_feature},
    'B2': {'avgpool': ref_feature},
    'B4': {'avgpool': ref_feature},
    'B6': {'avgpool': ref_feature},
    'ConvNeXtTiny': {'avgpool': ref_feature},
    'ConvNeXtSmall': {'avgpool': ref_feature},
    'ConvNeXtBase': {'avgpool': ref_feature},
    'ConvNeXtLarge': {'avgpool': ref_feature},
    'ViTb16': {'encoder.ln': ref_feature},
    'ViTb32': {'encoder.ln': ref_feature},
    'ViTl16': {'encoder.ln': ref_feature},
    'ViTl32': {'encoder.ln': ref_feature},
    'MLPNet': {'mlp': ref_feature}
    }

nets = {**cnns, **vits, **{'MLPNet': MLPNet}}


class BaseClassifier:
    def __init__(self):
        pass


class FeatureExtractorMixin:
    def set_net(self, net_name):
        assert (net_name in nets), f"No specified net: {net_name}."
        return nets[net_name]

    def create_extractor(self, net, net_name):
        # net = self.set_net(net_name)
        extractor = create_feature_extractor(net, feature_extractors[net_name])
        return extractor

    def get_feature(self, feature):
        return feature[ref_feature]
