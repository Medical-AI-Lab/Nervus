#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.models as models
from torchvision.ops import MLP


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

mlp = {
    'MLP': MLP
    }

nets = {**cnns, **vits, **mlp}


def set_net(net_name):
    assert (net_name in nets), f"No specified net: {net_name}."

    return nets[net_name]
