#!/usr/bin/env python
# -*- coding: utf-8 -*-r

import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MLP


class BaseNet:
    nets = {
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
            'ConvNeXtLarge': models.convnext_large,
            'ViTb16': models.vit_b_16,
            'ViTb32': models.vit_b_32,
            'ViTl16': models.vit_l_16,
            'ViTl32': models.vit_l_32
            }

    DUMMY = nn.Identity()

    @classmethod
    def set_net(cls, net_name):
        assert (net_name in cls.nets), f"No specified net: {net_name}."
        net = cls.nets[net_name]()
        return net

    @classmethod
    def MLPNet(cls, num_inputs):
        """
        Args:
            num_inputs (int): the number of inputs

        Returns:
            torchvision.ops.misc.MLP : MLP has no output layer.
        """
        hidden_channels = [256, 256, 256]
        dropout = 0.2
        mlp = MLP(in_channels=num_inputs, hidden_channels=hidden_channels, dropout=dropout)
        return mlp

    @classmethod
    def create_feature_extractor(cls, net_name):
        extractor = cls.nets[net_name]()
        if net_name.startswith('ResNet'):
            extractor.fc = cls.DUMMY
        elif net_name.startswith('DenseNet'):
            extractor.classifier = cls.DUMMY
        elif net_name.startswith('B'):
            extractor.classifier = cls.DUMMY
        elif net_name.startswith('Conv'):
            extractor.classifier = cls.DUMMY
        else:
            # ViT
            extractor.heads = cls.DUMMY
        return extractor

    @classmethod
    def get_classifier(cls, net_name):
        net = cls.nets[net_name]()
        if net_name.startswith('ResNet'):
            classifier = net.fc
        elif net_name.startswith('DenseNet'):
            classifier = net.classifier
        elif net_name.startswith('B'):
            classifier = net.classifier
        elif net_name.startswith('Conv'):
            classifier = net.classifier
        else:
            # ViT
            classifier = net.heads
        return classifier

    @classmethod
    def get_in_features(cls, net_name):
        net = cls.nets[net_name]()
        if net_name.startswith('ResNet'):
            in_features = net.fc.in_features
        elif net_name.startswith('DenseNet'):
            in_features = net.classifier.in_features
        elif net_name.startswith('B'):
            in_features = net.classifier[1].in_features
        elif net_name.startswith('Conv'):
            in_features = net.classifier[2].in_features
        else:
            # ViT
            in_features = net.heads.head.in_features
        return in_features
