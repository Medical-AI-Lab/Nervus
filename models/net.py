#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torchvision.models as models
from torchvision.ops import MLP
from torchvision.models.feature_extraction import create_feature_extractor


class BaseNet:
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

    nets = {**cnns, **vits}


class FeatureExtractor(BaseNet):
    # * {'target layer': 'reference name'}
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
                    'ViTl32': {'encoder.ln': ref_feature}
                    }

    def __init__(self):
        super().__init__()

    @classmethod
    def set_net(cls, net_name):
        assert (net_name in cls.nets), f"No specified net: {net_name}."
        net = cls.nets[net_name]()  # * Since cls.nets[net_name] is just an object, it is required to inistantiate with ().
        return net

    @classmethod
    def create_extractor(cls, net_name):
        net = cls.set_net(net_name)
        extractor = create_feature_extractor(net, cls.feature_extractors[net_name])
        return extractor

    @classmethod
    def get_feature(cls, feature_dict):
        return feature_dict[cls.ref_feature].squeeze() # Without squeeze(), shape does not match between feature and classfifer


def MLPNet(num_inputs):
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
