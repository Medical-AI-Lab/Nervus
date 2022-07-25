#!/usr/bin/env python
# -*- coding: utf-8 -*-r

from collections import OrderedDict
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MLP

import sys
from pathlib import Path
sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('models.net')


class BaseNet:
    cnn = {
            'ResNet18': models.resnet18,
            'ResNet': models.resnet50,
            'DenseNet': models.densenet161,
            'EfficientNetB0': models.efficientnet_b0,
            'EfficientNetB2': models.efficientnet_b2,
            'EfficientNetB4': models.efficientnet_b4,
            'EfficientNetB6': models.efficientnet_b6,
            'EfficientNetV2s': models.efficientnet_v2_s,
            'EfficientNetV2m': models.efficientnet_v2_m,
            'EfficientNetV2l': models.efficientnet_v2_l,
            'ConvNeXtTiny': models.convnext_tiny,
            'ConvNeXtSmall': models.convnext_small,
            'ConvNeXtBase': models.convnext_base,
            'ConvNeXtLarge': models.convnext_large
            }

    vit = {
            'ViTb16': models.vit_b_16,
            'ViTb32': models.vit_b_32,
            'ViTl16': models.vit_l_16,
            'ViTl32': models.vit_l_32,
            'ViTH14': models.vit_h_14
            }

    vit_weight = {
                'ViTb16': models.ViT_B_16_Weights.DEFAULT, # DEFAULT =
                'ViTb32': models.ViT_B_32_Weights.DEFAULT, # DEFAULT =
                'ViTl16': models.ViT_L_16_Weights.DEFAULT, # DEFAULT =
                'ViTl32': models.ViT_L_32_Weights.DEFAULT, # DEFAULT =
                'ViTH14': models.ViT_H_14_Weights.DEFAULT  # DEFAULT = IMAGENET1K_SWAG_E2E_V1
                }

    net = {**cnn, **vit}

    classifier = {
                'ResNet18': 'fc',
                'ResNet': 'fc',
                'DenseNet': 'classifier',
                'EfficientNetB0': 'classifier',
                'EfficientNetB2': 'classifier',
                'EfficientNetB4': 'classifier',
                'EfficientNetB6': 'classifier',
                'EfficientNetV2s': 'classifier',
                'EfficientNetV2m': 'classifier',
                'EfficientNetV2l': 'classifier',
                'ConvNeXtTiny': 'classifier',
                'ConvNeXtSmall': 'classifier',
                'ConvNeXtBase': 'classifier',
                'ConvNeXtLarge': 'classifier',
                'ViTb16': 'heads',
                'ViTb32': 'heads',
                'ViTl16': 'heads',
                'ViTl32': 'heads',
                'ViTH14': 'heads'
                }

    in_layrer = {
            'ResNet': ['conv1'],
            'DenseNet': ['features', 'conv0'],       # ._module['features']._module['conv0']
            'EfficientNet': ['features', '0', '0'],  # ._module['features']._module['0']._module['0]
            'ConvNeXt': ['features', '0', '0'],
            'ViT': ['conv_proj']
    }

    mlp_config = {
                'hidden_channels': [256, 256, 256],
                'dropout': 0.2
                }

    DUMMY = nn.Identity()

    @classmethod
    def MLPNet(cls, mlp_num_inputs, inplace=None):
        assert isinstance(mlp_num_inputs, int), f"Invalid number of inputs for MLP: {mlp_num_inputs}."
        mlp = MLP(in_channels=mlp_num_inputs, hidden_channels=cls.mlp_config['hidden_channels'], inplace=inplace, dropout=cls.mlp_config['dropout'])
        return mlp

    @classmethod
    def align_in_channels_1ch(cls, net_name, net):
        # in_layer = ......
        # in_layer.in_features = 1
        # in_layer_in_channels = nn.Parameter(in_layer.weight.sum(dim=1).unsqueeze(1))
        if net_name.startswith('ResNet'):
            net.conv1.in_channels = 1
            net.conv1.weight = nn.Parameter(net.conv1.weight.sum(dim=1).unsqueeze(1))

        elif net_name.startswith('DenseNet'):
            net.features.conv0.in_channels = 1
            net.features.conv0.weight = nn.Parameter(net.features.conv0.weight.sum(dim=1).unsqueeze(1))

        elif net_name.startswith('Efficient'):
            net.features[0][0].in_channels = 1
            net.features[0][0].weight = nn.Parameter(net.features[0][0].weight.sum(dim=1).unsqueeze(1))

        elif net_name.startswith('ConvNeXt'):
            net.features[0][0].in_channels = 1
            net.features[0][0].weight = nn.Parameter(net.features[0][0].weight.sum(dim=1).unsqueeze(1))

        elif net_name.startswith('ViT'):
            net.conv_proj.in_channels = 1
            net.conv_proj.weight = nn.Parameter(net.conv_proj.weight.sum(dim=1).unsqueeze(1))

        else:
            logger.error(f"No specified net: {net_name}.")
        return net

    @classmethod
    def set_net(cls, net_name, in_channels=None, vit_image_size=None):
        assert net_name in cls.net, f"No specified net: {net_name}."
        assert (in_channels == 1) or (in_channels == 3), f"Invalid in_channels: {in_channels}."
        if net_name in cls.cnn:
            net = cls.cnn[net_name]()
        else:
            net = cls.set_vit(net_name, vit_image_size)

        if in_channels == 1:
            net = cls.align_in_channels_1ch(net_name, net)
        return net

    @classmethod
    def set_vit(cls, net_name, vit_image_size=None):
        assert isinstance(vit_image_size, int), f"Invalid image size for ViT: {vit_image_size}."
        base_vit = cls.vit[net_name]
        pretrained_vit = base_vit(weights=cls.vit_weight[net_name])

        # Align weight depending on image size
        weight = pretrained_vit.state_dict()
        patch_size = int(net_name[-2:])  # 'ViTb16' -> 16
        aligned_weight = models.vision_transformer.interpolate_embeddings(
                                                    image_size=vit_image_size,
                                                    patch_size=patch_size,
                                                    model_state=weight
                                                    )
        aligned_vit = base_vit(image_size=vit_image_size)  # Specify new image size.
        aligned_vit.load_state_dict(aligned_weight)        # Load weight which can handle nee image size.
        return aligned_vit

    @classmethod
    def constuct_extractor(cls, net_name, mlp_num_inputs=None, in_channels=None, vit_image_size=None):
        if net_name == 'MLP':
            extractor = cls.MLPNet(mlp_num_inputs)
        else:
            extractor = cls.set_net(net_name, in_channels, vit_image_size)
            extractor._modules[cls.classifier[net_name]] = cls.DUMMY
        return extractor

    @classmethod
    def get_classifier(cls, net_name):
        net = cls.net[net_name]()
        classifier = net._modules[cls.classifier[net_name]]
        return classifier

    @classmethod
    def construct_multi_classifier(cls, net_name, num_classes_in_internal_label):
        classifiers = dict()

        if net_name == 'MLP':
            in_features = cls.mlp_config['hidden_channels'][-1]
            for internal_label_name, num_classes in num_classes_in_internal_label.items():
                classifiers[internal_label_name] = nn.Linear(in_features, num_classes)

        elif net_name.startswith('ResNet') or net_name.startswith('DenseNet'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.in_features
            for internal_label_name, num_classes in num_classes_in_internal_label.items():
                classifiers[internal_label_name] = nn.Linear(in_features, num_classes)

        elif net_name.startswith('EfficientNet'):
            base_classifier = cls.get_classifier(net_name)
            dropout = base_classifier[0].p
            in_features = base_classifier[1].in_features
            for internal_label_name, num_classes in num_classes_in_internal_label.items():
                classifiers[internal_label_name] = nn.Sequential(
                                                        nn.Dropout(p=dropout, inplace=False),  # if inplace==True, cannot backward.
                                                        nn.Linear(in_features, num_classes)
                                                    )

        elif net_name.startswith('ConvNeXt'):
            base_classifier = cls.get_classifier(net_name)
            layer_norm = base_classifier[0]
            flatten = base_classifier[1]
            in_features = base_classifier[2].in_features
            for internal_label_name, num_classes in num_classes_in_internal_label.items():
                # * Shape is changed before nn.Linear.
                classifiers[internal_label_name] = nn.Sequential(
                                                        layer_norm,
                                                        flatten,
                                                        nn.Linear(in_features, num_classes)
                                                    )
        elif net_name.startswith('ViT'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.head.in_features
            for internal_label_name, num_classes in num_classes_in_internal_label.items():
                classifiers[internal_label_name] = nn.Sequential(OrderedDict([
                                                        ('head', nn.Linear(in_features, num_classes))
                                                    ]))

        else:
            logger.error(f"No specified net: {net_name}.")

        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    @classmethod
    def get_classifier_in_features(cls, net_name):
        """ This is used in class MultiNetFusion(MultiWidget() only.

        Args:
            net_name (_type_): _description_

        Returns:
            _type_: _description_
        """
        if net_name == 'MLP':
            in_features = cls.mlp_config['hidden_channels'][-1]

        elif net_name.startswith('ResNet') or net_name.startswith('DenseNet'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.in_features

        elif net_name.startswith('EfficientNet'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier[1].in_features

        elif net_name.startswith('ConvNeXt'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier[2].in_features

        elif net_name.startswith('ViT'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.head.in_features

        else:
            logger.error(f"No specified net: {net_name}.")
        return in_features

    @classmethod
    def construct_aux_module(cls, net_name):
        """ Construct module to align the shape of feature from extractor depending on net.

        Args:
            net_name (str): net name

        Returns:
            nn.Module: layers
        """
        # * Needs to align shape of the feature extractor when ConvNeXt
        #  >>> BaseNet.get_classifier('ConvNeXtTiny')
        # Sequential(
        # (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=768, out_features=1000, bias=True)
        # )
        aux_module = cls.DUMMY
        if net_name.startswith('ConvNeXt'):
            base_classifier = cls.get_classifier(net_name)
            layer_norm = base_classifier[0]
            flatten = base_classifier[1]
            aux_module = nn.Sequential(
                                layer_norm,
                                flatten
                                )
        return aux_module
