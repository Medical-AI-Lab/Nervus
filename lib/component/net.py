#!/usr/bin/env python
# -*- coding: utf-8 -*-r

from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.ops import MLP
import torchvision.models as models
from typing import Dict, Optional, Union


class BaseNet:
    """
    Class to construct network
    """
    cnn = {
            'ResNet18': 'resnet18',
            'ResNet': 'resnet50',
            'DenseNet': 'densenet161',
            'EfficientNetB0': 'efficientnet_b0',
            'EfficientNetB2': 'efficientnet_b2',
            'EfficientNetB4': 'efficientnet_b4',
            'EfficientNetB6': 'efficientnet_b6',
            'EfficientNetV2s': 'efficientnet_v2_s',
            'EfficientNetV2m': 'efficientnet_v2_m',
            'EfficientNetV2l': 'efficientnet_v2_l',
            'ConvNeXtTiny': 'convnext_tiny',
            'ConvNeXtSmall': 'convnext_small',
            'ConvNeXtBase': 'convnext_base',
            'ConvNeXtLarge': 'convnext_large'
            }

    vit = {
            'ViTb16': 'vit_b_16',
            'ViTb32': 'vit_b_32',
            'ViTl16': 'vit_l_16',
            'ViTl32': 'vit_l_32',
            'ViTH14': 'vit_h_14'
            }

    net = {**cnn, **vit}

    _classifier = {
            'ResNet': 'fc',
            'DenseNet': 'classifier',
            'EfficientNet': 'classifier',
            'ConvNext': 'classifier',
            'ViT': 'heads'
            }

    classifier = {
                'ResNet18': _classifier['ResNet'],
                'ResNet': _classifier['ResNet'],
                'DenseNet': _classifier['DenseNet'],
                'EfficientNetB0': _classifier['EfficientNet'],
                'EfficientNetB2': _classifier['EfficientNet'],
                'EfficientNetB4': _classifier['EfficientNet'],
                'EfficientNetB6': _classifier['EfficientNet'],
                'EfficientNetV2s': _classifier['EfficientNet'],
                'EfficientNetV2m': _classifier['EfficientNet'],
                'EfficientNetV2l': _classifier['EfficientNet'],
                'ConvNeXtTiny': _classifier['ConvNext'],
                'ConvNeXtSmall': _classifier['ConvNext'],
                'ConvNeXtBase':  _classifier['ConvNext'],
                'ConvNeXtLarge':  _classifier['ConvNext'],
                'ViTb16': _classifier['ViT'],
                'ViTb32': _classifier['ViT'],
                'ViTl16': _classifier['ViT'],
                'ViTl32': _classifier['ViT'],
                'ViTH14': _classifier['ViT']
                }

    mlp_config = {
                'hidden_channels': [256, 256, 256],
                'dropout': 0.2
                }



    DUMMY = nn.Identity()


    @classmethod
    def set_mlp(cls, mlp_num_inputs: int = None, inplace: bool = None) -> MLP:
        """
        Construct MLP.

        Args:
            mlp_num_inputs (int): the number of input of MLP
            inplace (bool, optional): parameter for the activation layer, which can optionally do the operation in-place. Defaults to None.

        Returns:
            MLP: MLP
        """
        assert isinstance(mlp_num_inputs, int), f"Invalid number of inputs for MLP: {mlp_num_inputs}."
        return MLP(
                in_channels=mlp_num_inputs,
                hidden_channels=cls.mlp_config['hidden_channels'],
                inplace=inplace,
                dropout=cls.mlp_config['dropout']
                )

    @classmethod
    def set_net(
                cls,
                net_name: str = None,
                in_channel: int = None,
                vit_image_size: int = None,
                pretrained: bool = None
                ) -> nn.Module:
        """
        Modify network depending on in_channel and vit_image_size.

        Args:
            net_name (str): network name
            in_channel (int, optional): image channel(any of 1ch or 3ch). Defaults to None.
            vit_image_size (int, optional): image size which ViT handles if ViT is used. Defaults to None.
                                            vit_image_size should be power of patch size.
            pretrained (bool, optional): True when use pretrained CNN or ViT, otherwise False. Defaults to None.

        Returns:
            nn.Module: modified network
        """
        assert net_name in cls.net, f"No specified net: {net_name}."

        if net_name in cls.cnn:
            assert (vit_image_size == 0), \
                f"vit_image_size should be set 0 except using ViT, but got {vit_image_size}."
            _cnn = getattr(models, cls.cnn[net_name])
            if pretrained:
                net = _cnn(weights='DEFAULT')
            else:
                net = _cnn()

        elif net_name in cls.vit:
            assert (vit_image_size > 0), \
                f"vit_image_size must be positive integer, but got {vit_image_size}."
            _vit = getattr(models, cls.vit[net_name])
            if pretrained:
                net = cls.make_vit_with_aligned_weight(_vit, vit_name=net_name, vit_image_size=vit_image_size)
            else:
                net = _vit(image_size=vit_image_size)

        else:
            raise ValueError(f"No specified net: {net_name}.")

        # Align net depending on input channels.
        if in_channel == 1:
            net = cls.align_in_channels_1ch(net_name=net_name, net=net)

        return net

    @classmethod
    def make_vit_with_aligned_weight(
                cls,
                vit: nn.Module,
                vit_name: str = None,
                vit_image_size: int = None
                ) -> nn.Module:
        """
        Return pretrained ViT with aligned weight.

        Args:
            vit (nn.Module): ViT object
            vit_name (str): ViT name
            vit_image_size (int): image size which ViT handles

        Returns:
            nn.Module: pretrained ViT with aligned weight
        """
        pretrained_vit = vit(weights='DEFAULT')
        weight = pretrained_vit.state_dict()
        patch_size = int(vit_name[-2:])  # 'ViTb16' -> 16
        aligned_weight = models.vision_transformer.interpolate_embeddings(
                                                    image_size=vit_image_size,
                                                    patch_size=patch_size,
                                                    model_state=weight
                                                    )

        # Set ViT with vit_image_size and aligned_weight
        aligned_vit = vit(image_size=vit_image_size)
        aligned_vit.load_state_dict(aligned_weight)
        return aligned_vit

    @classmethod
    def align_in_channels_1ch(cls, net_name: str = None, net: nn.Module = None) -> nn.Module:
        """
        Align the first layer of network to handle grayscale, or 1ch image.

        Args:
            net_name (str): network name
            net (nn.Module): network itself

        Returns:
            nn.Module: network available for grayscale, or 1ch
        """
        if net_name.startswith('ResNet'):
            net.conv1.in_channels = 1
            net.conv1.weight = nn.Parameter(net.conv1.weight.sum(dim=1).unsqueeze(1))
            return net

        if net_name.startswith('DenseNet') or net_name.startswith('Efficient'):
            net.features.conv0.in_channels = 1
            net.features.conv0.weight = nn.Parameter(net.features.conv0.weight.sum(dim=1).unsqueeze(1))
            return net

        if net_name.startswith('ConvNeXt'):
            net.features[0][0].in_channels = 1
            net.features[0][0].weight = nn.Parameter(net.features[0][0].weight.sum(dim=1).unsqueeze(1))
            return net

        if net_name.startswith('ViT'):
            net.conv_proj.in_channels = 1
            net.conv_proj.weight = nn.Parameter(net.conv_proj.weight.sum(dim=1).unsqueeze(1))
            return net

        raise ValueError(f"No specified net: {net_name}.")

    @classmethod
    def construct_extractor(
                            cls,
                            net_name: str = None,
                            mlp_num_inputs: int = None,
                            in_channel: int = None,
                            vit_image_size: int = None,
                            pretrained: bool = None
                            ) -> nn.Module:
        """
        Construct extractor of network depending on net_name.

        Args:
            net_name (str): network name.
            mlp_num_inputs (int, optional): number of input of MLP. Defaults to None.
            in_channel (int, optional): image channel(any of 1ch or 3ch). Defaults to None.
            vit_image_size (int, optional): image size which ViT handles if ViT is used. Defaults to None.
            pretrained (bool, optional): True when use pretrained CNN or ViT, otherwise False. Defaults to None.

        Returns:
            nn.Module: extractor of network
        """
        if net_name == 'MLP':
            extractor = cls.set_mlp(mlp_num_inputs=mlp_num_inputs)
        else:
            extractor = cls.set_net(
                                    net_name=net_name,
                                    in_channel=in_channel,
                                    vit_image_size=vit_image_size,
                                    pretrained=pretrained
                                    )
            # Replace classifier with DUMMY(=nn.Identity()).
            setattr(extractor, cls.classifier[net_name], cls.DUMMY)
        return extractor

    @classmethod
    def get_classifier(cls, net_name: str) -> nn.Module:
        """
        Get classifier of network depending on net_name.

        Args:
            net_name (str): network name

        Returns:
            nn.Module: classifier of network
        """
        _net = getattr(models, cls.net[net_name])
        classifier = getattr(_net(), cls.classifier[net_name])
        return classifier

    @classmethod
    def construct_multi_classifier(
                                    cls,
                                    net_name: str = None,
                                    num_outputs_for_label: Dict[str, int] = None
                                    ) -> nn.ModuleDict:
        """
        Construct classifier for multi-label.

        Args:
            net_name (str): network name
            num_outputs_for_label (Dict[str, int]): number of outputs for each label

        Returns:
            nn.ModuleDict: classifier for multi-label
        """
        classifiers = dict()
        if net_name == 'MLP':
            in_features = cls.mlp_config['hidden_channels'][-1]
            for label_name, num_outputs in num_outputs_for_label.items():
                classifiers[label_name] = nn.Linear(in_features, num_outputs)

        elif net_name.startswith('ResNet') or net_name.startswith('DenseNet'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.in_features
            for label_name, num_outputs in num_outputs_for_label.items():
                classifiers[label_name] = nn.Linear(in_features, num_outputs)

        elif net_name.startswith('EfficientNet'):
            base_classifier = cls.get_classifier(net_name)
            dropout = base_classifier[0].p
            in_features = base_classifier[1].in_features
            for label_name, num_outputs in num_outputs_for_label.items():
                classifiers[label_name] = nn.Sequential(
                                                        nn.Dropout(p=dropout, inplace=False),
                                                        nn.Linear(in_features, num_outputs)
                                                    )

        elif net_name.startswith('ConvNeXt'):
            base_classifier = cls.get_classifier(net_name)
            layer_norm = base_classifier[0]
            flatten = base_classifier[1]
            in_features = base_classifier[2].in_features
            for label_name, num_outputs in num_outputs_for_label.items():
                # Shape is changed before nn.Linear.
                classifiers[label_name] = nn.Sequential(
                                                        layer_norm,
                                                        flatten,
                                                        nn.Linear(in_features, num_outputs)
                                                    )

        elif net_name.startswith('ViT'):
            base_classifier = cls.get_classifier(net_name)
            in_features = base_classifier.head.in_features
            for label_name, num_outputs in num_outputs_for_label.items():
                classifiers[label_name] = nn.Sequential(
                                                OrderedDict([
                                                        ('head', nn.Linear(in_features, num_outputs))
                                                        ])
                                                )

        else:
            raise ValueError(f"No specified net: {net_name}.")

        multi_classifier = nn.ModuleDict(classifiers)
        return multi_classifier

    @classmethod
    def get_classifier_in_features(cls, net_name: str) -> int:
        """
        Return in_feature of network indicating by net_name.
        This class is used in class MultiNetFusion() only.

        Args:
            net_name (str): net_name

        Returns:
            int : in_feature

        Required:
        classifier.in_feature
        classifier.[1].in_features
        classifier.[2].in_features
        classifier.head.in_features
        """
        if net_name == 'MLP':
            return cls.mlp_config['hidden_channels'][-1]

        base_classifier = cls.get_classifier(net_name)

        if net_name.startswith('ResNet') or net_name.startswith('DenseNet'):
            return base_classifier.in_features

        if net_name.startswith('EfficientNet'):
            return base_classifier[1].in_features

        if net_name.startswith('ConvNeXt'):
            return base_classifier[2].in_features

        if net_name.startswith('ViT'):
            return base_classifier.head.in_features

        raise ValueError(f"No specified net: {net_name}.")

    @classmethod
    def construct_aux_module(cls, net_name: str) -> nn.Sequential:
        """
        Construct module to align the shape of feature from extractor depending on network.
        Actually, only when net_name == 'ConvNeXt'.
        Because ConvNeXt has the process of aligning the dimensions in its classifier.

        Needs to align shape of the feature extractor only when ConvNeXt
        (classifier):
        Sequential(
            (0): LayerNorm2d((768,), eps=1e-06, element-wise_affine=True)
            (1): Flatten(start_dim=1, end_dim=-1)
            (2): Linear(in_features=768, out_features=1000, bias=True)
        )

        Args:
            net_name (str): net name

        Returns:
            nn.Module: layers such that they align the dimension of the output from the extractor like the original ConvNeXt.
        """
        if net_name.startswith('ConvNeXt'):
            base_classifier = cls.get_classifier(net_name)
            layer_norm = base_classifier[0]
            flatten = base_classifier[1]
            return nn.Sequential(
                                layer_norm,
                                flatten
                                )
        else:
            return cls.DUMMY


class MultiMixin:
    """
    Class to define auxiliary function to handle multi-label.
    """
    def multi_forward(self, out_features: int) -> Dict[str, float]:
        """
        Forward out_features to classifier for each label.

        Args:
            out_features (int): output from extractor

        Returns:
            Dict[str, float]: output of classifier of each label
        """
        output = dict()
        for label_name, classifier in self.multi_classifier.items():
            output[label_name] = classifier(out_features)
        return output


class MultiWidget(nn.Module, BaseNet, MultiMixin):
    """
    Class for a widget to inherit multiple classes simultaneously.
    """
    pass


class MultiNet(MultiWidget):
    """
    Model of MLP, CNN or ViT.
    """
    def __init__(
                self,
                net_name: str = None,
                num_outputs_for_label: Dict[str, int] = None,
                mlp_num_inputs: int = None,
                in_channel: int = None,
                vit_image_size: int = None,
                pretrained: bool = None
                ) -> None:
        """
        Args:
            net_name (str): MLP, CNN or ViT name
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            mlp_num_inputs (int): number of input of MLP.
            in_channel (int): number of image channel, ie gray scale(=1) or color image(=3).
            vit_image_size (int): image size to be input to ViT.
            pretrained (bool): True when use pretrained CNN or ViT, otherwise False.
        """
        super().__init__()

        self.net_name = net_name
        self.num_outputs_for_label = num_outputs_for_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channel = in_channel
        self.vit_image_size = vit_image_size
        self.pretrained = pretrained

        # self.extractor_net = MLP or CVmodel
        self.extractor_net = self.construct_extractor(
                                                    net_name=self.net_name,
                                                    mlp_num_inputs=self.mlp_num_inputs,
                                                    in_channel=self.in_channel,
                                                    vit_image_size=self.vit_image_size,
                                                    pretrained=self.pretrained
                                                    )
        # Multi classifier
        self.multi_classifier = self.construct_multi_classifier(net_name=self.net_name, num_outputs_for_label=self.num_outputs_for_label)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward.

        Args:
            x (torch.Tensor): tabular data or image

        Returns:
            Dict[str, torch.Tensor]: output
        """
        out_features = self.extractor_net(x)
        output = self.multi_forward(out_features)
        return output


class MultiNetFusion(MultiWidget):
    """
    Fusion model of MLP and CNN or ViT.
    """
    def __init__(
                self,
                net_name: str = None,
                num_outputs_for_label: Dict[str, int] = None,
                mlp_num_inputs: int = None,
                in_channel: int = None,
                vit_image_size: int = None,
                pretrained: bool = None
                ) -> None:
        """
        Args:
            net_name (str): CNN or ViT name. It is clear that MLP is used in fusion model.
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            mlp_num_inputs (int): number of input of MLP. Defaults to None.
            in_channel (int): number of image channel, ie gray scale(=1) or color image(=3).
            vit_image_size (int): image size to be input to ViT.
            pretrained (bool): True when use pretrained CNN or ViT, otherwise False.
        """
        assert (net_name != 'MLP'), 'net_name should not be MLP.'

        super().__init__()

        self.net_name = net_name
        self.num_outputs_for_label = num_outputs_for_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channel = in_channel
        self.vit_image_size = vit_image_size
        self.pretrained = pretrained

        # Extractor of MLP and Net
        self.extractor_mlp = self.construct_extractor(net_name='MLP', mlp_num_inputs=self.mlp_num_inputs)
        self.extractor_net = self.construct_extractor(
                                                    net_name=self.net_name,
                                                    in_channel=self.in_channel,
                                                    vit_image_size=self.vit_image_size,
                                                    pretrained=self.pretrained
                                                    )
        self.aux_module = self.construct_aux_module(self.net_name)

        # Intermediate MLP
        self.in_features_from_mlp = self.get_classifier_in_features('MLP')
        self.in_features_from_net = self.get_classifier_in_features(self.net_name)
        self.inter_mlp_in_feature = self.in_features_from_mlp + self.in_features_from_net
        self.inter_mlp = self.set_mlp(mlp_num_inputs=self.inter_mlp_in_feature, inplace=False)

        # Multi classifier
        self.multi_classifier = self.construct_multi_classifier(net_name='MLP', num_outputs_for_label=num_outputs_for_label)

    def forward(self, x_mlp: torch.Tensor, x_net: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward.

        Args:
            x_mlp (torch.Tensor): tabular data
            x_net (torch.Tensor): image

        Returns:
            Dict[str, torch.Tensor]: output
        """
        out_mlp = self.extractor_mlp(x_mlp)
        out_net = self.extractor_net(x_net)
        out_net = self.aux_module(out_net)

        out_features = torch.cat([out_mlp, out_net], dim=1)
        out_features = self.inter_mlp(out_features)
        output = self.multi_forward(out_features)
        return output


def get_last_extractor(net: Union[MultiNet, MultiNetFusion] = None, net_name: str = None) -> nn.Module:
    """
    Return the last extractor of network, or CNN or ViT.
    net should be one loaded weight.
    This is for Grad-CAM.

    Args:
        net (Union[MultiNet, MultiNetFusion]): network itself
        net_name (str): network name

    Returns:
        nn.Module: last extractor of network, or CNN or ViT
    """
    assert hasattr(net, 'extractor_net'), 'net should have extractor_net.'
    _extractor = net.extractor_net

    if net_name.startswith('ResNet'):
        return _extractor.layer4[-1]

    if net_name.startswith('DenseNet'):
        return _extractor.features.denseblock4.denselayer24

    if net_name.startswith('EfficientNet'):
        return _extractor.features[-1]

    if net_name.startswith('ConvNeXt'):
        return _extractor.features[-1][-1].block

    if net_name.startswith('ViT'):
        return _extractor.encoder.layers[-1]

    raise ValueError(f"Cannot extract last extractor of net: {net_name}.")


def create_net(
            mlp: Optional[str] = None,
            net: Optional[str] = None,
            num_outputs_for_label: Dict[str, int] = None,
            mlp_num_inputs: int = None,
            in_channel: int = None,
            vit_image_size: int = None,
            pretrained: bool = None
            ) -> Union[MultiNet, MultiNetFusion]:
    """
    Create network.

    Args:
        mlp (Optional[str]): 'MLP' or None
        net (Optional[str]):  CNN, ViT name or None
        num_outputs_for_label (Dict[str, int]): number of outputs for each label
        mlp_num_inputs (int): number of input of MLP.
        in_channel (int): number of image channel, ie gray scale(=1) or color image(=3).
        vit_image_size (int): image size to be input to ViT.
        pretrained (bool): True when use pretrained CNN or ViT, otherwise False.

    Returns:
        Union[MultiNet, MultiNetFusion]: network
    """
    _isMLPModel = (mlp is not None) and (net is None)
    _isCVModel = (mlp is None) and (net is not None)
    _isFusion = (mlp is not None) and (net is not None)

    if _isMLPModel:
        multi_net = MultiNet(
                            net_name='MLP',
                            num_outputs_for_label=num_outputs_for_label,
                            mlp_num_inputs=mlp_num_inputs,
                            in_channel=in_channel,
                            vit_image_size=vit_image_size,
                            pretrained=False  # No pretrained MLP
                            )

    elif _isCVModel:
        multi_net = MultiNet(
                            net_name=net,
                            num_outputs_for_label=num_outputs_for_label,
                            mlp_num_inputs=mlp_num_inputs,
                            in_channel=in_channel,
                            vit_image_size=vit_image_size,
                            pretrained=pretrained
                            )

    elif _isFusion:
        multi_net = MultiNetFusion(
                                net_name=net,
                                num_outputs_for_label=num_outputs_for_label,
                                mlp_num_inputs=mlp_num_inputs,
                                in_channel=in_channel,
                                vit_image_size=vit_image_size,
                                pretrained=pretrained
                                )
    else:
        raise ValueError(f"Invalid model type: mlp={mlp}, net={net}.")

    return multi_net
