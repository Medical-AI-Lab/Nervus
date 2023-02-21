#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import copy
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from .component import create_net
from .logger import BaseLogger
from lib import ParamSet
from typing import Dict, Tuple, Union

# Alias of typing
# eg. {'labels': {'label_A: torch.Tensor([0, 1, ...]), ...}}
LabelDict = Dict[str, Dict[str, Union[torch.IntTensor, torch.FloatTensor]]]


logger = BaseLogger.get_logger(__name__)


class BaseModel(ABC):
    """
    Class to construct model. This class is the base class to construct model.
    """
    def __init__(self, params: ParamSet) -> None:
        """
        Class to define Model

        Args:
            param (ParamSet): parameters
        """
        self.params = params
        self.label_list = self.params.label_list
        self.device = self.params.device
        self.gpu_ids = self.params.gpu_ids

        self.network = create_net(
                                mlp=self.params.mlp,
                                net=self.params.net,
                                num_outputs_for_label=self.params.num_outputs_for_label,
                                mlp_num_inputs=self.params.mlp_num_inputs,
                                in_channel=self.params.in_channel,
                                vit_image_size=self.params.vit_image_size,
                                pretrained=self.params.pretrained
                                )

    def train(self) -> None:
        """
        Make self.network training mode.
        """
        self.network.train()

    def eval(self) -> None:
        """
        Make self.network evaluation mode.
        """
        self.network.eval()

    @abstractmethod
    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor]]
                        ]:
        raise NotImplementedError

    def to_gpu(self, gpu_ids):
        """
        Make model compute on the GPU.
        """
        if gpu_ids != []:
            assert torch.cuda.is_available(), 'No available GPU on this machine.'
            self.network.to(self.device)
            self.network = nn.DataParallel(self.network, device_ids=gpu_ids)

    def init_network(self) -> None:
        """
        Initialize network.
        This method is used at test to reset the current weight by redefining network.
        """
        self.network = create_net(
                                mlp=self.params.mlp,
                                net=self.params.net,
                                num_outputs_for_label=self.params.num_outputs_for_label,
                                mlp_num_inputs=self.params.mlp_num_inputs,
                                in_channel=self.params.in_channel,
                                vit_image_size=self.params.vit_image_size,
                                pretrained=self.params.pretrained
                                )


class ModelMixin:
    """
    Class including methods for save or load weight, or learning_curve.
    """
    # variables to keep best_weight and best_epoch temporarily.
    acting_best_weight = None
    acting_best_epoch = None

    # For weight
    def store_weight(self) -> None:
        """
        Store weight.
        """
        self.acting_best_epoch = self.loss_store.epoch_loss['total'].get_best_epoch()
        _network = copy.deepcopy(self.network)
        if hasattr(_network, 'module'):
            # When DataParallel used, move weight to CPU.
            self.acting_best_weight = copy.deepcopy(_network.module.to(torch.device('cpu')).state_dict())
        else:
            self.acting_best_weight = copy.deepcopy(_network.state_dict())

    def save_weight(self, save_datetime_dir: str, as_best: bool) -> None:
        """
        Save weight.

        Args:
            save_datetime_dir (str): save_datetime_dir
            as_best (bool): True if weight is saved as best, otherwise False. Defaults to None.
        """
        assert isinstance(as_best, bool), 'Argument as_best should be bool.'
        save_dir = Path(save_datetime_dir, 'weights')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
        save_path = Path(save_dir, save_name)

        if as_best:
            save_name_as_best = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '_best' + '.pt'
            save_path_as_best = Path(save_dir, save_name_as_best)
            if save_path.exists():
                # Check if best weight already saved. If exists, rename with '_best'
                save_path.rename(save_path_as_best)
            else:
                torch.save(self.acting_best_weight, save_path_as_best)
        else:
            save_name = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
            torch.save(self.acting_best_weight, save_path)

    def load_weight(self, weight_path: Path) -> None:
        """
        Load wight from weight_path.

        Args:
            weight_path (Path): path to weight
        """
        logger.info(f"Load weight: {weight_path}.\n")
        weight = torch.load(weight_path)
        self.network.load_state_dict(weight)


class ModelWidget(BaseModel, ModelMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class MLPModel(ModelWidget):
    """
    Class for MLP model
    """

    def __init__(self, params: ParamSet) -> None:
        """
        Args:
            params: (ParamSet): parameters
        """
        super().__init__(params)

    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor]]
                        ]:
        """
        Unpack data for forwarding of MLP and pass them to device.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, Union[LabelDict, torch.IntTensor]]
                ]: inputs, labels, or inputs, labels, and periods
        """
        in_data = {'inputs': data['inputs'].to(self.device)}
        labels = {'labels': {label_name: label.to(self.device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        labels = {**labels, **{'periods': data['periods'].to(self.device)}}
        return in_data, labels

    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        inputs = in_data['inputs']
        output = self.network(inputs)
        return output


class CVModel(ModelWidget):
    """
    Class for CNN or ViT model
    """
    def __init__(self, params: ParamSet) -> None:
        """
        Args:
            params: (ParamSet): parameters
        """
        super().__init__(params)

    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor]]
                    ]:
        """
        Unpack data for forwarding of CNN or ViT Model and pass them to device.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, Union[LabelDict, torch.IntTensor]]
                ]: inputs, labels, or inputs, labels, and periods
        """
        in_data = {'image': data['image'].to(self.device)}
        labels = {'labels': {label_name: label.to(self.device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        labels = {**labels, **{'periods': data['periods'].to(self.device)}}
        return in_data, labels


    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        image = in_data['image']
        output = self.network(image)
        return output


class FusionModel(ModelWidget):
    """
    Class for MLP+CNN or MLP+ViT model.
    """
    def __init__(self, params: ParamSet) -> None:
        """
        Args:
            params: (ParamSet): parameters
        """
        super().__init__(params)

    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor]]
                    ]:
        """
        Unpack data for forwarding of MLP+CNN or MLP+ViT and pass them to device.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, LabelDict]
                ]: inputs, labels, or inputs, labels, and periods
        """
        in_data = {
                'inputs': data['inputs'].to(self.device),
                'image': data['image'].to(self.device)
                }
        labels = {'labels': {label_name: label.to(self.device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        labels = {**labels, **{'periods': data['periods'].to(self.device)}}
        return in_data, labels

    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        inputs = in_data['inputs']
        image = in_data['image']
        output = self.network(inputs, image)
        return output


def create_model(params: ParamSet) -> nn.Module:
    """
    Construct model.

    Args:
        params (ParamSet): parameters

    Returns:
        nn.Module: model
    """
    _isMLPModel = (params.mlp is not None) and (params.net is None)
    _isCVModel = (params.mlp is None) and (params.net is not None)
    _isFusion = (params.mlp is not None) and (params.net is not None)

    if _isMLPModel:
        return MLPModel(params)
    elif _isCVModel:
        return CVModel(params)
    elif _isFusion:
        return FusionModel(params)
    else:
        raise ValueError(f"Invalid model type: mlp={params.mlp}, net={params.net}.")
