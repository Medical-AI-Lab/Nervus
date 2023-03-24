#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import copy
from pathlib import Path
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.distributed as dist
from .component import create_net
from .logger import BaseLogger
from lib import ParamSet
from typing import List, Dict, Tuple, Union


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

        self.network = create_net(
                                mlp=self.params.mlp,
                                net=self.params.net,
                                num_outputs_for_label=self.params.num_outputs_for_label,
                                mlp_num_inputs=self.params.mlp_num_inputs,
                                in_channel=self.params.in_channel,
                                vit_image_size=self.params.vit_image_size,
                                pretrained=self.params.pretrained
                                )

        # variables to keep temporary best_weight and best_epoch
        self.acting_best_weight = None
        self.acting_best_epoch = None

    def train(self) -> None:
        """
        Make network training mode.
        """
        self.network.train()

    def eval(self) -> None:
        """
        Make network evaluation mode.
        """
        self.network.eval()

    @abstractmethod
    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                        ]:
        raise NotImplementedError

    def store_weight(self, at_epoch: int = None) -> None:
        """
        Store weight and epoch number when it is saved.

        Args:
            at_epoch (int): epoch number when save weight
        """
        self.acting_best_epoch = at_epoch

        _network = copy.deepcopy(self.network)
        if hasattr(_network, 'module'):
            # When using DDP, pass weight to CPU.
            self.acting_best_weight = copy.deepcopy(_network.module.to(torch.device('cpu')).state_dict())
        else:
            self.acting_best_weight = copy.deepcopy(_network.state_dict())

    def save_weight(self, save_datetime_dir: str, as_best: bool = None) -> None:
        """
        Save weight.

        Args:
            save_datetime_dir (str): save_datetime_dir
            as_best (bool): True if weight is saved as best, otherwise False. Defaults to None.
        """

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

    def load_weight(self, weight_path: Path, on_device: torch.device = None) -> None:
        """
        Load wight from weight_path.

        Args:
            weight_path (Path): path to weight
            on_device (torch.device) : the location where all tensors should be loaded.
        """
        logger.info(f"Load weight: {weight_path}.\n")
        weight = torch.load(weight_path, map_location=on_device)
        self.network.load_state_dict(weight)

    def init_network(self) -> None:
        """
        Initialize network.
        This method is used at test in order to reset the current weight by redefining network.
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


class MLPModel(BaseModel):
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
                data: Dict,
                device: torch.device
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                        ]:
        """
        Unpack data for forwarding of MLP and calculating loss
        by passing them to device.
        When deepsurv, period and network are also returned.

        Args:
            data (Dict): dictionary of data
            device (torch.device): device

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                ]: input of model and data for calculating loss.
        eg.
        ([inputs], [labels]), or ([inputs], [labels, periods, network]) when deepsurv
        """
        in_data = {'inputs': data['inputs'].to(device)}
        labels = {'labels': {label_name: label.to(device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        # When deepsurv
        labels = {
                  **labels,
                  **{'periods': data['periods'].to(device), 'network': self.network.to(device)}
                }
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


class CVModel(BaseModel):
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
                data: Dict,
                device: torch.device
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                    ]:
        """
        Unpack data for forwarding of CNN or ViT and calculating loss by passing them to device.
        When deepsurv, period and network are also returned.

        Args:
            data (Dict): dictionary of data
            device (torch.device): device

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                ]: input of model and data for calculating loss.
        eg.
        ([image], [labels]), or ([image], [labels, periods, network]) when deepsurv
        """
        in_data = {'image': data['image'].to(device)}
        labels = {'labels': {label_name: label.to(device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        # When deepsurv
        labels = {
                  **labels,
                  **{'periods': data['periods'].to(device), 'network': self.network.to(device)}
                }
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


class FusionModel(BaseModel):
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
                data: Dict,
                device: torch.device
                ) -> Tuple[
                        Dict[str, torch.FloatTensor],
                        Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                    ]:
        """
        Unpack data for forwarding of MLP+CNN or MLP+ViT and calculating loss
        by passing them to device.
        When deepsurv, period and network are also returned.

        Args:
            data (Dict): dictionary of data
            device (torch.device): device

        Returns:
            Tuple[
                Dict[str, torch.FloatTensor],
                Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                ]: input of model and data for calculating loss.
        eg.
        ([inputs, image], [labels]), or ([inputs, image], [labels, periods, network]) when deepsurv
        """
        in_data = {
                'inputs': data['inputs'].to(device),
                'image': data['image'].to(device)
                }
        labels = {'labels': {label_name: label.to(device) for label_name, label in data['labels'].items()}}

        if not any(data['periods']):
            return in_data, labels

        # When deepsurv
        labels = {
                  **labels,
                  **{'periods': data['periods'].to(device), 'network': self.network.to(device)}
                }
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
    _isFusionModel = (params.mlp is not None) and (params.net is not None)

    if _isMLPModel:
        return MLPModel(params)
    elif _isCVModel:
        return CVModel(params)
    elif _isFusionModel:
        return FusionModel(params)
    else:
        raise ValueError(f"Invalid model type: mlp={params.mlp}, net={params.net}.")


#
# The below is for distributed learning.
#
def is_master(rank: int) -> bool:
    """
    Return whether master or not.

    Args:
        rank (int): rank, or process id

    Returns:
        bool: whether master or not
    """
    MASTER = 0
    return (rank == MASTER)


def set_device(rank: int = None, gpu_ids: List[int] = None) -> torch.device:
    """
    Define device depending on gou_ids and rank.

    Args:
        rank (int): rank, or process id
        gpu_ids (List[int]): GPU ids

    Returns:
        torch.device :device

    Note:
        When using GPU, device is defined by rank-th on gpu_ids.
        eg.
        gpu_ids = [1, 2, 0],
        rank=0 -> gpu_id=gpu_ids[rank]=1
    """
    if gpu_ids == []:
        return torch.device('cpu')
    else:
        assert torch.cuda.is_available(), 'No available GPU on this machine.'
        return torch.device(f"cuda:{gpu_ids[rank]}")


def setup(rank: int = None, world_size: int = None, on_cpu: bool = None) -> None:
    """
    Initialize the process group.

    Args:
        rank (int): rank, or process id
        world_size (int): the total number of process
        on_cpu (bool]): Whether .................
    """
    if on_cpu:
        backend = 'gloo'  # For CPU
    else:
        backend = 'nccl'  # For GPU
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def set_world_size(gpu_ids: List[int], on_cpu: bool = None) -> int:
    """
    Set world_size, ie, total number of processes.

    Args:
        gpu_ids (List[int]): GPU ids
        on_cpu (bool): True when distributed learning 1CPU/N-Process (N>1) on CPU.

    Returns:
        int: world_size

    Note:
        1CPU/1Process or 1GPU/1Process.
    """
    NUM_PROCESS_ON_CPU = 4  # Appropriate value based on CPU performance is desirable.
    if gpu_ids == []:
        if on_cpu:
            # 1CPU/N-Process (N>1)
            return NUM_PROCESS_ON_CPU
        else:
            # 1CPU/1Process
            return 1
    else:
        # When using GPU, 1GPU/1Process
        return len(gpu_ids)


def setenv(is_seed_fixed: bool = False) -> None:
    """
    Set environment variables.

    Args:
        is_seed_fixed (bool): Whether seed is fixed or not

    """
    os.environ['GLOO_SOCKET_IFNAME'] = 'en0'  # 'eth0'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    if is_seed_fixed:
        # For reproducible learning
        SEED = 0
        np.random.seed(SEED)
        random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
