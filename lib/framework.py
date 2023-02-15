#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import copy
from abc import ABC, abstractmethod
import pandas as pd
import torch
import torch.nn as nn
from .component import create_net
from .logger import BaseLogger
from lib import ParamSet
from typing import Dict, Tuple, Union


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

        if self.params.isTrain:
            from .component import set_criterion, set_optimizer, create_loss_store
            self.criterion = set_criterion(self.params.criterion, self.params.device)
            self.optimizer = set_optimizer(self.params.optimizer, self.network, self.params.lr)
            self.loss_store = create_loss_store(self.params.task, self.criterion, self.params.label_list, self.params.device)
        else:
            pass

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

    def _enable_on_gpu_if_available(self) -> None:
        """
        Make model compute on the GPU.
        """
        if self.gpu_ids != []:
            assert torch.cuda.is_available(), 'No avalibale GPU on this machine.'
            self.network.to(self.device)
            self.network = nn.DataParallel(self.network, device_ids=self.gpu_ids)
        else:
            pass

    @abstractmethod
    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                        Dict[str, torch.Tensor],
                        Dict[str, Union[int, float]]
                        ]:
        pass

    def multi_label_to_device(self, multi_label: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """
        Pass the value of each label to the device

        Args:
            multi_label (Dict[str, Union[int, float]]): dictionary of each label and its value

        Returns:
            Dict[str, Union[int, float]]: dictionary of each label and its value which is on devide
        """
        assert any(multi_label), 'multi-label is empty.'
        _multi_label = dict()
        for label_name, each_data in multi_label.items():
            _multi_label[label_name] = each_data.to(self.device)
        return _multi_label

    def backward(self) -> None:
        """
        Backward
        """
        self.loss = self.loss_store.batch_loss['total']
        self.loss.backward()

    def optimize_parameters(self) -> None:
        """
        Update parameters
        """
        self.optimizer.step()

    # Loss
    @abstractmethod
    def cal_batch_loss(self) -> None:
        pass

    def cal_running_loss(self, batch_size: int) -> None:
        """
        Calculate loss for each iteration.

        Args:
            batch_size (int): batch size. Defaults to None.
        """
        self.loss_store.cal_running_loss(batch_size)

    def cal_epoch_loss(self, epoch: int, phase: str, dataset_size: int) -> None:
        """
        Calculate loss for each epoch.

        Args:
            epoch (int): epoch number
            phase (str): phase, ie. 'train' or 'val'
            dataset_size (int): dataset size. Defaults to None.
        """
        self.loss_store.cal_epoch_loss(epoch, phase, dataset_size)

    def is_total_val_loss_updated(self) -> bool:
        """
        Check if val loss updated or not.

        Returns:
            bool: True if val loss updated, otherwise False.
        """
        _total_epoch_loss = self.loss_store.epoch_loss['total']
        is_updated = _total_epoch_loss.is_val_loss_updated()
        return is_updated

    def print_epoch_loss(self, num_epochs: int, epoch: int) -> None:
        """
        Print loss for each epoch.

        Args:
            num_epochs (int): total numger of epochs
            epoch (int): current epoch number
        """
        self.loss_store.print_epoch_loss(num_epochs, epoch)

    def init_network(self) -> None:
        """
        Initialize network.
        This method is used at test to reset the current weight by redefining netwrok.
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
            as_best (bool): True if weight is saved as best, otherise False. Defaults to None.
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

        # Make model compute on GPU after loading weight.
        self._enable_on_gpu_if_available()

    # For learning curve
    def save_learning_curve(self, save_datetime_dir: str) -> None:
        """
        Save leraning curve.

        Args:
            save_datetime_dir (str): save_datetime_dir
        """
        save_dir = Path(save_datetime_dir, 'learning_curve')
        save_dir.mkdir(parents=True, exist_ok=True)
        epoch_loss = self.loss_store.epoch_loss
        for label_name in self.label_list + ['total']:
            each_epoch_loss = epoch_loss[label_name]
            df_each_epoch_loss = pd.DataFrame({
                                                'train_loss': each_epoch_loss.train,
                                                'val_loss': each_epoch_loss.val
                                            })
            best_epoch = str(each_epoch_loss.get_best_epoch()).zfill(3)
            best_val_loss = f"{each_epoch_loss.get_best_val_loss():.4f}"
            save_name = 'learning_curve_' + label_name + '_val-best-epoch-' + best_epoch + '_val-best-loss-' + best_val_loss + '.csv'
            save_path = Path(save_dir, save_name)
            df_each_epoch_loss.to_csv(save_path, index=False)


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
                        Dict[str, torch.Tensor],
                        Dict[str, Union[int, float]]
                        ]:
        """
        Unpack data for forwarding of MLP.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]
                ]: inputs and labels
        """
        in_data = {'inputs': data['inputs']}
        labels = {'labels': data['labels']}
        return in_data, labels

    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        inputs = in_data['inputs'].to(self.device)
        output = self.network(inputs)
        return output

    def cal_batch_loss(
                    self,
                    output: Dict[str, torch.Tensor],
                    labels: Dict[str, Union[int, float]]
                    ) -> None:
        """
        Calculate loss for each bach.

        Args:
            output (Dict[str, torch.Tensor]): output
            labels (Dict[str, Union[int, float]]): labels
        """
        _labels = self.multi_label_to_device(labels['labels'])
        self.loss_store.cal_batch_loss(output, _labels)


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
                        Dict[str, torch.Tensor],
                        Dict[str, Union[int, float]]
                        ]:
        """
        Unpack data for forwarding of CNN or ViT Model.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]
                ]: image and labels
        """
        in_data = {'image': data['image']}
        labels = {'labels': data['labels']}
        return in_data, labels

    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        image = in_data['image'].to(self.device)
        output = self.network(image)
        return output

    def cal_batch_loss(
                    self,
                    output: Dict[str, torch.Tensor],
                    labels: Dict[str, Union[int, float]]
                    ) -> None:
        """
        Calculate loss for each bach.

        Args:
            output (Dict[str, torch.Tensor]): output
            labels (Dict[str, Union[int, float]]): labels
        """
        _labels = self.multi_label_to_device(labels['labels'])
        self.loss_store.cal_batch_loss(output, _labels)


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
                        Dict[str, torch.Tensor],
                        Dict[str, Union[int, float]]
                        ]:
        """
        Unpack data for forwarding of MLP+CNN or MLP+ViT.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]
                ]: inputs, image and labels
        """
        in_data = {
                    'inputs': data['inputs'],
                    'image': data['image']
                }
        labels = {'labels': data['labels']}
        return in_data, labels

    def __call__(self, in_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward

        Args:
            in_data (Dict[str, torch.Tensor]): data to be input into model

        Returns:
            Dict[str, torch.Tensor]: output
        """
        inputs = in_data['inputs'].to(self.device)
        image = in_data['image'].to(self.device)
        output = self.network(inputs, image)
        return output

    def cal_batch_loss(
                    self,
                    output: Dict[str, torch.Tensor],
                    labels: Dict[str, Union[int, float]]
                    ) -> None:
        """
        Calculate loss for each bach.

        Args:
            output (Dict[str, torch.Tensor]): output
            labels (Dict): labels
        """
        _labels = self.multi_label_to_device(labels['labels'])
        self.loss_store.cal_batch_loss(output, _labels)


class MLPDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with MLP
    """
    def __init__(self, params: ParamSet) -> None:
        """
        Args:
            params (ParamSet): parameters
        """
        super().__init__(params)

    def set_data(
                self,
                data: Dict
                ) -> Tuple[
                            Dict[str, torch.Tensor],
                            Dict[str, Union[int, float]]
                        ]:
        """
        Unpack data for forwarding of DeepSurv model with MLP

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]]
                ]: inputs, and labels, periods
        """
        in_data = {'inputs': data['inputs']}
        labels = {
                    'labels': data['labels'],
                    'periods': data['periods']
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
        inputs = in_data['inputs'].to(self.device)
        output = self.network(inputs)
        return output

    def cal_batch_loss(
                    self,
                    output: Dict[str, torch.Tensor],
                    labels: Dict[str, Union[int, float]]
                    ) -> None:
        """
        Calculate loss for each bach.

        Args:
            outputs (Dict[str, torch.Tensor]): output
            labels (Dict[str, Union[int, float]]): labels and periods
        """
        _labels = self.multi_label_to_device(labels['labels'])
        _periods = labels['periods'].float().to(self.device)
        self.loss_store.cal_batch_loss(output, _labels, _periods, self.network)


class CVDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with CNN or ViT
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
                            Dict[str, torch.Tensor],
                            Dict[str, Union[int, float]]
                        ]:
        """
        Unpack data for forwarding of DeepSurv model with with CNN or ViT

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]]
                ]: image, and labels, periods
        """
        in_data = {'image': data['image']}
        labels = {
                'labels': data['labels'],
                'periods': data['periods']
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
        image = in_data['image'].to(self.device)
        output = self.network(image)
        return output

    def cal_batch_loss(
                    self,
                    output: Dict[str, torch.Tensor],
                    labels: Dict[str, Union[int, float]]
                    ) -> None:
        """
        Calculate loss for each bach.

        Args:
            output (Dict[str, torch.Tensor]): output
            lables and periods (Dict[str, Union[int, float]]): labels and periods
        """
        _labels = self.multi_label_to_device(labels['labels'])
        _periods = labels['periods'].float().to(self.device)
        self.loss_store.cal_batch_loss(output, _labels, _periods, self.network)


class FusionDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with MLP+CNN or MLP+ViT model.
    """
    def __init__(self, params: ParamSet) -> None:
        """
        Args:
            params: (ParamSet): parameters
        """
        super().__init__(params)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of DeepSurv with MLP+CNN or MLP+ViT.

        Args:
            data (Dict): dictionary of data

        Returns:
            Tuple[
                Dict[str, torch.Tensor],
                Dict[str, Union[int, float]]
                ]: inputs, image, and labels, periods
        """
        in_data = {
                'inputs': data['inputs'],
                'image': data['image']
                }
        labels = {
                'labels': data['labels'],
                'periods': data['periods']
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
        inputs = in_data['inputs'].to(self.device)
        image = in_data['image'].to(self.device)
        output = self.network(inputs, image)
        return output

    def cal_batch_loss(
                        self,
                        output: Dict[str, torch.Tensor],
                        labels: Dict[str, Union[int, float]]
                        ) -> None:
        """
        Calculate loss for each bach.

        Args:
            output (Dict[str, torch.Tensor]): output
            labels (Dict[str, Union[int, float]]): labels and periods
        """
        _labels = self.multi_label_to_device(labels['labels'])
        _periods = labels['periods'].float().to(self.device)
        self.loss_store.cal_batch_loss(output, _labels, _periods, self.network)


def create_model(params: ParamSet) -> nn.Module:
    """
    Construct model.

    Args:
        params (ParamSet): parameters

    Returns:
        nn.Module: model
    """
    task = params.task
    _isMLPModel = (params.mlp is not None) and (params.net is None)
    _isCVModel = (params.mlp is None) and (params.net is not None)
    _isFusion = (params.mlp is not None) and (params.net is not None)

    if (task == 'classification') or (task == 'regression'):
        if _isMLPModel:
            model = MLPModel(params)
        elif _isCVModel:
            model = CVModel(params)
        elif _isFusion:
            model = FusionModel(params)
        else:
            raise ValueError(f"Invalid model type: mlp={params.mlp}, net={params.net}.")

    elif task == 'deepsurv':
        if _isMLPModel:
            model = MLPDeepSurv(params)
        elif _isCVModel:
            model = CVDeepSurv(params)
        elif _isFusion:
            model = FusionDeepSurv(params)
        else:
            raise ValueError(f"Invalid model type: mlp={params.mlp}, net={params.net}.")

    else:
        raise ValueError(f"Invalid task: {task}.")

    if params.isTrain:
        model._enable_on_gpu_if_available()
    # When test, execute model._enable_on_gpu_if_available() in load_weight(),
    # ie. after loading weight.
    return model
