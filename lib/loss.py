#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import torch
from lib.logger import Logger as logger
from typing import List, Dict, Union
from torch import Tensor
import torch.nn as nn


@dataclass
class EpochLoss:
    """
    Class to store epoch loss of each internal label.
    """
    train: List[float] = field(default_factory=list)
    val: List[float] = field(default_factory=list)
    best_val_loss: float = None
    best_epoch: int = None
    update_flag: bool = False

    def append_epoch_loss(self, phase: str, new_epoch_loss: float) -> None:
        """
        Append loss to list depending on phase.

        Args:
            phase (str): train or val
            new_epoch_loss (float): loss value
        """
        getattr(self, phase).append(new_epoch_loss)

    def get_latest_loss(self, phase: str) -> float:
        """
        Retern the latest loss of phase.
        Args:
            phase (str): train or val
        Returns:
            float: the latest loss
        """
        latest_loss = getattr(self, phase)[-1]
        return latest_loss

    def get_best_val_loss(self) -> float:
        """
        Return the best val loss.

        Returns:
            float: the base val loss
        """
        return self.best_val_loss

    def set_best_val_loss(self, best_val_loss: float) -> None:
        """
        Set a val loss to keep it as the best loss.

        Args:
            best_val_loss (float): the best val loss
        """
        self.best_val_loss = best_val_loss

    def get_best_epoch(self) -> float:
        """
        Return the epoch at which val loss is the best.

        Returns:
            float: epoch
        """
        return self.best_epoch

    def set_best_epoch(self, best_epoch: int) -> None:
        """
        Set best_epoch to keep it as the best epoch.

        Args:
            best_epoch (int): the bset epoch
        """
        self.best_epoch = best_epoch

    def up_update_flag(self) -> None:
        """
        Set flag True to indicate that the best loss is updated.
        """
        self.update_flag = True

    def down_update_flag(self) -> None:
        """
        Set flag False to indicate that the best loss is not updated.
        """
        self.update_flag = False

    def is_val_loss_updated(self) -> bool:
        """
        Check if if val loss is updated.

        Returns:
            bool: True if val loss is updated.
        """
        return self.update_flag

    def check_best_val_loss_epoch(self, epoch: int) -> None:
        """
        Check if val loss is the bset at epoch.

        Args:
            epoch (int): epoch at which loss is checked if it is the best.
        """
        if epoch == 0:
            _best_val_loss = self.get_latest_loss('val')
            self.set_best_val_loss(_best_val_loss)
            self.set_best_epoch(epoch + 1)
            self.up_update_flag()
        else:
            _latest_val_loss = self.get_latest_loss('val')
            _best_val_loss = self.get_best_val_loss()
            if _latest_val_loss < _best_val_loss:
                self.set_best_val_loss(_latest_val_loss)
                self.set_best_epoch(epoch + 1)
                self.up_update_flag()
            else:
                self.down_update_flag()


class LossRegistory(ABC):
    """
    Class for calculating loss and store it.
    First, losses are calculated for each iteration and then are accumulated in EpochLoss class.
    """
    def __init__(self, internal_label_list: List[str]) -> None:
        """
        Args:
            internal_label_list (List[str]): list of internal labels
        """
        self.internal_label_list = internal_label_list

        self.batch_loss = self._init_batch_loss()       # For every batch
        self.running_loss = self._init_running_loss()   # accumlates bacth loss
        self.epoch_loss = self._init_epoch_loss()       # For every epoch

    def _init_batch_loss(self) -> Dict[str, None]:
        """
        Initialize dictinary to store loss of each internal label for each batch.

        Returns:
            Dict[str, None]: dictinary to store loss of each internal label for each batch
        """
        _batch_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _batch_loss[internal_label_name] = None
        return _batch_loss

    def _init_running_loss(self) -> Dict[str, float]:
        """
        Initialize dictinary to store loss of each internal label for each iteration.

        Returnes:
            Dict[str, float]: dictinary to store loss of each internal label for each iteration
        """
        _running_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _running_loss[internal_label_name] = 0.0
        return _running_loss

    def _init_epoch_loss(self) -> None:
        """
        Initialize dictinary to store loss of each internal label for each epoch.

        Returnes:
            Dict[str, float]: dictinary to store loss of each internal label for each epoch
        """
        _epoch_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _epoch_loss[internal_label_name] = EpochLoss()
        return _epoch_loss

    @abstractmethod
    def cal_batch_loss(
                        cls,
                        multi_output: Dict[str, float],
                        multi_label: Dict[str, int],
                        period: Tensor = None,
                        network: nn.Module = None
                    ) -> None:
        pass

    def cal_running_loss(self, batch_size: int = None) -> None:
        """
        Calculate loss for each iteration.
        batch_loss is accumulated in runnning_loss.

        Args:
            batch_size (int): batch size. Defaults to None.
        """
        assert (batch_size is not None), 'Invalid batch_size: batch_size=None.'
        for internal_label_name in self.internal_label_list:
            _running_loss = self.running_loss[internal_label_name] + (self.batch_loss[internal_label_name].item() * batch_size)
            self.running_loss[internal_label_name] = _running_loss
            self.running_loss['total'] = self.running_loss['total'] + _running_loss

    def cal_epoch_loss(self, epoch: int, phase: str, dataset_size: int = None) -> None:
        """
        Calculate loss for each epoch.

        Args:
            epoch (int): epoch number
            phase (str): pahse, ie. 'train' or 'val'
            dataset_size (int): dataset size. Defaults to None.
        """
        assert (dataset_size is not None), 'Invalid dataset_size: dataset_size=None.'
        # Update loss list label-wise
        _total = 0.0
        for internal_label_name in self.internal_label_list:
            _new_epoch_loss = self.running_loss[internal_label_name] / dataset_size
            self.epoch_loss[internal_label_name].append_epoch_loss(phase, _new_epoch_loss)
            _total = _total + _new_epoch_loss

        _total = _total / len(self.internal_label_list)
        self.epoch_loss['total'].append_epoch_loss(phase, _total)

        # Updated val_best_loss and best_epoch label-wise when val
        if phase == 'val':
            for internal_label_name in self.internal_label_list + ['total']:
                self.epoch_loss[internal_label_name].check_best_val_loss_epoch(epoch)

        # Initialize
        self.batch_loss = self._init_batch_loss()
        self.running_loss = self._init_running_loss()


class LossMixin:
    """
    Class to print epoch loss.
    """
    def print_epoch_loss(self, num_epochs: int, epoch: int) -> None:
        """
        Print train_loss and val_loss for the ith epoch.

        Args:
            num_epochs (int): ith epoch
            epoch (int): epoch numger
        """
        _total_epoch_loss = self.epoch_loss['total']
        train_loss = _total_epoch_loss.get_latest_loss('train')
        val_loss = _total_epoch_loss.get_latest_loss('val')
        epoch_comm = f"epoch [{epoch+1:>3}/{num_epochs:<3}]"
        train_comm = f"train_loss: {train_loss:>8.4f}"
        val_comm = f"val_loss: {val_loss:>8.4f}"

        updated_commemt = ''
        if (epoch > 0) and (_total_epoch_loss.is_val_loss_updated()):
            updated_commemt = '   Updated best val_loss!'
        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_commemt
        logger.logger.info(comment)


class LossWidget(LossRegistory, LossMixin):
    """
    Class for a widget to inherit multiple classes simultaneously.
    """
    pass


class ClsLoss(LossWidget):
    """
    Class to calculate loss for classification.
    """
    def __init__(self, criterion: nn.Module, internal_label_list: List[str], device: torch.device) -> None:
        """
        Args:
            criterion (nn.Module): ctiterion
            internal_label_list (List[str]): internal label list
            device (torch.device): device
        """
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output: Dict[str, Tensor], multi_label: Dict[str, Union[int, float]]) -> None:
        """
        Calculate loss for each batch.

        Args:
            multi_output (Dict[str, Tensor]): output from model
            multi_label (Dict[str, Union[int, float]]): dictionary of each label and its value
        """
        for internal_label_name in multi_label.keys():
            _output = multi_output[internal_label_name]
            _label = multi_label[internal_label_name]
            self.batch_loss[internal_label_name] = self.criterion(_output, _label)

        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in multi_label.keys():
            _total = torch.add(_total, self.batch_loss[internal_label_name])

        self.batch_loss['total'] = _total


class RegLoss(LossWidget):
    """
    Class to calculate loss for regression.
    """
    def __init__(self, criterion: nn.Module, internal_label_list: List[str], device: torch.device) -> None:
        """
        Args:
            criterion (nn.Module): ctiterion
            internal_label_list (List[str]): internal label list
            device (torch.device): device
        """
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output: Dict[str, Tensor], multi_label: Dict[str, Union[int, float]]) -> None:
        """
        Calculate loss for each batch.

        Args:
            multi_output (Dict[str, Tensor]): output from model
            multi_label (Dict[str, Union[int, float]]): dictionary of each label and its value
        """
        for internal_label_name in multi_label.keys():
            _output = multi_output[internal_label_name].squeeze()
            _label = multi_label[internal_label_name].float()
            self.batch_loss[internal_label_name] = self.criterion(_output, _label)

        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in multi_label.keys():
            _total = torch.add(_total, self.batch_loss[internal_label_name])

        self.batch_loss['total'] = _total


class DeepSurvLoss(LossWidget):
    """
    Class to calculate loss for deepsurv
    """
    def __init__(self, criterion: nn.Module, internal_label_list: List[str], device: torch.device) -> None:
        """
        Args:
            criterion (nn.Module): ctiterion
            internal_label_list (List[str]): internal label list
            device (torch.device): device
        """
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output: Dict[str, Tensor], multi_label: Dict[str, Union[int, float]], period: Tensor, network: nn.Module) -> None:
        """
        Calculate loss for each batch.

        Args:
            multi_output (Dict[str, Tensor]):  output from model
            multi_label (Dict[str, Union[int, float]]): dictionary of each label and its value
            period (Tensor): periods
            network (nn.Module): network
        """

        # multi_labelの中にinternal_label_nameは1つだけだが、
        # 上のClassification, Regressionの形を合わせておく
        for internal_label_name in multi_label.keys():
            _pred = multi_output[internal_label_name]
            _label = multi_label[internal_label_name].reshape(-1, 1)
            _period = period.reshape(-1, 1)
            self.batch_loss[internal_label_name] = self.criterion(_pred, _period, _label, network)

        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in multi_label.keys():
            _total = torch.add(_total, self.batch_loss[internal_label_name])

        self.batch_loss['total'] = _total


def create_loss_reg(task: str, criterion: nn.Module, internal_label_list: List[str], device: torch.device) -> LossRegistory:
    """
    Set LossRegistory depending on task

    Args:
        task (str): task
        criterion (nn.Module): criterion
        internal_label_list (List[str]): internal label list
        device (torch.device): device

    Returns:
        LossRegistory: LossRegistory
    """
    if task == 'classification':
        loss_reg = ClsLoss(criterion, internal_label_list, device)
    elif task == 'regression':
        loss_reg = RegLoss(criterion, internal_label_list, device)
    elif task == 'deepsurv':
        loss_reg = DeepSurvLoss(criterion, internal_label_list, device)
    else:
        logger.logger.error(f"Cannot identify task: {task}.")
    return loss_reg
