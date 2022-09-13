#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dataclasses
from abc import ABC, abstractmethod
import torch
from lib.logger import Logger as logger
from typing import List, Dict
from torch import Tensor
import torch.nn as nn


@dataclasses.dataclass
class EpochLoss:
    """
    Epoch loss for each internal label is stored in this class.
    """
    train: List[float] = dataclasses.field(default_factory=list)
    val: List[float] = dataclasses.field(default_factory=list)
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

    First, losses are calculated for each iteration and accumulated in EpochLoss class.
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
        Initialize dictinary to store loss of each internal label for every batch.

        Returns:
            Dict[str, None]: dictinary to store loss of each internal label
        """
        _batch_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _batch_loss[internal_label_name] = None
        return _batch_loss

    def _init_running_loss(self) -> None:
        """
        Initialize dictinary to store loss of each internal label for every iteration.
        """
        _running_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _running_loss[internal_label_name] = 0.0
        return _running_loss

    def _init_epoch_loss(self) -> None:
        """
        Initialize dictinary to store loss of each internal label for evary epoch.
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
        Calculate loss evary iteration.
        batch_loss is accumulated in runnning_loss

        Args:
            batch_size (int): batch size. Defaults to None.
        """
        assert (batch_size is not None), 'Invalid batch_size: batch_size=None.'
        for internal_label_name in self.internal_label_list:
            _running_loss = self.running_loss[internal_label_name] + (self.batch_loss[internal_label_name].item() * batch_size)
            self.running_loss[internal_label_name] = _running_loss
            self.running_loss['total'] = self.running_loss['total'] + _running_loss

    def cal_epoch_loss(self, epoch, phase, dataset_size=None):
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
    Class to print epoch loss
    """
    def print_epoch_loss(self, num_epochs, epoch):
        _total_epoch_loss = self.epoch_loss['total']
        train_loss = _total_epoch_loss.get_latest_loss('train')
        val_loss = _total_epoch_loss.get_latest_loss('val')
        epoch_comm = f"epoch [{epoch+1:>3}/{num_epochs:<3}]"
        train_comm = f"train_loss: {train_loss:>8.4f}"
        val_comm = f"val_loss: {val_loss:>8.4f}"

        updated_commemt = ''
        if (epoch > 0) and (_total_epoch_loss.is_val_loss_updated()):
            updated_commemt = '   Updated val_loss!'
        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_commemt
        logger.logger.info(comment)


class LossWidget(LossRegistory, LossMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class ClassificationLoss(LossWidget):
    def __init__(self, criterion, internal_label_list, device):
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output, multi_label):
        for internal_label_name in multi_label.keys():
            _output = multi_output[internal_label_name]
            _label = multi_label[internal_label_name]
            self.batch_loss[internal_label_name] = self.criterion(_output, _label)

        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in multi_label.keys():
            _total = torch.add(_total, self.batch_loss[internal_label_name])

        self.batch_loss['total'] = _total


class RegressionLoss(LossWidget):
    def __init__(self, criterion, internal_label_list, device):
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output, multi_label):
        for internal_label_name in multi_label.keys():
            _output = multi_output[internal_label_name].squeeze()
            _label = multi_label[internal_label_name].float()
            self.batch_loss[internal_label_name] = self.criterion(_output, _label)

        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in multi_label.keys():
            _total = torch.add(_total, self.batch_loss[internal_label_name])

        self.batch_loss['total'] = _total


class DeepSurvLoss(LossWidget):
    def __init__(self, criterion, internal_label_list, device):
        super().__init__(internal_label_list)

        self.criterion = criterion
        self.device = device

    def cal_batch_loss(self, multi_output, multi_label, period, network):
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


def create_loss_reg(task, criterion, internal_label_list, device):
    if task == 'classification':
        loss_reg = ClassificationLoss(criterion, internal_label_list, device)
    elif task == 'regression':
        loss_reg = RegressionLoss(criterion, internal_label_list, device)
    elif task == 'deepsurv':
        loss_reg = DeepSurvLoss(criterion, internal_label_list, device)
    else:
        logger.logger.error(f"Cannot identify task: {task}.")
    return loss_reg
