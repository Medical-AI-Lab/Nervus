#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import pandas as pd
from ..logger import BaseLogger
from typing import List, Dict

from pathlib import Path


logger = BaseLogger.get_logger(__name__)


class LabelLoss:
    """
    Class to store epoch loss of each label.
    """
    def __init__(self) -> None:
        # running_loss = batch_loss + batch_loss + ...
        self.train_running_loss = 0.0
        self.val_running_loss = 0.0

        self.train_epoch_loss = []
        self.val_epoch_loss = []

        self.best_val_loss = 0.0
        self.best_epoch = 0
        self.is_val_loss_updated = False

    def add_running_loss(self, phase: str, new_running_loss: float) -> None:
        """
        Added running_loss to previous one for each phase.

        Args:
            phase (str): 'train' or 'val'
            new_running_loss (float): loss value
        """
        _target = phase + '_running_loss'
        _added = getattr(self, _target) + new_running_loss
        setattr(self, _target, _added)

    def get_running_loss(self, phase: str) -> float:
        """
        Return running loss of phase

        Args:
            phase (str): 'train' or 'val'

        Returns:
            float: running loss
        """
        _target = phase + '_running_loss'
        return getattr(self, _target)

    def append_epoch_loss(self, phase: str, new_epoch_loss: float) -> None:
        """
        Append loss to list depending on phase.

        Args:
            phase (str): train or val
            new_epoch_loss (float): loss value
        """
        _target = phase + '_epoch_loss'
        getattr(self, _target).append(new_epoch_loss)

    def get_epoch_loss(self, phase: str) -> float:
        """
        Return epoch loss of phase

        Args:
            phase (str): 'train' or 'val'

        Returns:
            float: epoch loss
        """
        _target = phase + '_epoch_loss'
        return getattr(self, _target)

    def get_latest_epoch_loss(self, phase: str) -> float:
        """
        Return the latest loss of phase.
        Args:
            phase (str): train or val
        Returns:
            float: the latest loss
        """
        _target = phase + '_epoch_loss'
        return getattr(self, _target)[-1]

    def set_best_val_loss(self, best_val_loss: float) -> None:
        """
        Set a val loss to keep it as the best loss.

        Args:
            best_val_loss (float): the best val loss
        """
        self.best_val_loss = best_val_loss

    def get_best_val_loss(self) -> float:
        """
        Return the best val loss.

        Returns:
            float: the base val loss
        """
        return self.best_val_loss

    def set_best_epoch(self, best_epoch: int) -> None:
        """
        Set best_epoch to keep it as the best epoch.

        Args:
            best_epoch (int): the best epoch
        """
        self.best_epoch = best_epoch

    def get_best_epoch(self) -> int:
        """
        Return the epoch at which val loss is the best.

        Returns:
            int: epoch
        """
        return self.best_epoch

    def update_best_val_loss(self, at_epoch: int = 0) -> None:
        """
        Update val_epoch_loss is the best.

        Args:
            at_epoch (int): epoch when checked
        """
        if at_epoch == 0:
            _best_val_loss = self.get_latest_epoch_loss('val')  # latest one should be the best.
            self.set_best_val_loss(_best_val_loss)
            self.set_best_epoch(at_epoch + 1)

        # When at_epoch > 0
        _latest_val_loss = self.get_latest_epoch_loss('val')
        _best_val_loss = self.get_best_val_loss()
        if _latest_val_loss < _best_val_loss:
            self.set_best_val_loss(_latest_val_loss)
            self.set_best_epoch(at_epoch + 1)
            self.is_val_loss_updated = True
        else:
            self.is_val_loss_updated = False

    def check_if_val_loss_updated(self) -> bool:
        """
        Check if if val_epoch_loss is updated.

        Returns:
            bool: True if val_epoch_loss was updated.
        """
        return self.is_val_loss_updated


class LossStore:
    """
    Class for calculating loss and store it.
    """
    def __init__(self, label_list, device) -> None:
        """
        Args:
            label_list (List[str]): list of internal labels
        """
        self.label_list = label_list
        self.device = device
        self.loss_stores = self._set_label_loss(self.label_list + ['total'])
        # Added a special label 'total' to store total of losses of all labels.

    def _set_label_loss(self, label_list: List[str]) -> Dict[str, LabelLoss]:
        """
        Set class to store loss of each label total loss.

        Args:
            label_list (List[str]): label list and total

        Returns:
            Dict[str, LabelLoss]: dictionary to store loss of each label for each epoch

        eg.
        {label_A: LabelLoss(), label_B: LabelLoss(), ... , total: LabelLoss()}
        """
        _losses = {label_name: LabelLoss() for label_name in label_list}
        return _losses


    def store(self, losses: Dict[str, torch.FloatTensor], phase: str, batch_size: int) -> None:
        """
        Store label-wise losses of phase.

        Args:
            losses (Dict[str, torch.FloatTensor]): loss calculated by criterion for each label
            epoch (int): epoch
            phase (str): 'train' or 'val'
            batch_size (int): batch size
        """
        for label_name in self.label_list:
            # For each label
            _new_running_loss = losses[label_name].item() * batch_size  # torch.FloatTensor -> float
            self.loss_stores[label_name].add_running_loss(phase, _new_running_loss)
            # For total
            self.loss_stores['total'].add_running_loss(phase, _new_running_loss)


    def cal_epoch_loss(self, at_epoch: int = 0, dataset_info: Dict[str, int] = None) -> None:
        """
        Calculate epoch loss for each phase all at once.

        Args:
            epoch (int): epoch number
            dataset_info (Dict[str, int]): dataset sizes of 'train' and 'val'
        """
        # For each label
        for label_name in self.label_list:
            for phase in ['train', 'val']:
                _running_loss = self.loss_stores[label_name].get_running_loss(phase)
                _new_epoch_loss = _running_loss / dataset_info[phase + '_data']
                self.loss_stores[label_name].append_epoch_loss(phase, _new_epoch_loss)

        # For total
        _total = 0.0
        for phase in ['train', 'val']:
            for label_name in self.label_list:
                _total = _total + self.loss_stores[label_name].get_epoch_loss(phase)
            _total = _total / len(self.label_list)
            self.loss_stores['total'].append_epoch_loss(phase, _total)


        # Updated val_best_loss and best_epoch label-wise
        for label_name in self.label_list + ['total']:

            self.epoch_loss[label_name].check_best_val_loss_epoch(epoch)


        # Check if updated.
        for label_name in self.label_list:
            self.loss_stores[label_name].update_best_val_loss(at_epoch=at_epoch)




        # Initialize running_loss after calculating epoch loss
        self._init_running_losses(self)


    def _init_running_losses(self) -> None:
        """
        Initialize running_loss for each phase of LabelLoss() of each label.
        """
        for label_name in self.loss_stores.keys():  # including 'total'
            setattr(self.loss_stores[label_name], 'train_running_loss', 0.0)
            setattr(self.loss_stores[label_name], 'val_running_loss', 0.0)




    def is_val_loss_updated(self) -> bool:
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
        Print train_loss and val_loss for the ith epoch.

        Args:
            num_epochs (int): ith epoch
            epoch (int): epoch number
        """

        _total_epoch_loss = self.epoch_loss['total']
        train_loss = _total_epoch_loss.get_latest_loss('train')

        val_loss = _total_epoch_loss.get_latest_loss('val')



        epoch_comm = f"epoch [{epoch+1:>3}/{num_epochs:<3}]"
        train_comm = f"train_loss: {train_loss:>8.4f}"
        val_comm = f"val_loss: {val_loss:>8.4f}"

        updated_comment = ''
        if (epoch > 0) and (_total_epoch_loss.is_val_loss_updated()):
            updated_comment = '   Updated best val_loss!'
        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_comment
        logger.info(comment)



    # For learning curve
    def save_learning_curve(self, save_datetime_dir: str) -> None:
        """
        Save learning curve.

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





def set_loss_store(label_list: List[str], device: torch.device) -> LossStore:
    """
    Return class LossStore.

    Args:
        label_list (List[str]): label list
        device (torch.device): device

    Returns:
        LossStore: LossStore
    """
    return LossStore(label_list, device)
