#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import pandas as pd
from ..logger import BaseLogger
from typing import List, Dict, Union


logger = BaseLogger.get_logger(__name__)


class LabelLoss:
    """
    Class to store loss for every bash and epoch loss of each label.
    """
    def __init__(self) -> None:
        # Accumulate batch_loss(=loss * batch_size)
        self.train_batch_loss = 0.0
        self.val_batch_loss = 0.0

        # epoch_loss = batch_loss / dataset_size
        self.train_epoch_loss = []       # List[float]
        self.val_epoch_loss = []         # List[float]

        self.best_val_loss = None        # float
        self.best_epoch = None           # int
        self.is_val_loss_updated = None  # bool

    def get_loss(self, phase: str, target: str) -> Union[float, List[float]]:
        """
        Return loss depending on phase and target

        Args:
            phase (str): 'train' or 'val'
            target (str): 'batch' or 'epoch'

        Returns:
            Union[float, List[float]]: batch_loss or epoch_loss
        """
        _target = phase + '_' + target + '_loss'
        return getattr(self, _target)

    def store_batch_loss(self, phase: str, new_batch_loss: torch.FloatTensor, batch_size: int) -> None:
        """
        Add new batch loss to previous one for phase by multiplying by batch_size.

        Args:
            phase (str): 'train' or 'val'
            new_batch_loss (torch.FloatTensor): batch loss calculated by criterion
            batch_size (int): batch size
        """
        _new = new_batch_loss.item() * batch_size  # torch.FloatTensor -> float
        _prev = self.get_loss(phase, 'batch')
        _added = _prev + _new
        _target = phase + '_' + 'batch_loss'
        setattr(self, _target, _added)

    def append_epoch_loss(self, phase: str, new_epoch_loss: float) -> None:
        """
        Append epoch loss depending on phase and target

        Args:
            phase (str): 'train' or 'val'
            new_epoch_loss (float): batch loss or epoch loss
        """
        _target = phase + '_' + 'epoch_loss'
        getattr(self, _target).append(new_epoch_loss)

    def get_latest_epoch_loss(self, phase: str) -> float:
        """
        Return the latest loss of phase.

        Args:
            phase (str): train or val

        Returns:
            float: the latest loss
        """
        return self.get_loss(phase, 'epoch')[-1]

    def update_best_val_loss(self, at_epoch: int = None) -> None:
        """
        Update val_epoch_loss is the best.

        Args:
            at_epoch (int): epoch when checked
        """
        _latest_val_loss = self.get_latest_epoch_loss('val')

        if at_epoch == 1:
            self.best_val_loss = _latest_val_loss
            self.best_epoch = at_epoch
            self.is_val_loss_updated = True
        else:
            # When at_epoch > 1
            if _latest_val_loss < self.best_val_loss:
                self.best_val_loss = _latest_val_loss
                self.best_epoch = at_epoch
                self.is_val_loss_updated = True
            else:
                self.is_val_loss_updated = False


class LossStore:
    """
    Class for calculating loss and store it.
    """
    def __init__(self, label_list: List[str], num_epochs: int, dataset_info: Dict[str, int]) -> None:
        """
        Args:
            label_list (List[str]): list of internal labels
            num_epochs (int) : number of epochs
            dataset_info (Dict[str, int]):  dataset sizes of 'train' and 'val'
        """
        self.label_list = label_list
        self.num_epochs = num_epochs
        self.dataset_info = dataset_info

        # Added a special label 'total' to store total of losses of all labels.
        self.label_losses = {label_name: LabelLoss() for label_name in self.label_list + ['total']}

    def store(self, phase: str, losses: Dict[str, torch.FloatTensor], batch_size: int = None) -> None:
        """
        Store label-wise batch losses of phase to previous one.

        Args:
            phase (str): 'train' or 'val'
            losses (Dict[str, torch.FloatTensor]): loss for each label calculated by criterion
            batch_size (int): batch size

        # Note:
            self.loss_stores['total'] is already total of losses of all label, which is calculated in criterion.py,
            therefore, it is OK just to multiply by batch_size. This is done in add_batch_loss().
        """
        for label_name in self.label_list + ['total']:
            _new_batch_loss = losses[label_name]
            self.label_losses[label_name].store_batch_loss(phase, _new_batch_loss, batch_size)

    def cal_epoch_loss(self, at_epoch: int = None) -> None:
        """
        Calculate epoch loss for each phase all at once.

        Args:
            at_epoch (int): epoch number
        """
        # For each label
        for label_name in self.label_list:
            for phase in ['train', 'val']:
                _batch_loss = self.label_losses[label_name].get_loss(phase, 'batch')
                _dataset_size = self.dataset_info[phase]
                _new_epoch_loss = _batch_loss / _dataset_size
                self.label_losses[label_name].append_epoch_loss(phase, _new_epoch_loss)

        # For total, average by dataset_size and the number of labels.
        for phase in ['train', 'val']:
            _batch_loss = self.label_losses['total'].get_loss(phase, 'batch')
            _dataset_size = self.dataset_info[phase]
            _new_epoch_loss = _batch_loss / (_dataset_size * len(self.label_list))
            self.label_losses['total'].append_epoch_loss(phase, _new_epoch_loss)

        # Update val_best_loss and best_epoch.
        for label_name in self.label_list + ['total']:
            self.label_losses[label_name].update_best_val_loss(at_epoch=at_epoch)

        # Initialize batch_loss after calculating epoch loss.
        for label_name in self.label_list + ['total']:
            self.label_losses[label_name].train_batch_loss = 0.0
            self.label_losses[label_name].val_batch_loss = 0.0

    def is_val_loss_updated(self) -> bool:
        """
        Check if val_loss of 'total' is updated.

        Returns:
            bool: Updated or not
        """
        return self.label_losses['total'].is_val_loss_updated

    def get_best_epoch(self) -> int:
        """
        Returns best epoch.

        Returns:
            int: best epoch
        """
        return self.label_losses['total'].best_epoch

    def print_epoch_loss(self, at_epoch: int = None) -> None:
        """
        Print train_loss and val_loss for the ith epoch.

        Args:
            at_epoch (int): epoch number
        """
        train_epoch_loss = self.label_losses['total'].get_latest_epoch_loss('train')
        val_epoch_loss = self.label_losses['total'].get_latest_epoch_loss('val')

        _epoch_comm = f"epoch [{at_epoch:>3}/{self.num_epochs:<3}]"
        _train_comm = f"train_loss: {train_epoch_loss :>8.4f}"
        _val_comm = f"val_loss: {val_epoch_loss:>8.4f}"
        _updated_comment = ''
        if (at_epoch > 1) and (self.is_val_loss_updated()):
            _updated_comment = '   Updated best val_loss!'
        comment = _epoch_comm + ', ' + _train_comm + ', ' + _val_comm + _updated_comment
        logger.info(comment)

    def save_learning_curve(self, save_datetime_dir: str) -> None:
        """
        Save learning curve.

        Args:
            save_datetime_dir (str): save_datetime_dir
        """
        save_dir = Path(save_datetime_dir, 'learning_curve')
        save_dir.mkdir(parents=True, exist_ok=True)

        for label_name in self.label_list + ['total']:
            _label_loss = self.label_losses[label_name]
            _train_epoch_loss = _label_loss.get_loss('train', 'epoch')
            _val_epoch_loss = _label_loss.get_loss('val', 'epoch')

            df_label_epoch_loss = pd.DataFrame({
                                                'train_loss': _train_epoch_loss,
                                                'val_loss': _val_epoch_loss
                                            })

            _best_epoch = str(_label_loss.best_epoch).zfill(3)
            _best_val_loss = f"{_label_loss.best_val_loss:.4f}"
            save_name = 'learning_curve_' + label_name + '_val-best-epoch-' + _best_epoch + '_val-best-loss-' + _best_val_loss + '.csv'
            save_path = Path(save_dir, save_name)
            df_label_epoch_loss.to_csv(save_path, index=False)


def set_loss_store(label_list: List[str], num_epochs: int, dataset_info: Dict[str, int]) -> LossStore:
    """
    Return class LossStore.

    Args:
        label_list (List[str]): label list
        num_epochs (int) : number of epochs
        dataset_info (Dict[str, int]):  dataset sizes of 'train' and 'val'

    Returns:
        LossStore: LossStore
    """
    return LossStore(label_list, num_epochs, dataset_info)
