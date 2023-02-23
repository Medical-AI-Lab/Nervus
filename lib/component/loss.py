#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import pandas as pd
from ..logger import BaseLogger
from typing import List, Dict


logger = BaseLogger.get_logger(__name__)


class LabelLoss:
    """
    Class to store loss for every bash and epoch loss of each label.
    """
    def __init__(self) -> None:
        # Accumulate batch_loss(=loss * batch_size)
        self.train_batch_loss = []  # List[float]
        self.val_batch_loss = []

        # epoch_loss = sum(batch_loss) / dataset_size
        self.train_epoch_loss = []
        self.val_epoch_loss = []

        self.best_val_loss = None        # float
        self.best_epoch = None           # int
        self.is_val_loss_updated = None  # bool

    def append_loss(self, new_loss: float, phase: str, target: str) -> None:
        """
        Append loss depending on phase and target

        Args:
            new_epoch_loss (float): batch loss or epoch loss
            phase (str): 'train' or 'val'
            target (str): 'batch' or 'epoch'
        """
        _target = phase + '_' + target + '_loss'
        getattr(self, _target).append(new_loss)

    def get_loss(self, phase: str, target: str) -> List[float]:
        """
        Return loss depending on phase and target

        Args:
            new_epoch_loss (float): batch loss or epoch loss
            phase (str): 'train' or 'val'
            target (str): 'batch' or 'epoch'

        Returns:
            float: loss list
        """
        _target = phase + '_' + target + '_loss'
        return getattr(self, _target)

    def get_latest_epoch_loss(self, phase: str) -> float:
        """
        Return the latest loss of phase.
        Args:
            phase (str): train or val
        Returns:
            float: the latest loss
        """
        return self.get_loss(phase, 'epoch')[-1]

    def update_best_val_loss(self, at_epoch: int = 0) -> None:
        """
        Update val_epoch_loss is the best.

        Args:
            at_epoch (int): epoch when checked
        """
        if at_epoch == 0:
            _best_val_loss = self.get_latest_epoch_loss('val')
            self.best_val_loss = _best_val_loss
            self.best_epoch = at_epoch + 1
            self.is_val_loss_updated = True
        else:
            # When at_epoch > 0
            _latest_val_loss = self.get_latest_epoch_loss('val')
            if _latest_val_loss < self.best_val_loss:
                self.best_val_loss = _latest_val_loss
                self.best_epoch = at_epoch + 1
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

    def store(self, losses: Dict[str, torch.FloatTensor], phase: str, batch_size: int = 0) -> None:
        """
        Store label-wise losses of phase.

        Args:
            losses (Dict[str, torch.FloatTensor]): loss for each label calculated by criterion
            phase (str): 'train' or 'val'
            batch_size (int): batch size
        """
        # self.loss_stores['total'] is already total of losses of all label, calculated by criterion.
        for label_name in self.label_list + ['total']:
            _label_batch_loss = losses[label_name].item() * batch_size  # torch.FloatTensor -> float
            self.label_losses[label_name].append_loss(_label_batch_loss, phase, 'batch')

    def cal_epoch_loss(self, at_epoch: int = 0) -> None:
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
                _new_epoch_loss = sum(_batch_loss) / _dataset_size
                self.label_losses[label_name].append_loss(_new_epoch_loss, phase, 'epoch')

        # For total, average by dataset_size and the number of labels.
        for phase in ['train', 'val']:
            _batch_loss = self.label_losses['total'].get_loss(phase, 'batch')
            _dataset_size = self.dataset_info[phase]
            _new_epoch_loss = sum(_batch_loss) / (_dataset_size * len(self.label_list))
            self.label_losses['total'].append_loss(_new_epoch_loss, phase, 'epoch')

        # Update val_best_loss and best_epoch.
        for label_name in self.label_list + ['total']:
            self.label_losses[label_name].update_best_val_loss(at_epoch=at_epoch)

        # Initialize batch_loss after calculating epoch loss.
        for label_name in self.label_list + ['total']:
            self.label_losses[label_name].train_batch_loss = []
            self.label_losses[label_name].val_batch_loss = []

    def is_val_loss_updated(self) -> bool:
        """
        Check if val_loss of 'total' updated.

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

    def print_epoch_loss(self, at_epoch: int = 0) -> None:
        """
        Print train_loss and val_loss for the ith epoch.

        Args:
            num_epochs (int): ith epoch
            at_epoch (int): epoch number
        """
        train_epoch_loss = self.label_losses['total'].get_latest_epoch_loss('train')
        val_epoch_loss = self.label_losses['total'].get_latest_epoch_loss('val')

        epoch_comm = f"epoch [{at_epoch + 1:>3}/{self.num_epochs:<3}]"
        train_comm = f"train_loss: {train_epoch_loss :>8.4f}"
        val_comm = f"val_loss: {val_epoch_loss:>8.4f}"

        updated_comment = ''
        if (at_epoch > 0) and (self.is_val_loss_updated()):
            updated_comment = '   Updated best val_loss!'

        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_comment
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
