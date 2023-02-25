#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Dict, Union

# Alias of typing
# eg. {'labels': {'label_A: torch.Tensor([0, 1, ...]), ...}}
LabelDict = Dict[str, Dict[str, Union[torch.IntTensor, torch.FloatTensor]]]


class RMSELoss(nn.Module):
    """
    Class to calculate RMSE.
    """
    def __init__(self, eps: float = 1e-7) -> None:
        """
        Args:
            eps (float, optional): value to avoid 0. Defaults to 1e-7.
        """
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat: float, y: float) -> torch.FloatTensor:
        """
        Calculate RMSE.

        Args:
            yhat (float): prediction value
            y (float): ground truth value

        Returns:
            float: RMSE
        """
        _loss = self.mse(yhat, y) + self.eps
        return torch.sqrt(_loss)


class Regularization:
    """
    Class to calculate regularization loss.

    Args:
        object (object): object
    """
    def __init__(self, order: int, weight_decay: float) -> None:
        """
        The initialization of Regularization class.

        Args:
            order: (int) norm order number
            weight_decay: (float) weight decay rate
        """
        super().__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, network: nn.Module) -> torch.FloatTensor:
        """"
        Calculates regularization(self.order) loss for network.

        Args:
            model: (torch.nn.Module object)

        Returns:
            torch.FloatTensor: the regularization(self.order) loss
        """
        reg_loss = 0
        for name, w in network.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    """
    Class to calculate RMSE.
    """
    def __init__(self, device: torch.device) -> None:
        """
        Args:
            device (torch.device): device
        """
        super().__init__()
        self.L2_reg = 0.05
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.device = device

    def forward(
                self,
                output: torch.FloatTensor,
                label: torch.IntTensor,
                periods: torch.FloatTensor,
                network: nn.Module
                ) -> torch.FloatTensor:
        """
        Calculates Negative Log Likelihood.

        Args:
            output (torch.FloatTensor): prediction value, ie risk prediction
            label (torch.IntTensor): occurrence of event
            periods (torch.FloatTensor): period
            network (nn.Network): network

        Returns:
            torch.FloatTensor: Negative Log Likelihood
        """
        mask = torch.ones(periods.shape[0], periods.shape[0]).to(self.device)  # output and mask should be on the same device.
        mask[(periods.T - periods) > 0] = 0

        _loss = torch.exp(output) * mask
        # Note: torch.sum(_loss, dim=0) possibly returns nan, in particular MLP.
        _loss = torch.sum(_loss, dim=0) / torch.sum(mask, dim=0)
        _loss = torch.log(_loss).reshape(-1, 1)

        num_occurs = torch.sum(label)

        if num_occurs.item() == 0.0:
            loss = torch.tensor([1e-7], requires_grad=True)  # To avoid zero division, set small value as loss
        else:
            neg_log_loss = -torch.sum((output - _loss) * label) / num_occurs
            l2_loss = self.reg(network)
            loss = neg_log_loss + l2_loss
        return loss


class ClsCriterion:
    """
    Class of criterion for classification.
    """
    def __init__(self, device: torch.device = None) -> None:
        """
        Set CrossEntropyLoss.
        """
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def __call__(
                self,
                outputs: Dict[str, torch.FloatTensor],
                labels: Dict[str, LabelDict]
                ) -> Dict[str, torch.FloatTensor]:
        """
        Calculate loss.

        Args:
            outputs (Dict[str, torch.FloatTensor], optional): output
            labels (Dict[str, LabelDict]): labels

        Returns:
            Dict[str, torch.FloatTensor]: loss for each label and their total loss

        # No reshape and no cast:
        output: [64, 2]: torch.float32
        label:  [64]   : torch.int64
        label.dtype should be torch.int64, otherwise nn.CrossEntropyLoss() causes error.

        eg.
        outputs = {'label_A': [[0.8, 0.2], ...] 'label_B': [[0.7, 0.3]], ...}
        labels = { 'labels': {'label_A: 1: [1, 1, 0, ...], 'label_B': [0, 0, 1, ...], ...} }

        -> losses = {total: loss_total, label_A: loss_A, label_B: loss_B, ... }
        """
        _labels = labels['labels']

        # loss for each label and total of their losses
        losses = dict()
        losses['total'] = torch.tensor([0.0]).to(self.device)
        for label_name in labels['labels'].keys():
            _output = outputs[label_name]
            _label = _labels[label_name]
            _label_loss = self.criterion(_output, _label)
            losses[label_name] = _label_loss
            losses['total'] = torch.add(losses['total'], _label_loss)
        return losses


class RegCriterion:
    """
    Class of criterion for regression.
    """
    def __init__(self, criterion_name: str = None, device: torch.device = None) -> None:
        """
        Set MSE, RMSE or MAE.
        """
        self.device = device

        if criterion_name == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion_name == 'RMSE':
            self.criterion = RMSELoss()
        elif criterion_name == 'MAE':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Invalid criterion for regression: {criterion_name}.")

    def __call__(
                self,
                outputs: Dict[str, torch.FloatTensor],
                labels: Dict[str, LabelDict]
                ) -> Dict[str, torch.FloatTensor]:
        """
        Calculate loss.

        Args:
            Args:
            outputs (Dict[str, torch.FloatTensor], optional): output
            labels (Dict[str, LabelDict]): labels

        Returns:
            Dict[str, torch.FloatTensor]: loss for each label and their total loss

        # Reshape and cast
        output: [64, 1] -> [64]: torch.float32
        label:             [64]: torch.float64 -> torch.float32
        # label.dtype should be torch.float32, otherwise cannot backward.

        eg.
        outputs = {'label_A': [[10.8], ...] 'label_B': [[15.7]], ...}
        labels = {'labels': {'label_A: 1: [10, 9, ...], 'label_B': [12, 17,], ...}}
        -> losses = {total: loss_total, label_A: loss_A, label_B: loss_B, ... }
        """
        _outputs = {label_name: _output.squeeze() for label_name, _output in outputs.items()}
        _labels = {label_name: _label.to(torch.float32) for label_name, _label in labels['labels'].items()}

        # loss for each label and total of their losses
        losses = dict()
        losses['total'] = torch.tensor([0.0]).to(self.device)
        for label_name in labels['labels'].keys():
            _output = _outputs[label_name]
            _label = _labels[label_name]
            _label_loss = self.criterion(_output, _label)
            losses[label_name] = _label_loss
            losses['total'] = torch.add(losses['total'], _label_loss)
        return losses


class DeepSurvCriterion:
    """
    Class of criterion for deepsurv.
    """
    def __init__(self,device: torch.device = None) -> None:
        """
        Set NegativeLogLikelihood.

        Args:
            device (torch.device, optional): device
        """
        self.device = device
        self.criterion = NegativeLogLikelihood(self.device).to(self.device)

    def __call__(
                self,
                outputs: Dict[str, torch.FloatTensor],
                labels: Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]
                ) -> Dict[str, torch.FloatTensor]:
        """
        Calculate loss.

        Args:
            outputs (Dict[str, torch.FloatTensor], optional): output
            labels (Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]): labels, periods, and network

        Returns:
            Dict[str, torch.FloatTensor]: loss for each label and their total loss

        # Reshape and no cast
        output:         [64, 1]: torch.float32
        label:  [64] -> [64, 1]: torch.int64
        period: [64] -> [64, 1]: torch.float32

        eg.
        outputs = {'label_A': [[10.8], ...] 'label_B': [[15.7]], ...}
        labels = {
                    'labels': {'label_A: 1: [1, 0, 1, ...] },
                    'periods': [5, 10, 7, ...],
                    'network': network
                }
        -> losses = {total: loss_total, label_A: loss_A, label_B: loss_B, ... }
        """
        _labels = {label_name: _label.reshape(-1, 1) for label_name, _label in labels['labels'].items()}
        _periods = labels['periods'].reshape(-1, 1)
        _network = labels['network']

        # loss for each label and total of their losses
        losses = dict()
        losses['total'] = torch.tensor([0.0]).to(self.device)
        for label_name in labels['labels'].keys():
            _output = outputs[label_name]
            _label = _labels[label_name]
            _label_loss = self.criterion(_output, _label, _periods, _network)
            losses[label_name] = _label_loss
            losses['total'] = torch.add(losses['total'], _label_loss)
        return losses


def set_criterion(
                criterion_name: str,
                device: torch.device
                ) -> Union[ClsCriterion, RegCriterion, DeepSurvCriterion]:
    """
    Return criterion class

    Args:
        criterion_name (str): criterion name
        device (torch.device): device

    Returns:
        Union[ClsCriterion, RegCriterion, DeepSurvCriterion]: criterion class
    """

    if criterion_name == 'CEL':
        return ClsCriterion(device=device)

    elif criterion_name in ['MSE', 'RMSE', 'MAE']:
        return RegCriterion(criterion_name=criterion_name, device=device)

    elif criterion_name == 'NLL':
        return DeepSurvCriterion(device=device)

    else:
        raise ValueError(f"Invalid criterion: {criterion_name}.")
