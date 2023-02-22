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

    def forward(self, yhat: float, y: float) -> float:
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

    def __call__(self, network: nn.Module) -> float:
        """"
        Calculates regularization(self.order) loss for network.

        Args:
            model: (torch.nn.Module object)

        Returns:
            torch.Tensor[float]: the regularization(self.order) loss
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
                periods: torch.IntTensor,
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


CRITERIONS = {
        'CEL': nn.CrossEntropyLoss,
        'MSE': nn.MSELoss,
        'RMSE': RMSELoss,
        'MAE': nn.L1Loss,
        'NLL': NegativeLogLikelihood
        }


class ClsCriterion:
    """
    Class of criterion for classification.
    """
    def __init__(self, criterion_name: str = None) -> None:
        """
        Set CrossEntropyLoss.
        """
        self.criterion_name = criterion_name
        self.criterion = nn.CrossEntropyLoss()

    def __call__(
                self,
                output: torch.FloatTensor = None,
                label: torch.IntTensor = None
                ) -> torch.FloatTensor:
        """
        Calculate loss.

        Args:
            output (torch.FloatTensor): output
            label (torch.IntTensor): label

        Returns:
            torch.FloatTensor: loss

        # No reshape:
        # eg.
        # output: [64, 2]
        # label:  [64, 2]
        """
        loss = self.criterion(output, label)
        return loss


class RegCriterion:
    """
    Class of criterion for regression.
    """
    def __init__(self, criterion_name: str = None) -> None:
        """
        Set MSE, RMSE or MAE.
        """
        self.criterion_name = criterion_name
        self.criterion = CRITERIONS[self.criterion_name]()

    def __call__(
                self,
                output: torch.FloatTensor = None,
                label: torch.FloatTensor = None
                ) -> torch.FloatTensor:
        """
        Calculate loss.

        Args:
            output (torch.FloatTensor): output
            label (torch.FloatTensor): label

        Returns:
            torch.FloatTensor: loss

        Reshape
        eg.
        output: [64, 1] -> [64]
        label:             [64]
        """
        output = output.squeeze()
        loss = self.criterion(output, label)
        return loss


class DeepSurvCriterion:
    """
    Class of criterion for deepsurv.
    """
    def __init__(self, criterion_name: str = None, device: torch.device = None) -> None:
        """
        Set NegativeLogLikelihood.

        Args:
            device (torch.device, optional): device
        """
        self.criterion_name = criterion_name
        self.device = device
        self.criterion = NegativeLogLikelihood(self.device).to(self.device)

    def __call__(
                self,
                output: torch.FloatTensor = None,
                label: torch.IntTensor = None,
                periods: torch.IntTensor = None,
                network: nn.Module = None
                ) -> torch.FloatTensor:
        """
        Calculate loss.

        Args:
            output (torch.FloatTensor): output, or risk prediction
            label (torch.IntTensor): label
            periods (torch.IntTensor): period
            network (nn.Module): network. Its weight is used when calculating loss.

        Returns:
            torch.FloatTensor: loss

        Reshape
        eg.
        output:          [64, 1]
        label:   [64] -> [64, 1]
        periods: [64] -> [64, 1]
        """
        label = label.reshape(-1, 1)
        periods = periods.reshape(-1, 1)
        loss = self.criterion(output, label, periods, network)
        return loss


def select_criterion(
                criterion_name: str,
                device: torch.device
                ) -> Union[ClsCriterion, RegCriterion, DeepSurvCriterion]:
    """
    Set criterion for task.

    Args:
        criterion_name (str): criterion name
        device (torch.device): device

    Returns:
        Union[ClsCriterion, RegCriterion, DeepSurvCriterion]: criterion
    """

    if criterion_name == 'CEL':
        return ClsCriterion(criterion_name=criterion_name)

    elif criterion_name in ['MSE', 'RMSE', 'MAE']:
        return RegCriterion(criterion_name=criterion_name)

    elif criterion_name == 'NLL':
        return  DeepSurvCriterion(criterion_name=criterion_name, device=device)

    else:
        raise ValueError(f"Invalid criterion: {criterion_name}.")


class Criterion:
    """
    criterion class.
    """
    def __init__(self, criterion_name: str, device: torch.device) -> None:
        """
        Args:
            criterion_name (str): criterion name
            device (torch.device): device
        """
        self.criterion_name = criterion_name
        self.device = device
        self.criterion = select_criterion(self.criterion_name, self.device)

    def __call__(
                self,
                outputs: Dict[str, torch.FloatTensor],
                labels: Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]],
                ) -> Dict[str, torch.FloatTensor]:
        """
        Calculate loss for each label.

        Args:
            outputs (Dict[str, torch.FloatTensor], optional): output
            labels (Dict[str, Union[LabelDict, torch.IntTensor, nn.Module]]): input of model and data for calculating loss.

        Return:
            Dict[str, torch.FloatTensor]: loss for each label and their total loss

        eg.
        outputs = {'label_A': [0.8, 0.2], 'label_B': [0.7, 0.3], ...}
        labels = {
                    'labels': {'label_A: 1: [1, 0, 1, ..], 'label_B': [0, 0, 1, ...], ...},
                    'periods': [5, 10, 7, ...],
                    'network': network
                }
        """
        _labels = labels['labels']
        _label_list = labels['labels'].keys()

        if 'periods' not in labels:
            _args_nll = {}
        else:
            _args_nll = {'periods': labels['periods'], 'network': labels['networks']}

        losses = dict()
        _total = torch.tensor([0.0]).to(self.device)
        for label_name in _label_list:
            _args = {**{'output': outputs[label_name], 'label': _labels[label_name]}, **_args_nll}
            # For each label
            losses[label_name] = self.criterion(**_args)
            # For total
            _total = torch.add(_total, losses[label_name])

        losses['total'] = _total
        return losses


def set_criterion(criterion_name: str, device: torch.device) -> Criterion:
    """
    Set class Criterion.

    Args:
        criterion_name (str): criterion name
        device (torch.device): device

    Returns:
        Criterion: Criterion
    """
    return Criterion(criterion_name, device)
