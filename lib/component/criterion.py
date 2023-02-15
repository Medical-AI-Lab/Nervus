#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import List


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


class Regularization(object):
    """
    Class to calculate regularization loss.

    Args:
        object (_type_): _description_
    """
    def __init__(self, order: int, weight_decay: float) -> None:
        """
        The initialization of Regularization class.

        Args:
            order: (int) norm order number
            weight_decay: (float) weight decay rate
        """
        super(Regularization, self).__init__()
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
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0.05
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.device = device

    def forward(self, risk_pred: List[float], y: List[int], e: List[int], network: nn.Module) -> float:
        """
        Calculates Negative Log Likelihood.

        Args:
            risk_pred (List[float]): prediction value
            y (List[int]): period
            e (List[int]): ground truth label
            network (nn.Network): network

        Returns:
            float: Negative Log Likelihood
        """
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)  # risk_pred and mask should be on the same device.
        mask[(y.T - y) > 0] = 0

        loss_1 = torch.exp(risk_pred) * mask
        # Note: torch.sum(loss_1, dim=0) possibly returns nan, in particular MLP.
        loss_1 = torch.sum(loss_1, dim=0) / torch.sum(mask, dim=0)
        loss_1 = torch.log(loss_1).reshape(-1, 1)

        num_occurs = torch.sum(e)

        if num_occurs.item() == 0.0:
            loss = torch.tensor([1e-7], requires_grad=True)  # To avoid zero division, set small value as loss
        else:
            neg_log_loss = -torch.sum((risk_pred - loss_1) * e) / num_occurs
            l2_loss = self.reg(network)
            loss = neg_log_loss + l2_loss
        return loss


class Criterion:
    """
    Criterion.
    """
    criterions = {
        'CEL': nn.CrossEntropyLoss,
        'MSE': nn.MSELoss,
        'RMSE': RMSELoss,
        'MAE': nn.L1Loss,
        'NLL': NegativeLogLikelihood
        }


def set_criterion(criterion_name: str, device: torch.device) -> nn.Module:
    """
    Set criterion.

    Args:
        criterion_name (str): criterion nama
        device (torch.device): device

    Returns:
        nn: criterion
    """
    assert (criterion_name in Criterion.criterions), f"No specified criterion: {criterion_name}."

    if criterion_name == 'NLL':
        criterion = Criterion.criterions[criterion_name](device).to(device)
    else:
        criterion = Criterion.criterions[criterion_name]()
    return criterion
