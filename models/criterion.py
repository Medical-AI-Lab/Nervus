#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        _loss = self.mse(yhat, y) + self.eps
        return torch.sqrt(_loss)


class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class
        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.
        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss


class NegativeLogLikelihood(nn.Module):
    def __init__(self, device):
        super(NegativeLogLikelihood, self).__init__()
        self.L2_reg = 0.05
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)
        self.device = device

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0]).to(self.device)  # risk_pred and mask should be on the same device.
        mask[(y.T - y) > 0] = 0
        loss_1 = torch.exp(risk_pred) * mask
        loss_1 = torch.sum(loss_1, dim=0) / torch.sum(mask, dim=0)
        loss_1 = torch.log(loss_1).reshape(-1, 1)
        num_occurs = torch.sum(e)
        if num_occurs.item() == 0.0:
            loss = torch.tensor([1e-7], requires_grad=True)  # To avoid zero division, set small value as loss
        else:
            neg_log_loss = -torch.sum((risk_pred-loss_1) * e) / num_occurs
            l2_loss = self.reg(model)
            loss = neg_log_loss + l2_loss
        return loss


criterions = {
    'CEL': nn.CrossEntropyLoss,
    'MSE': nn.MSELoss,
    'RMSE': RMSELoss,
    'MAE': nn.L1Loss,
    'NLL': NegativeLogLikelihood
    }


def set_criterion(criterion_name, device):
    assert (criterion_name in criterions), f"No specified criterion: {criterion_name}."

    if criterion_name == 'NLL':
        criterion = criterions[criterion_name](device).to(device)
    else:
        criterion = criterions[criterion_name]()
    return criterion
