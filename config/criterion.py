#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


def Criterion(criterion_name):
    if criterion_name == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()

    elif criterion_name == 'MSE':
        criterion = nn.MSELoss()

    elif criterion_name == 'RMSE':
        criterion = RMSELoss()

    elif criterion_name == 'MAE':
        criterion = nn.L1Loss()

    else:
        print('No specified criterion: {}.'.format(criterion_name))
        exit()

    return criterion


# ----- EOF -----
