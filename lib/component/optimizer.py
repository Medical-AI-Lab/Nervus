#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim
import torch.nn as nn


def set_optimizer(optimizer_name: str, network: nn.Module, lr: float) -> optim:
    """
    Set optimizer.
    Args:
        optimizer_name (str): criterion name
        network (torch.nn.Module): network
        lr (float): learning rate
    Returns:
        torch.optim: optimizer
    """
    optimizers = {
        'SGD': 'SGD',
        'Adadelta': 'Adadelta',
        'Adam': 'Adam',
        'RMSprop': 'RMSprop',
        'RAdam': 'RAdam'
        }

    assert (optimizer_name in optimizers), f"No specified optimizer: {optimizer_name}."

    _optim = getattr(optim, optimizers[optimizer_name])

    if lr is None:
        optimizer = _optim(network.parameters())
    else:
        optimizer = _optim(network.parameters(), lr=lr)
    return optimizer
