#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim
import torch.nn as nn


class Optimizer:
    """
    Optimizer.
    """
    optimizers = {
        'SGD': optim.SGD,
        'Adadelta': optim.Adadelta,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'RAdam': optim.RAdam
        }


def set_optimizer(optimizer_name: str, network: nn.Module, lr: float) -> optim:
    """
    Set optimizer.

    Args:
        optimizer_name (str): criteon name
        network (torch.nn.Module): network
        lr (float): learning rate

    Returns:
        torch.optim: optimizer
    """
    assert (optimizer_name in Optimizer.optimizers), f"No specified optimizer: {optimizer_name}."

    opritmizer = Optimizer.optimizers[optimizer_name](network.parameters())
    return opritmizer

