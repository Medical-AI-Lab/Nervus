#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim
from typing import List


class Optimizer:
    optimizers = {
        'SGD': optim.SGD,
        'Adadelta': optim.Adadelta,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'RAdam': optim.RAdam
        }


def set_optimizer(optimizer_name, network, lr):
    assert (optimizer_name in Optimizer.optimizers), f"No specified optimizer: {optimizer_name}."

    opritmizer = Optimizer.optimizers[optimizer_name](network.parameters(), lr=lr)
    return opritmizer
