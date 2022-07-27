#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim


optimizers = {
    'SGD': optim.SGD,
    'Adadelta': optim.Adadelta,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop,
    'RAdam': optim.RAdam
    }


def set_optimizer(optimizer_name, network, lr):
    assert (optimizer_name in optimizers), f"No specified optimizer: {optimizer_name}."

    opritmizer = optimizers[optimizer_name](network.parameters(), lr=lr)
    return opritmizer
