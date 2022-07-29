#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.optim as optim


class Optimizer:
    optimizers = {
        'SGD': optim.SGD,
        'Adadelta': optim.Adadelta,
        'Adam': optim.Adam,
        'RMSprop': optim.RMSprop,
        'RAdam': optim.RAdam
        }

    @classmethod
    def set_optimizer(cls, optimizer_name, network, lr):
        assert (optimizer_name in cls.optimizers), f"No specified optimizer: {optimizer_name}."

        opritmizer = cls.optimizers[optimizer_name](network.parameters(), lr=lr)
        return opritmizer
