#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import os
import copy
import pandas as pd
from typing import Tuple, Dict

import torch
from torch.utils.data.dataset import Dataset
from models import options
from models import env
from models import dataloader
# from models import criterion
# from models import optimizer
from models.framework import ModelNet


args = options.Options().check_options()
sp = env.SplitProvider(args.csv_name, args.task)

train_loader = dataloader.create_dataloader(args, sp,  split='train')
val_loader = dataloader.create_dataloader(args, sp,  split='val')

model = ModelNet(args, sp)

for epoch in range(args.epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            dataloader = train_loader
        else:
            model.eval()
            dataloader = val_loader

        for i, data in enumerate(dataloader):
            with torch.set_grad_enabled(phase == 'train'):
                model.set_data(data)
                model.forward()
                model.store_raw_loss()

                if phase == 'train':
                    model.backward()
                    model.optimize_paramters()

            model.store_iter_loss()

        model.store_epoch_loss(phase, len(dataloader))

    model.print_epoch_loss(args.epochs, epoch)
