#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from pathlib import Path
from typing import Tuple, Dict

import torch

from models.options import Options
from models.env import SplitProvider
from models.dataloader import create_dataloader
from models.framework import create_model
from logger.logger import Logger

logger = Logger.get_logger('train')


# Create directory for save
date_now = datetime.datetime.now()
date_name = date_now.strftime('%Y-%m-%d-%H-%M-%S')
save_dir = Path('results/sets', date_name)
save_dir.mkdir(parents=True, exist_ok=True)


args = Options().check_options()
sp = SplitProvider(args.csv_name, args.task)

train_loader = create_dataloader(args, sp, split='train')
val_loader = create_dataloader(args, sp, split='val')
model = create_model(args, sp)


for epoch in range(args.epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
            split_dataloader = train_loader
        else:
            model.eval()
            split_dataloader = val_loader

        for i, data in enumerate(split_dataloader):
            model.optimizer.zero_grad()
            model.set_data(data)

            with torch.set_grad_enabled(phase == 'train'):
                model.forward()
                model.cal_batch_loss()

                if phase == 'train':
                    model.backward()
                    model.optimize_paramters()

            batch_size = len(data['split'])
            model.cal_running_loss(batch_size)

        dataset_size = len(split_dataloader.dataset)
        model.cal_epoch_loss(phase, dataset_size)

    # if args.save_weight == 'each':
        # save weight every time val loss decreases label-wise.
        # model.save_weight(save_dir, frequency)

    model.print_epoch_loss(args.epochs, epoch)

# Save parameters
# Save weight
# Save learning curve for oveerall and label-wise

logger.info('Training finisehd.')
