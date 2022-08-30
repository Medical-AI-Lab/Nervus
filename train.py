#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
from pathlib import Path
import torch
import models as md
import logger


log = logger.get_logger('train')

# Create directory for save
date_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
save_dir = Path('results/sets', date_name)
save_dir.mkdir(parents=True, exist_ok=True)


log.info(f"\nTraining started at {date_name}.\n")

opt = md.check_train_options()
args = opt.args
sp = md.make_split_provider(args.csv_name, args.task)

dataloaders = {
    'train': md.create_dataloader(args, sp, split='train'),
    'val': md.create_dataloader(args, sp, split='val')
    }

model = md.create_model(args, sp)

for epoch in range(args.epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()
        else:
            log.error(f"Invalid phase: {phase}.")

        split_dataloader = dataloaders[phase]
        dataset_size = len(split_dataloader.dataset)

        for i, data in enumerate(split_dataloader):
            model.optimizer.zero_grad()
            model.set_data(data)

            with torch.set_grad_enabled(phase == 'train'):
                model.forward()
                model.cal_batch_loss()

                if phase == 'train':
                    model.backward()
                    model.optimize_parameters()

            model.cal_running_loss(batch_size=len(data['Filename']))

        model.cal_epoch_loss(epoch, phase, dataset_size=dataset_size)

    model.print_epoch_loss(epoch)

    if model.is_total_val_loss_updated():
        model.store_weight()
        if (epoch > 0) and (args.save_weight == 'each'):
            model.save_weight(date_name, as_best=False)

model.save_weight(date_name, as_best=True)
model.save_learning_curve(date_name)
opt.save_parameter(date_name)

log.info('\nTraining finished.\n')
