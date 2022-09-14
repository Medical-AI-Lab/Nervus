#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import torch
from lib import (
        check_train_options,
        make_split_provider,
        create_dataloader,
        create_model,
        set_logger
        )
from lib.logger import Logger as logger


def main(opt, date_name):
    args = opt.args
    sp = make_split_provider(args.csv_name, args.task)

    dataloaders = {
        'train': create_dataloader(args, sp, split='train'),
        'val': create_dataloader(args, sp, split='val')
        }

    # print_dataloder_info(dataloaders)

    model = create_model(args, sp)

    for epoch in range(args.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                logger.logger.error(f"Invalid phase: {phase}.")
                exit()

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


if __name__ == '__main__':
    set_logger()
    date_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger.logger.info(f"\nTraining started at {date_name}.\n")
    opt = check_train_options()
    main(opt, date_name)
    logger.logger.info('\nTraining finished.\n')
