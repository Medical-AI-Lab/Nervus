#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import torch
from lib import (
        check_train_options,
        set_params,
        create_model,
        set_logger
        )
from lib import Logger as logger


def main(opt):
    params = set_params(opt.args)
    params.print_parameter()
    params.print_dataset_info()

    dataloaders = params.dataloaders
    model = create_model(params)

    for epoch in range(params.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                raise ValueError(f"Invalid phase: {phase}.")

            split_dataloader = dataloaders[phase]
            for i, data in enumerate(split_dataloader):
                model.optimizer.zero_grad()
                in_data, labels = model.set_data(data)

                with torch.set_grad_enabled(phase == 'train'):
                    output = model(in_data)
                    model.cal_batch_loss(output, labels)

                    if phase == 'train':
                        model.backward()
                        model.optimize_parameters()

                model.cal_running_loss(batch_size=len(data['imgpath']))

            dataset_size = len(split_dataloader.dataset)
            model.cal_epoch_loss(epoch, phase, dataset_size=dataset_size)

        model.print_epoch_loss(params.epochs, epoch)

        if model.is_total_val_loss_updated():
            model.store_weight()
            if (epoch > 0) and (params.save_weight_policy == 'each'):
                model.save_weight(params.save_datetime_dir, as_best=False)

    model.save_learning_curve(params.save_datetime_dir)
    model.save_weight(params.save_datetime_dir, as_best=True)
    params.save_parameter()


if __name__ == '__main__':
    set_logger()
    datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logger.logger.info(f"\nTraining started at {datetime_name}.\n")

    opt = check_train_options(datetime_name)
    main(opt)

    logger.logger.info('\nTraining finished.\n')
