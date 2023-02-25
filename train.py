#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import torch
from lib import (
        set_options,
        create_model,
        print_parameter,
        save_parameter,
        create_dataloader,
        BaseLogger
        )

from lib.component import (
            set_criterion,
            set_optimizer,
            set_loss_store
        )


logger = BaseLogger.get_logger(__name__)


def main(
        args_model = None,
        args_dataloader = None,
        args_conf = None,
        args_print = None,
        args_save = None
        ):

    print_parameter(args_print)

    isMLP = args_model.mlp is not None
    save_weight_policy = args_conf.save_weight_policy
    save_datetime_dir = args_conf.save_datetime_dir

    model = create_model(args_model)
    model.to_gpu(args_conf.gpu_ids)
    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}

    criterion = set_criterion(args_conf.criterion, args_conf.device)
    loss_store = set_loss_store(args_conf.label_list, args_conf.epochs, args_conf.dataset_info)
    optimizer = set_optimizer(args_conf.optimizer, model.network, args_conf.lr)

    for epoch in range(1, args_conf.epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                raise ValueError(f"Invalid phase: {phase}.")

            split_dataloader = dataloaders[phase]
            for i, data in enumerate(split_dataloader):
                optimizer.zero_grad()

                in_data, labels = model.set_data(data)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(in_data)
                    losses = criterion(outputs, labels)

                    if phase == 'train':
                        loss = losses['total']
                        loss.backward()
                        optimizer.step()

                loss_store.store(phase, losses, batch_size=len(data['imgpath']))

        # Post-processing of each epoch
        loss_store.cal_epoch_loss(at_epoch=epoch)
        loss_store.print_epoch_loss(at_epoch=epoch)
        if loss_store.is_val_loss_updated():
            model.store_weight(at_epoch=loss_store.get_best_epoch())
            if (epoch > 1) and (save_weight_policy == 'each'):
                model.save_weight(save_datetime_dir, as_best=False)

    save_parameter(args_save, save_datetime_dir + '/' + 'parameters.json')
    loss_store.save_learning_curve(save_datetime_dir)
    model.save_weight(save_datetime_dir, as_best=True)
    if isMLP:
        dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')


if __name__ == '__main__':
    try:
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTraining started at {datetime_name}.\n")

        args = set_options(datetime_name=datetime_name, phase='train')
        main(**args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTraining finished.\n')
