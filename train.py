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


def main(args):
    model_params = args['model']
    dataloader_params = args['dataloader']
    conf_params = args['conf']
    print_params = args['print']
    save_params = args['save']
    print_parameter(print_params)

    epochs = conf_params.epochs
    save_weight_policy = conf_params.save_weight_policy
    save_datetime_dir = conf_params.save_datetime_dir

    model = create_model(model_params)
    model.to_gpu(model_params.gpu_ids)  #! GPU
    dataloaders = {split: create_dataloader(dataloader_params, split=split) for split in ['train', 'val']}

    criterion = set_criterion(model_params.criterion, model_params.device)
    loss_store = set_loss_store(model_params.label_list, model_params.device)
    optimizer = set_optimizer(model_params.optimizer, model.network, model_params.lr)

    for epoch in range(epochs):
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

                in_data, labels = model.set_data(data)  # including to(self.device)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(in_data)

                    # loss for each label and total of losses for all labels, ie. batch_loss label-wise
                    losses = criterion(outputs, labels)
                    loss = losses['total'] # Loss of back propagation

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_store.store(losses, batch_size=len(data['imgpath']))

            dataset_size = len(split_dataloader.dataset)
            loss_store.cal_epoch_loss(epoch, phase, dataset_size=dataset_size)
        #! ---- End of phase ---
        loss_store.print_epoch_loss(epochs, epoch)

        if loss_store.is_total_val_loss_updated():
            model.store_weight()
            if (epoch > 0) and (save_weight_policy == 'each'):
                model.save_weight(save_datetime_dir, as_best=False)
    #! ----- End of epoch -----

    loss_store.save_learning_curve(save_datetime_dir)
    model.save_weight(save_datetime_dir, as_best=True)

    if model_params.mlp is not None:
        dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')

    save_parameter(save_params, save_datetime_dir + '/' + 'parameters.json')


if __name__ == '__main__':
    try:
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTraining started at {datetime_name}.\n")

        args = set_options(datetime_name=datetime_name, phase='train')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTraining finished.\n')
