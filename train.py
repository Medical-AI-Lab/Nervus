#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib import (
        set_options,
        print_parameter,
        save_parameter,
        create_dataloader,
        create_model,
        is_master,
        set_world_size,
        setup,
        set_device,
        setenv,
        BaseLogger
        )
from lib.component import (
            set_criterion,
            set_optimizer,
            set_loss_store
            )


logger = BaseLogger.get_logger(__name__)


def train(
        rank,
        world_size,
        args_model = None,
        args_dataloader = None,
        args_conf = None
        ):

    gpu_ids = args_conf.gpu_ids
    setup(rank=rank, world_size=world_size, gpu_ids=gpu_ids)

    device = set_device(rank=rank, gpu_ids=gpu_ids)
    isMaster = is_master(rank)
    isDistributed = (gpu_ids != [])
    if isMaster:
        isMLP = args_model.mlp is not None
        save_weight_policy = args_conf.save_weight_policy
        save_datetime_dir = args_conf.save_datetime_dir

    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}

    model = create_model(args_model)
    model.network.to(device)
    model.network = DDP(model.network, device_ids=None)  # device_ids must be None on both CPU and GPUs.

    criterion = set_criterion(args_conf.criterion, device)
    optimizer = set_optimizer(args_conf.optimizer, model.network, args_conf.lr)
    if isMaster:
        loss_store = set_loss_store(args_conf.label_list, args_conf.epochs, args_conf.dataset_info)

    for epoch in range(1, args_conf.epochs + 1):
        for phase in ['train', 'val']:
            # Sync all processes before starting with a new epoch
            dist.barrier()

            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                raise ValueError(f"Invalid phase: {phase}.")

            split_dataloader = dataloaders[phase]
            """
            if isDistributed:
                # In distributed mode, calling the set_epoch() method
                # at the beginning of each epoch before creating
                # the DataLoader iterator is necessary
                # to make shuffling work properly across multiple epochs.
                # Otherwise, the same ordering will be always used.
                split_dataloader.sampler.set_epoch(epoch)
            """
            split_dataloader.sampler.set_epoch(epoch)  #! For debugging on CPU.

            for i, data in enumerate(split_dataloader):
                optimizer.zero_grad()

                in_data, labels = model.set_data(data, device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(in_data)
                    losses = criterion(outputs, labels)

                    if phase == 'train':
                        loss = losses['total']
                        loss.backward()
                        optimizer.step()

                # label-wise all-reduce
                for label_name in losses.keys():
                    dist.all_reduce(losses[label_name], op=dist.ReduceOp.SUM)

                if isMaster:
                    loss_store.store(phase, losses, batch_size=len(data['imgpath']))

        if isMaster:
            loss_store.cal_epoch_loss(at_epoch=epoch)
            loss_store.print_epoch_loss(at_epoch=epoch)

            if loss_store.is_val_loss_updated():
                model.store_weight(at_epoch=loss_store.get_best_epoch())
                if (epoch > 1) and (save_weight_policy == 'each'):
                    model.save_weight(save_datetime_dir, as_best=False)

    # Sync all processes after all epochs.
    dist.barrier()

    if isMaster:
        loss_store.save_learning_curve(save_datetime_dir)
        model.save_weight(save_datetime_dir, as_best=True)
        if isMLP:
            dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')

    dist.destroy_process_group()


def main(args):
    args_model = args['args_model']
    args_dataloader = args['args_dataloader']
    args_conf = args['args_conf']
    args_print = args['args_print']
    args_save = args['args_save']
    print_parameter(args_print)

    world_size = set_world_size(args_conf.gpu_ids, on_cpu=True)
    mp.spawn(
            train,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf,
                ),
            nprocs=world_size,
            join=True
            )

    save_datetime_dir = args_conf.save_datetime_dir
    save_parameter(args_save, save_datetime_dir + '/' + 'parameters.json')


if __name__ == '__main__':
    try:
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTraining started at {datetime_name}.\n")

        setenv()
        args = set_options(datetime_name=datetime_name, phase='train')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTraining finished.\n')
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
