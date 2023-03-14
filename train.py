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


# distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os


logger = BaseLogger.get_logger(__name__)


num_gpus = 2            # num process
world_size = num_gpus   # total number of processes


def train(
        rank,
        world_size,
        args_model = None,
        args_dataloader = None,
        args_conf = None,
        args_print = None,
        args_save = None
        ):

    isMaster = (rank == 0)

    # init
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    # For GPU, Distributed package doesn't have NCCL on mac
    #dist.init_process_group('nccl', rank=rank, world_size=world_size)

    if isMaster:
        print_parameter(args_print)

    isMLP = args_model.mlp is not None
    save_weight_policy = args_conf.save_weight_policy
    save_datetime_dir = args_conf.save_datetime_dir

    #model = create_model(args_model)
    #model.to_gpu(args_conf.gpu_ids)

    model = create_model(args_model)
    model.network = DDP(model.network, device_ids=None)
    model.network.to('cpu')

    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}

    criterion = set_criterion(args_conf.criterion, args_conf.device)
    optimizer = set_optimizer(args_conf.optimizer, model.network, args_conf.lr)
    loss_store = set_loss_store(args_conf.label_list, args_conf.epochs, args_conf.dataset_info)


    for epoch in range(1, args_conf.epochs + 1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                raise ValueError(f"Invalid phase: {phase}.")

            split_dataloader = dataloaders[phase]

            #######################################################################################
            split_dataloader.sampler.set_epoch(epoch)  # https://pytorch.org/docs/stable/data.html
            ########################################################################################

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

                #print(i, 'rank', rank, 'losses', losses)
                #print(i, 'rank', rank, 'losses_total', losses['total'])

                # label-wise all-reduce
                #dist.all_reduce(losses['total'], op=dist.ReduceOp.SUM)
                #losses['total'] = losses['total'] / world_size
                #print(f"Before, rank: {rank}, {losses}")

                for label_name in losses.keys():
                    #print('All-reduce', label_name)
                    dist.all_reduce(losses[label_name], op=dist.ReduceOp.SUM)
                    losses[label_name] = losses[label_name] / world_size   # Average between process

                # Store
                # masterだけでいい？
                loss_store.store(phase, losses, batch_size=len(data['imgpath']))
                #print(f"After, rank: {rank}, {losses}")
                #print(f"rank: {rank}, epoch: {epoch}, phase: {phase}, iter: {i+1}, batch_size: {len(data['imgpath'])}")


        if isMaster:
            loss_store.cal_epoch_loss(at_epoch=epoch)
            loss_store.print_epoch_loss(at_epoch=epoch)
            if loss_store.is_val_loss_updated():
                model.store_weight(at_epoch=loss_store.get_best_epoch())
                if (epoch > 1) and (save_weight_policy == 'each'):
                    model.save_weight(save_datetime_dir, as_best=False)

    if isMaster:
        save_parameter(args_save, save_datetime_dir + '/' + 'parameters.json')
        loss_store.save_learning_curve(save_datetime_dir)
        model.save_weight(save_datetime_dir, as_best=True)
        if isMLP:
            dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')

    dist.destroy_process_group()


def main(args):
    #print('args', args)

    args_model = args['args_model']
    args_dataloader = args['args_dataloader']
    args_conf = args['args_conf']
    args_print = args['args_print']
    args_save = args['args_save']

    mp.spawn(
            train,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf,
                args_print,
                args_save
                ),
            nprocs=world_size,
            join=True
            )



if __name__ == '__main__':
    try:
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTraining started at {datetime_name}.\n")

        args = set_options(datetime_name=datetime_name, phase='train')
        #main(**args)

        os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
        os.environ['MASTER_ADDR'] = 'localhost'  #'127.0.0.1'  # 'localhost'
        os.environ['MASTER_PORT'] = '29500'      #'12355' #'8888'
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

        dist.destroy_process_group()

    else:
        logger.info('\nTraining finished.\n')
        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
