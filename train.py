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


# For distributed
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from typing import List


logger = BaseLogger.get_logger(__name__)



MASTER = 0


def set_device(rank: int = None, gpu_ids: List[int] = None) -> torch.device:
    """
    Define device depending on gou_ids and rank.

    Args:
        rank (int): rank, or process id
        gpu_ids (List[int]): GPU ids

    Returns:
        torch.device :device

    eg.
    When using GPU, device is define by rank-th on gpu_ids.
    gpu_ids = [1, 2, 0],
    rank=0 -> gpu_id=gpu_ids[rank]=1
    """
    if gpu_ids == []:
        return torch.device('cpu')
    else:
        return torch.device(f"cuda:{gpu_ids[rank]}")


def setup(rank: int = None, world_size: int = None, gpu_ids: List[int] = None) -> None:
    """
    Initialize the process group.

    Args:
        rank (int): rank, or process id
        world_size (int): the total number of process
        gpu_ids (List[int]): GPU ids
    """
    if gpu_ids == []:
        backend = 'gloo'  # For CPU
    else:
        backend = 'nccl'  # For GPU
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)



def train(
        rank,
        world_size,
        args_model = None,
        args_dataloader = None,
        args_conf = None
        #dataloaders = None
        ):

    isMaster = (rank == MASTER)
    gpu_ids = args_conf.gpu_ids
    if isMaster:
        isMLP = args_model.mlp is not None
        save_weight_policy = args_conf.save_weight_policy
        save_datetime_dir = args_conf.save_datetime_dir


    # Set device
    device = set_device(rank=rank, gpu_ids=gpu_ids)

    # Setup
    setup(rank=rank, world_size=world_size, gpu_ids=gpu_ids)


    model = create_model(args_model)  #! -> framework.py
    model.network.to(device)
    model.network = DDP(model.network, device_ids=None)  # device_ids must be None on both CPU and GPUs.

    criterion = set_criterion(args_conf.criterion, device)
    optimizer = set_optimizer(args_conf.optimizer, model.network, args_conf.lr)
    if isMaster:
        loss_store = set_loss_store(args_conf.label_list, args_conf.epochs, args_conf.dataset_info)

    #! -----------
    #! Should be here ???
    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}
    #! -----------

    for epoch in range(1, args_conf.epochs + 1):
        for phase in ['train', 'val']:
            # Sync all processes before starting with a new epoch of training
            dist.barrier()

            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
            else:
                raise ValueError(f"Invalid phase: {phase}.")

            split_dataloader = dataloaders[phase]
            split_dataloader.sampler.set_epoch(epoch)  #! Here ?

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

                # Store
                if isMaster:
                    loss_store.store(phase, losses, batch_size=len(data['imgpath']))

        if isMaster:
            loss_store.cal_epoch_loss(at_epoch=epoch)
            loss_store.print_epoch_loss(at_epoch=epoch)

            if loss_store.is_val_loss_updated():
                model.store_weight(at_epoch=loss_store.get_best_epoch())
                if (epoch > 1) and (save_weight_policy == 'each'):
                    model.save_weight(save_datetime_dir, as_best=False)

        #! Wait until master store and save weight
        dist.barrier()

    dist.barrier()  #! No need?
    if isMaster:
        loss_store.save_learning_curve(save_datetime_dir)
        model.save_weight(save_datetime_dir, as_best=True)
        #! ----
        if isMLP:
            dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')

    dist.destroy_process_group()


def set_world_size(gpu_ids: List[int]) -> int:
    """
    Set world_size, ie, total number of processes.
    Not that 1CPU/1Process or 1GPU/1Process.

    Args:
        gpu_ids (List[int]): GPU ids

    Returns:
        int: world_size
    """
    if gpu_ids == []:
        return 4 #! <---- temporary  #1  # When CPU
    else:
        return len(gpu_ids)  # When GPU


def main(args):
    args_model = args['args_model']
    args_dataloader = args['args_dataloader']
    args_conf = args['args_conf']
    args_print = args['args_print']
    args_save = args['args_save']

    # Print parameter
    print_parameter(args_print)

    # Make dataloader
    #dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}

    world_size = set_world_size(args_conf.gpu_ids)

    mp.spawn(
            train,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf
                #dataloaders
                ),
            nprocs=world_size,
            join=True
            )

    # Save parameter
    save_datetime_dir = args_conf.save_datetime_dir
    save_parameter(args_save, save_datetime_dir + '/' + 'parameters.json')

    # Save scaler
    #if args_model.mlp is not None:
    #    dataloaders['train'].dataset.save_scaler(save_datetime_dir + '/' + 'scaler.pkl')





if __name__ == '__main__':
    try:
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTraining started at {datetime_name}.\n")

        args = set_options(datetime_name=datetime_name, phase='train')

        os.environ['GLOO_SOCKET_IFNAME'] = 'en0'  # macだからen0？、GPUだといらないかも
        os.environ['MASTER_ADDR'] = 'localhost'   #'127.0.0.1'  # 'localhost'
        os.environ['MASTER_PORT'] = '29500'       #'12355' #'8888'

        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTraining finished.\n')

        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    #finally:
    #    dist.destroy_process_group()
