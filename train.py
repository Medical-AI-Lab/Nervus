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



logger = BaseLogger.get_logger(__name__)


#num_gpus = 4 #8 #2      # num process . 4 -> rank = 0,1,2,3, 1 -> rank=0
#world_size = num_gpus   # total number of processes


MASTER = 0


def train(
        rank,
        world_size,
        args_model = None,
        args_dataloader = None,
        args_conf = None,
        #args_print = None,
        args_save = None
        ):


    isMaster = (rank == MASTER)


    # init
    #dist.init_process_group('gloo', rank=rank, world_size=world_size)
    # For GPU, Distributed package doesn't have NCCL on mac
    #dist.init_process_group('nccl', rank=rank, world_size=world_size)
    #
    #num_gpus = len(args_conf.gpu_ids)a
    #setup(num_gpus=num_gpus, rank=rank, world_size=world_size)
    gpu_ids = args_conf.gpu_ids
    if gpu_ids != []:
        backend = 'nccl'  # For GPU
    else:
        backend = 'gloo'  # For CPU

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"rank: {rank}, init!, backend: {backend}")


    if isMaster:
        #print_parameter(args_print)
        isMLP = args_model.mlp is not None
        save_weight_policy = args_conf.save_weight_policy
        save_datetime_dir = args_conf.save_datetime_dir


    #model = create_model(args_model)
    #model.to_gpu(args_conf.gpu_ids)

    model = create_model(args_model)


    # Set device
    if gpu_ids == []:  # When CPU
        device = torch.device('cpu')
        device_ids = None
    else:
        #! When using GPU, define device depending on process
        # rank starts from 0.
        # eg. gpu_ids = [0, 1, 2]
        # if rank = 0 -> gpu_id=gpu_ids[rank] = 0 corresponds to rank and GPU id
        device = torch.device(f"cuda:{gpu_ids[rank]}")
        device_ids = [device]



    #! For multi-device modules and CPU modules, device_ids must be None.
    #model.network = DDP(model.network, device_ids=None)
    #model.network.to('cpu')

    #! For single-device modules, device_ids can contain exactly one device id,
    #! which represents the only CUDA device where the input module corresponding to this process resides.
    #model.network = DDP(model.network, device_ids=None)    # OK?
    #model.network = DDP(model.network, device_ids=[rank])  # 1GPU/1process  if n(>1)-GPU/1process -> slower than 1GPU/1process
    #model.network.to(rank)

    #model.network = DDP(model.network, device_ids=device_ids)       # When specifying device_ids=[rank], error!
    model.network = DDP(model.network.to(device), device_ids=None)   # Both CPU and GPU
    model.network.to(device)  # to(rank)


    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in ['train', 'val']}

    #criterion = set_criterion(args_conf.criterion, args_conf.device)
    criterion = set_criterion(args_conf.criterion, device)             #! Define device depending on process
    optimizer = set_optimizer(args_conf.optimizer, model.network, args_conf.lr)

    if isMaster:
        loss_store = set_loss_store(args_conf.label_list, args_conf.epochs, args_conf.dataset_info)


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

            split_dataloader.sampler.set_epoch(epoch)

            for i, data in enumerate(split_dataloader):
                optimizer.zero_grad()

                #in_data, labels = model.set_data(data)

                in_data, labels = model.set_data(data, device)

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
                #print(f"Before, rank: {rank}, {losses}")
                for label_name in losses.keys():
                    dist.all_reduce(losses[label_name], op=dist.ReduceOp.SUM)
                #print(f"After, rank: {rank}, {losses}")

                # Store
                if isMaster:
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

        #! Wait until master store and save weight
        dist.barrier()

    dist.barrier()  #! No need?
    if isMaster:
        save_parameter(args_save, save_datetime_dir + '/' + 'parameters.json')
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
    #if isdist == 'yes':

    # total number of processes, 1GPU/1Process
    world_size = 4  #len(args_conf.gpu_ids)

    mp.spawn(
            train,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf,
                #args_print,
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

        os.environ['GLOO_SOCKET_IFNAME'] = 'en0'  # macだからen0？、GPUだといらないかも
        os.environ['MASTER_ADDR'] = 'localhost'   #'127.0.0.1'  # 'localhost'
        os.environ['MASTER_PORT'] = '29500'       #'12355' #'8888'

        #num_gpus = 4 #8 #2      # num process . 4 -> rank = 0,1,2,3, 1 -> rank=0
        #world_size = num_gpus   # total number of processes

        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTraining finished.\n')

        print(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    #finally:
    #    dist.destroy_process_group()
