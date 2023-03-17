#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from lib import (
        set_options,
        create_model,
        print_parameter,
        create_dataloader,
        BaseLogger
        )
from lib.component import set_likelihood


import datetime

# For distributed
import os
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



logger = BaseLogger.get_logger(__name__)



MASTER = 0


def test(
        rank,
        world_size,
        args_model = None,
        args_dataloader = None,
        args_conf = None
        #args_print = None
        ):

    isMaster = (rank == MASTER)

    # init
    #dist.init_process_group('gloo', rank=rank, world_size=world_size)
    # For GPU, Distributed package doesn't have NCCL on mac
    #dist.init_process_group('nccl', rank=rank, world_size=world_size)
    gpu_ids = args_conf.gpu_ids
    if gpu_ids != []:
        backend = 'nccl'  # For GPU
    else:
        backend = 'gloo'  # For CPU

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"rank: {rank}, init!, backend: {backend}")


    if isMaster:
        #print_parameter(args_print)
        save_datetime_dir = args_conf.save_datetime_dir


    test_splits = args_conf.test_splits


    model = create_model(args_model)
    #model.network = DDP(model.network, device_ids=None)
    #model.network.to('cpu')


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


    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in test_splits}
    likelihood = set_likelihood(args_conf.task, args_conf.num_outputs_for_label)


    for weight_path in args_conf.weight_paths:
        logger.info(f"Inference ...")

        #model.load_weight(weight_path)
        #model.to_gpu(args_conf.gpu_ids)
        #model.eval()

        # framework.py
        # load weight -> DDP
        logger.info(f"rank: {rank}, Load weight: {weight_path}.\n")

        # map_location = #torch.device('cpu')  # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        weight = torch.load(weight_path, map_location=device)
        model.network.load_state_dict(weight)


        print(f"rank: {rank}, to DDP\n")
        model.network = DDP(model.network, device_ids=device_ids)
        model.network.to(device)  # to(rank)
        model.eval()

        #! Should wait until load weight at all processes?
        dist.barrier()

        for i, split in enumerate(test_splits):
            for j, data in enumerate(dataloaders[split]):
                dist.barrier()
                in_data, _ = model.set_data(data, device)

                with torch.no_grad():
                    outputs = model(in_data)

                # Make a new likelihood every batch
                df_likelihood = likelihood.make_format(data, outputs)

                #! Gather df_likelihood from all processes.
                # Create enough room to store the collected objects.
                df_likelihood_list = [None for _ in range(world_size)]
                dist.all_gather_object(df_likelihood_list, df_likelihood)  # df_likelihood_list will be stored in all processes.  Maybe synced here.
                #print(f"rank: {rank}, {df_likelihood_list}\n")

                if isMaster:
                    df_likelihood = pd.concat(df_likelihood_list)
                    df_likelihood = df_likelihood.sort_values('uniqID')
                    #print(f"rank: {rank}, concat: {df_likelihood}\n")

                    if i + j == 0:
                        save_dir = Path(save_datetime_dir, 'likelihoods')
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = Path(save_dir, 'likelihood_' + Path(weight_path).stem + '.csv')
                        df_likelihood.to_csv(save_path, index=False)
                    else:
                        df_likelihood.to_csv(save_path, mode='a', index=False, header=False)
                # Wait until writing out
                dist.barrier()
                print('rank', rank, i, j)

        # Reset the current weight by initializing network.
        print(f"rank: {rank}, init model\n")
        #model.init_network()
        #model.network.to(device)

        model.init_network(device)
        # Wait until models of all process is initialized.
        dist.barrier()

    dist.destroy_process_group()



def main(args):
    args_model = args['args_model']
    args_dataloader = args['args_dataloader']
    args_conf = args['args_conf']
    args_print = args['args_print']

    print_parameter(args_print)

    # total number of processes, 1GPU/1Process
    world_size = 4  #len(args_conf.gpu_ids)

    mp.spawn(
            test,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf
                #args_print
                ),
            nprocs=world_size,
            join=True
            )



if __name__ == '__main__':
    try:
        #logger.info('\nTest started.\n')
        datetime_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger.info(f"\nTest started at {datetime_name}.\n")

        args = set_options(phase='test')
        #main(**args)

        os.environ['GLOO_SOCKET_IFNAME'] = 'en0'
        os.environ['MASTER_ADDR'] = 'localhost'  #'127.0.0.1'  # 'localhost'
        os.environ['MASTER_PORT'] = '29500'      #'12355' #'8888'

        #num_gpus = 4 #2 #3      # num process
        #world_size = num_gpus   # total number of processes

        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')

    #finally:
    #    dist.destroy_process_group()
