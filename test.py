#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import pandas as pd
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib import (
        set_options,
        print_parameter,
        create_dataloader,
        create_model,
        is_master,
        set_world_size,
        setup,
        set_device,
        setenv,
        BaseLogger
        )
from lib.component import set_likelihood


logger = BaseLogger.get_logger(__name__)


def test(
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

    test_splits = args_conf.test_splits
    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in test_splits}
    model = create_model(args_model)

    likelihood = set_likelihood(args_conf.task, args_conf.num_outputs_for_label)

    for weight_path in args_conf.weight_paths:
        logger.info(f"Inference ...")

        model.network.to(device)
        model.load_weight(weight_path, on_device=device)
        model.network = DDP(model.network, device_ids=None)
        model.eval()

        # Wait until all processes load weight.
        dist.barrier()

        for i, split in enumerate(test_splits):
            for j, data in enumerate(dataloaders[split]):
                # Sync all processes before starting with a new data
                dist.barrier()

                in_data, _ = model.set_data(data, device)
                with torch.no_grad():
                    outputs = model(in_data)

                # Make a new likelihood every batch
                df_likelihood = likelihood.make_format(data, outputs)

                # Create enough room to store the collected objects.
                df_likelihood_list = [None for _ in range(world_size)]
                dist.all_gather_object(df_likelihood_list, df_likelihood)

                if isMaster:
                    df_likelihood = pd.concat(df_likelihood_list)
                    df_likelihood = df_likelihood.sort_values('uniqID')
                    if i + j == 0:
                        save_dir = Path(args_conf.save_datetime_dir, 'likelihoods')
                        save_dir.mkdir(parents=True, exist_ok=True)
                        save_path = Path(save_dir, 'likelihood_' + Path(weight_path).stem + '.csv')
                        df_likelihood.to_csv(save_path, index=False)
                    else:
                        df_likelihood.to_csv(save_path, mode='a', index=False, header=False)

        # Reset the current weight by initializing network.
        model.init_network()

        # Wait until models of all process is initialized.
        dist.barrier()

    dist.destroy_process_group()


def main(args):
    args_model = args['args_model']
    args_dataloader = args['args_dataloader']
    args_conf = args['args_conf']
    args_print = args['args_print']
    print_parameter(args_print)

    world_size = set_world_size(args_conf.gpu_ids, on_cpu=True)
    mp.spawn(
            test,
            args=(
                world_size,
                args_model,
                args_dataloader,
                args_conf
                ),
            nprocs=world_size,
            join=True
            )


if __name__ == '__main__':
    try:
        logger.info('\nTest started.\n')

        setenv()
        args = set_options(phase='test')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')
