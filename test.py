#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import torch.nn as nn
from lib import (
        set_options,
        create_model,
        print_parameter,
        create_dataloader,
        BaseLogger
        )
from lib.component import set_likelihood


logger = BaseLogger.get_logger(__name__)


def main(
        args_model = None,
        args_dataloader = None,
        args_conf = None,
        args_print = None
        ):

    print_parameter(args_print)

    gpu_ids = args_conf.gpu_ids
    test_splits = args_conf.test_splits
    save_datetime_dir = args_conf.save_datetime_dir
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids != [] else torch.device('cpu')

    dataloaders = {split: create_dataloader(args_dataloader, split=split) for split in test_splits}
    model = create_model(args_model)
    likelihood = set_likelihood(args_conf.task, args_conf.num_outputs_for_label)

    for weight_path in args_conf.weight_paths:
        logger.info(f"Inference ...")

        model.network.to(device)
        model.load_weight(weight_path, on_device=device)
        if gpu_ids != []:
            model.network = nn.DataParallel(model.network, device_ids=gpu_ids)

        model.eval()
        for i, split in enumerate(test_splits):
            for j, data in enumerate(dataloaders[split]):
                in_data, _ = model.set_data(data, device)

                with torch.no_grad():
                    outputs = model(in_data)

                # Make a new likelihood every batch
                df_likelihood = likelihood.make_format(data, outputs)

                if i + j == 0:
                    save_dir = Path(save_datetime_dir, 'likelihoods')
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = Path(save_dir, 'likelihood_' + Path(weight_path).stem + '.csv')
                    df_likelihood.to_csv(save_path, index=False)
                else:
                    df_likelihood.to_csv(save_path, mode='a', index=False, header=False)

        # Reset the current weight by initializing network.
        model.init_network()


if __name__ == '__main__':
    try:
        logger.info('\nTest started.\n')

        args = set_options(phase='test')
        main(**args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')
