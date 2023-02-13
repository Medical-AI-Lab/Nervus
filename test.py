#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from lib import (
        set_options,
        create_model,
        print_paramater,
        BaseLogger
        )
from lib.component import (
        create_dataloader,
        set_likelihood
        )
from pathlib import Path


logger = BaseLogger.get_logger(__name__)


def main(args):
    print_paramater(args.print_params)

    model = create_model(args.model_params)
    test_splits = args.conf_params.test_splits
    dataloaders = {split: create_dataloader(args.dataloader_params, split=split) for split in test_splits}

    task = args.conf_params.task
    num_outputs_for_label = args.conf_params.num_outputs_for_label
    save_datetime_dir = args.conf_params.save_datetime_dir
    weight_paths = args.conf_params.weight_paths
    likelihood = set_likelihood(task, num_outputs_for_label)

    for weight_path in weight_paths:
        logger.info(f"Inference ...")

        model.load_weight(weight_path)  # weight is orverwritten every time weight is loaded.
        model.eval()

        for i, split in enumerate(test_splits):
            for j, data in enumerate(dataloaders[split]):
                in_data, _ = model.set_data(data)

                with torch.no_grad():
                    output = model(in_data)
                    df_likelihood = likelihood.make_format(data, output)

                if i + j == 0:
                    save_dir = Path(save_datetime_dir, 'likelihoods')
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_name = 'likelihood_' + Path(weight_path).stem + '.csv'
                    save_path = Path(save_dir, save_name)
                    df_likelihood.to_csv(save_path, index=False)
                else:
                    df_likelihood.to_csv(save_path, mode='a', index=False, header=False)

if __name__ == '__main__':
    try:
        logger.info('\nTest started.\n')

        args = set_options(phase='test')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')
