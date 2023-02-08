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


logger = BaseLogger.get_logger(__name__)


def main(args):
    print_paramater(args.print_params, title='Test')

    model = create_model(args.model_params)
    test_splits = args.conf_params.test_splits
    dataloaders = {split: create_dataloader(args.dataloader_params, split=split) for split in test_splits}

    task = args.conf_params.task
    num_outputs_for_label = args.conf_params.num_outputs_for_label
    save_datetime_dir = args.conf_params.save_datetime_dir
    weight_paths = args.conf_params.weight_paths
    likelihood = set_likelihood(task, num_outputs_for_label, save_datetime_dir)

    for weight_path in weight_paths:
        logger.info(f"Inference ...")

        # weight is orverwritten every time weight is loaded.
        model.load_weight(weight_path)
        model.eval()

        likelihood.init_likelihood()
        for split in test_splits:
            split_dataloader = dataloaders[split]
            for i, data in enumerate(split_dataloader):
                in_data, _ = model.set_data(data)

                with torch.no_grad():
                    output = model(in_data)
                    likelihood.make_likelihood(data, output)

        likelihood.save_likelihood(save_datetime_dir, weight_path.split('/')[-1].replace('.pt', ''))

        if len(weight_paths) > 1:
            model.init_network(args.model_params)


if __name__ == '__main__':
    try:
        logger.info('\nTest started.\n')

        args = set_options(phase='test')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')
