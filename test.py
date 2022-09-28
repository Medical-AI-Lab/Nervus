#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from lib import (
        check_test_options,
        create_model,
        set_logger
        )
from lib import Logger as logger


def _collect_weight(test_datetime):
    weight_paths = list(Path('./results/sets', test_datetime, 'weights').glob('*'))
    assert weight_paths != [], f"No weight for {test_datetime}."
    weight_paths.sort(key=lambda path: path.stat().st_mtime)
    return weight_paths


def main(opt):
    args = opt.args
    model = create_model(args)
    model.print_dataset_info()

    weight_paths = _collect_weight(args.test_datetime)
    for weight_path in weight_paths:
        logger.logger.info(f"Inference with {weight_path.name}.")

        model.load_weight(weight_path)  # weight is reset every time

        model.inner_execute(args.isTrain, ['train', 'val', 'test'])
        """
        model.eval()

        for split in ['train', 'val', 'test']:
            split_dataloader = model.dataloaders[split]

            for i, data in enumerate(split_dataloader):
                model.set_data(data)

                with torch.no_grad():
                    model.forward()

                model.make_likelihood(data)
        """
        model.save_likelihood(save_name=weight_path.stem)


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nTest started.\n')

    opt = check_test_options()
    main(opt)

    logger.logger.info('\nTest finished.\n')
