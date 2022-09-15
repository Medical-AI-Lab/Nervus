#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from lib import (
        check_test_options,
        make_split_provider,
        create_dataloader,
        print_dataset_info,
        create_model,
        set_logger
        )
from lib.logger import Logger as logger


def _collect_weight(test_datetime):
    weight_paths = list(Path('./results/sets', test_datetime, 'weights').glob('*'))
    assert weight_paths != [], f"No weight for {test_datetime}."
    weight_paths.sort(key=lambda path: path.stat().st_mtime)
    return weight_paths


def main(opt):
    args = opt.args
    sp = make_split_provider(args.csv_name, args.task)

    dataloaders = {
        'train': create_dataloader(args, sp, split='train'),
        'val': create_dataloader(args, sp, split='val'),
        'test': create_dataloader(args, sp, split='test')
        }

    print_dataset_info(dataloaders)

    weight_paths = _collect_weight(args.test_datetime)
    for weight_path in weight_paths:
        logger.logger.info(f"Inference with {weight_path.name}.")

        model = create_model(args, sp, weight_path=weight_path)
        model.eval()

        for split in ['train', 'val', 'test']:
            split_dataloader = dataloaders[split]

            for i, data in enumerate(split_dataloader):
                model.set_data(data)

                with torch.no_grad():
                    model.forward()

                model.make_likelihood(data)

        model.save_likelihood(save_name=weight_path.stem)


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nTest started.\n')
    opt = check_test_options()
    main(opt)
    logger.logger.info('\nTest finished.\n')
