#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import lib


def _collect_weight(test_datetime):
    weight_paths = list(Path('./results/sets', test_datetime, 'weights').glob('*'))
    assert weight_paths != [], f"No weight for {test_datetime}."
    weight_paths.sort(key=lambda path: path.stat().st_mtime)
    return weight_paths


def print_dataset_info(dataloaders):
    train_total = len(dataloaders['train'].dataset)
    val_total = len(dataloaders['val'].dataset)
    test_total = len(dataloaders['test'].dataset)
    log.info(f"train_data = {train_total}")
    log.info(f"  val_data = {val_total}")
    log.info(f" test_data = {test_total}")
    log.info('')


def main(opt, log):
    log.info('\nTest started.\n')
    args = opt.args
    sp = lib.make_split_provider(args.csv_name, args.task)

    dataloaders = {
        'train': lib.create_dataloader(args, sp, split='train'),
        'val': lib.create_dataloader(args, sp, split='val'),
        'test': lib.create_dataloader(args, sp, split='test')
        }

    print_dataset_info(dataloaders)

    weight_paths = _collect_weight(args.test_datetime)
    for weight_path in weight_paths:
        log.info(f"Inference with {weight_path.name}.")

        model = lib.create_model(args, sp, weight_path=weight_path)
        model.eval()

        for split in ['train', 'val', 'test']:
            split_dataloader = dataloaders[split]

            for i, data in enumerate(split_dataloader):
                model.set_data(data)

                with torch.no_grad():
                    model.forward()

                model.make_likelihood(data)

        model.save_likelihood(save_name=weight_path.stem)
    log.info('\nTest finished.\n')


if __name__ == '__main__':
    log = lib.get_logger('test')
    opt = lib.check_test_options()
    main(opt, log)
