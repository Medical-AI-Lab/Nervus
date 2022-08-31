#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import models as md
import logger


log = logger.get_logger('test')


log.info('\nTest started.\n')

opt = md.check_test_options()
args = opt.args
sp = md.make_split_provider(args.csv_name, args.task)

dataloaders = {
    'train': md.create_dataloader(args, sp, split='train'),
    'val': md.create_dataloader(args, sp, split='val'),
    'test': md.create_dataloader(args, sp, split='test')
    }

train_total = len(dataloaders['train'].dataset)
val_total = len(dataloaders['val'].dataset)
test_total = len(dataloaders['test'].dataset)
log.info(f"train_data = {train_total}")
log.info(f"  val_data = {val_total}")
log.info(f" test_data = {test_total}")

weight_paths = list(Path('./results/sets', args.test_datetime, 'weights').glob('*'))
weight_paths.sort(key=lambda path: path.stat().st_mtime)


for weight_path in weight_paths:
    log.info(f"Inference with {weight_path.name}.")

    model = md.create_model(args, sp, weight_path=weight_path)
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
