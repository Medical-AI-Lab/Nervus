#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple, Dict
import pandas as pd

import torch

from models.options import check_test_options
from models.env import SplitProvider
from models.dataloader import create_dataloader
from models.framework import create_model
from models.framework import set_likelihood

from logger.logger import Logger


logger = Logger.get_logger('test')

opt = check_test_options()
args = opt.args
sp = SplitProvider(args.csv_name, args.task)

dataloaders = {
    'train': create_dataloader(args, sp, split='train'),
    'val': create_dataloader(args, sp, split='val'),
    'test': create_dataloader(args, sp, split='test')
    }

train_total = len(dataloaders['train'].dataset)
val_total = len(dataloaders['val'].dataset)
test_total = len(dataloaders['test'].dataset)
logger.info(f"train_data = {train_total}")
logger.info(f"  val_data = {val_total}")
logger.info(f" test_data = {test_total}")

weight_paths = list(Path('./results/sets', args.test_datetime, 'weights').glob('*'))
weight_paths.sort(key=lambda path: path.stat().st_mtime)


for weight_path in weight_paths:
    logger.info(f"Inference with {weight_path.name}.")

    model = create_model(args, sp, weight_path=weight_path)
    model.eval()

    lh = set_likelihood(args.task, sp.class_name_in_raw_label, args.test_datetime)
    for split in ['train', 'val', 'test']:
        split_dataloader = dataloaders[split]

        for i, data in enumerate(split_dataloader):
            model.set_data(data)

            with torch.no_grad():
                model.forward()

            lh.make_likehood(data, model.get_output())

    lh.save_likelihood(save_name=weight_path.stem)

logger.info('Inference finished.')
