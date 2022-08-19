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

from logger.logger import Logger


logger = Logger.get_logger('test')


class Likelihood:
    def __init__(self, task, num_classes_in_internal_label, test_datetime):
        self.task = task
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.test_datetime = test_datetime
        self.df_likelihood = pd.DataFrame()

    def _convert_to_numpy(self, raw_data):
        converted_data = raw_data.to('cpu').detach().numpy().copy()
        return converted_data

    def _make_pred_names(self, internal_label_name, num_classes):
        pred_names = []
        for i in range(num_classes):
            pred_name_i = 'pred_' + internal_label_name.replace('internal_', '') + '_' + str(i)
            pred_names.append(pred_name_i)
        return pred_names

    def make_likehood(self, data, output):
        """
        Updates DataFrame of likelihood every batch

        Args:
            data (dict): batch data from dataloader
        """

        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'split': data['split']
                            })
        if self.task == 'deepsurv':
            _df_period = pd.DataFrame({
                                'period': self._convert_to_numpy(data['period'])
                                })
            _df_new = pd.concat([_df_new, _df_period], axis=1)

        for internal_label_name, output in output.items():
            # raw_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name]
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # internal_label if deepsurv
            if self.task == 'deepsurv':
                _df_internal_label = pd.DataFrame({
                                            internal_label_name: self._convert_to_numpy(data['internal_labels'][internal_label_name])
                                            })
                _df_new = pd.concat([_df_new, _df_internal_label], axis=1)

            # output
            pred_names = self._make_pred_names(internal_label_name, self.num_classes_in_internal_label[internal_label_name])
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)

    def save_likelihood(self, weight_name=None):
        save_dir = Path('./results/sets', self.test_datetime, 'likelihoods')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'likelihood_' + weight_name + '.csv')
        self.df_likelihood.to_csv(save_path, index=False)


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

    lh = Likelihood(args.task, sp.num_classes_in_internal_label, args.test_datetime)
    for split in ['train', 'val', 'test']:
        split_dataloader = dataloaders[split]

        for i, data in enumerate(split_dataloader):
            model.set_data(data)

            with torch.no_grad():
                model.forward()

            lh.make_likehood(data, model.get_output())

    lh.save_likelihood(weight_name=weight_path.stem)

logger.info('Inference finished.')
