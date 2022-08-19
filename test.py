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

        """
        # eg. separated format
        # label_A, label_B, internal_label_A, internal_label_B, pred_label_A_0,  pred_label_A_1, pred_label_B_0, pred_label_B_1
        # label
        _df_raw_label = pd.DataFrame(data['raw_labels'])
        _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

        if self.task == 'deepsurv':
            _internal_label_dict = dict()
            for internal_label_name, label_value in data['internal_labels'].items():
                _internal_label_dict[internal_label_name] = self._convert_to_numpy(label_value)
            _df_internal_label = pd.DataFrame(_internal_label_dict)
            _df_new = pd.concat([_df_new, _df_internal_label], axis=1)

        # output
        _df_output = pd.DataFrame()
        for internal_label_name, output in output.items():
            pred_names = self._make_pred_names(internal_label_name, self.num_classes_in_internal_label[internal_label_name])
            _df_each = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_output = pd.concat([_df_output, _df_each], axis=1)
        _df_new = pd.concat([_df_new, _df_output], axis=1)
        """

        # eg. merged format
        # label_A,  internal_label_A,　pred_label_A_0,  pred_label_A_1, label_B, internal_label_B, pred_label_B_0, pred_label_B_1
        # data['raw_labels']
        # data['internal_labels']
        # output
        for internal_label_name, output in output.items():
            # raw_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name]
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # internal_label is deepsurv
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


for weight_path in weight_paths:
    logger.info(f"Load {weight_path.name}.")

    model = create_model(args, sp, weight_path=weight_path)
    model.eval()

    lh = Likelihood(args.task, sp.num_classes_in_internal_label, args.test_datetime)
    for split in ['train', 'val', 'test']:
        split_dataloader = dataloaders[split]

        for i, data in enumerate(split_dataloader):
            model.set_data(data)

            with torch.no_grad():
                model.forward()

            lh.make_likehood(data, model.get_output())  # batchごとにlikelihoodを追加していく

    lh.save_likelihood(weight_name=weight_path.stem)

# data = {
#        'Filename': filename,
#        'ExamID': examid,
#        'Institution': institution,
#        'raw_labels': raw_label_dict,
#        # 'internal_labels': internal_label_dict,
#        # 'inputs': inputs_value,
#        # 'image': image,
#        'period': period,
#        'split': split
#        }
