#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import pandas as pd

import torch

from models.options import check_test_options
from models.env import SplitProvider
from models.dataloader import create_dataloader
from models.framework import create_model

from logger.logger import Logger


logger = Logger.get_logger('test')


class BaseLikelihood(ABC):
    def __init__(self, class_name_in_raw_label, test_datetime):
        self.class_name_in_raw_label = class_name_in_raw_label
        self.test_datetime = test_datetime
        self.df_likelihood = pd.DataFrame()

    def _convert_to_numpy(self, raw_data):
        converted_data = raw_data.to('cpu').detach().numpy().copy()
        return converted_data

    @abstractmethod
    def _make_pred_names(self, raw_label_name):
        pass

    @abstractmethod
    def make_likehood(self, data, output):
        """
        Make DataFrame of likelihood every batch

        Args:
            data (dict): batch data from dataloader
            output (dict): output of model
        """
        pass

    def save_likelihood(self, weight_name=None):
        save_dir = Path('./results/sets', self.test_datetime, 'likelihoods')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'likelihood_' + weight_name + '.csv')
        self.df_likelihood.to_csv(save_path, index=False)


class ClsLikelihood(BaseLikelihood):
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)

    def _make_pred_names(self, raw_label_name):
        pred_names = []
        class_names = self.class_name_in_raw_label[raw_label_name]
        for class_name in class_names.keys():
            pred_name = 'pred_' + raw_label_name + '_' + class_name
            pred_names.append(pred_name)
        return pred_names

    def make_likehood(self, data, output):
        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'split': data['split']
                            })

        for internal_label_name, output in output.items():
            # raw_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name]
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # output
            pred_names = self._make_pred_names(raw_label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


class RegLikelihood(BaseLikelihood):
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)

    def _make_pred_names(self, raw_label_name):
        pred_names = []
        pred_names.append('pred_' + raw_label_name)
        return pred_names

    def make_likehood(self, data, output):
        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'split': data['split']
                            })

        for internal_label_name, output in output.items():
            # raw_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name]
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # output
            pred_names = self._make_pred_names(raw_label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


class DeepSurvLikelihood(BaseLikelihood):
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)

    def _make_pred_names(self, raw_label_name):
        pred_names = []
        pred_names.append('pred_' + raw_label_name)
        return pred_names

    def make_likehood(self, data, output):
        _period_list = self._convert_to_numpy(data['period'])
        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'split': data['split'],
                            'period': [int(period) for period in _period_list]
                            })

        for internal_label_name, output in output.items():
            # raw_label
            _internal_label_list = self._convert_to_numpy(data['internal_labels'][internal_label_name])
            internal_label_list = [int(internal_label) for internal_label in _internal_label_list]

            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name],
                                    internal_label_name: internal_label_list
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # output
            pred_names = self._make_pred_names(raw_label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)



def set_likelihood(task, class_name_in_raw_label, test_datetime):
    if task == 'classification':
        return ClsLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'regression':
        return RegLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'deepsurv':
        return DeepSurvLikelihood(class_name_in_raw_label, test_datetime)
    else:
        logger.error(f"Invalid task:{task}.")


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

    lh.save_likelihood(weight_name=weight_path.stem)

logger.info('Inference finished.')
