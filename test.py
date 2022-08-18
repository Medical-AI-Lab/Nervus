#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Tuple, Dict
import pandas as pd

import torch

from models.options import Options
from models.env import SplitProvider
from models.dataloader import create_dataloader
from models.framework import create_model

from logger.logger import Logger


logger = Logger.get_logger('test')

opt = Options()
args = opt.check_test_options()

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


def convert_to_numpy(raw_data):
    converted_data = dict()
    for internal_label_name, value in raw_data.items():
        converted_data[internal_label_name] = value.to('cpu').detach().numpy().copy()
    return converted_data


def make_pred_names(internal_label_name, num_classes):
    pred_names = []
    for i in range(num_classes):
        pred_name_i = 'pred_' + internal_label_name.replace('internal_', '') + '_' + str(i)
        pred_names.append(pred_name_i)
    return pred_names


def _make_likehood(multi_output, num_classes_in_internal_label, isDeepSurv=None):
    df_likelihood = pd.DataFrame()
    for internal_label_name, outputs in multi_output.items():
        pred_names = make_pred_names(internal_label_name, num_classes_in_internal_label[internal_label_name])
        _df_each = pd.DataFrame(outputs, columns=pred_names)
        df_likelihood = pd.concat([df_likelihood, _df_each], axis=1)
    return df_likelihood


def make_likehood(data, num_classes_in_internal_label, isDeepSurv=None):
    multi_output = convert_to_numpy(model.multi_output)
    pass


for weight_path in weight_paths:
    logger.info(f"Load {weight_path.name}.")

    model = create_model(args, sp, weight_path=weight_path)
    model.eval()

    df_result = pd.DataFrame()
    for split in ['train', 'val', 'test']:
        split_dataloader = dataloaders[split]

        for i, data in enumerate(split_dataloader):
            model.set_data(data)

            with torch.no_grad():
                model.forward()

            #multi_output = convert_to_numpy(model.multi_output)
            #df_likelihood = make_likehood(multi_output, sp.num_classes_in_internal_label)
            df_likelihood = make_likehood(data, sp.num_classes_in_internal_label)


            if args.task == 'deepsurv':
                internal_labels = convert_to_numpy(data['internal_labels'])


'Filename'
'Institution'
'raw_labels'





        # savelikelihood

# cla, refの時 raw_labelだけいる
# deepsurvの時 raw_label, internale_label いる

# Filename
# Institution
# raw_labels
# internal_label if deepsurv
# period if deepsurv
# split
# pred_(raw_label_name)_0,
# pred_(raw_label_name)_1,
# pred_(raw_label_name)_3,


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
