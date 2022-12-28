#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from distutils.util import strtobool
from pathlib import Path
import re
from typing import List, Tuple, Union

# Added
from .component import make_split_provider
import json
import pickle
from .logger import Logger as logger
from typing import Dict
import torch
import pandas as pd


class Options:
    """
    Class for options.
    """
    def __init__(self,  datetime: str = None, isTrain: bool = None) -> None:
        """
            Args:
            datetime (str, optional): date time    Args:
            isTrain (bool, optional): Variable indicating whether training or not. Defaults to None.
        """
        self.parser = argparse.ArgumentParser(description='Options for training or test')

        # The blow is common argument both at training and test.
        self.parser.add_argument('--csvpath', type=str, required=True, help='path to csv for training or test')

        # GPU Ids
        self.parser.add_argument('--gpu_ids', type=str, default='cpu', help='gpu ids: e.g. 0, 0-1-2, 0-2. Use cpu for CPU (Default: cpu)')

        if isTrain:
            # Task
            self.parser.add_argument('--task', type=str, required=True, choices=['classification', 'regression', 'deepsurv'], help='Task')

            # Model
            self.parser.add_argument('--model',      type=str, required=True, help='model: MLP, CNN, ViT, or MLP+(CNN or ViT)')
            self.parser.add_argument('--pretrained', type=strtobool, default=False, help='For use of pretrained model(CNN or ViT)')

            # Training and Internal validation
            self.parser.add_argument('--criterion', type=str,   required=True, choices=['CEL', 'MSE', 'RMSE', 'MAE', 'NLL'], help='criterion')
            self.parser.add_argument('--optimizer', type=str,   default='Adam', choices=['SGD', 'Adadelta', 'RMSprop', 'Adam', 'RAdam'], help='optimzer')
            self.parser.add_argument('--lr',        type=float,                metavar='N', help='learning rate')
            self.parser.add_argument('--epochs',    type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

            # Batch size
            self.parser.add_argument('--batch_size', type=int,  required=True, metavar='N', help='batch size in training')

            # Preprocess for image
            self.parser.add_argument('--augmentation',       type=str,  default='no', choices=['xrayaug', 'trivialaugwide', 'randaug', 'no'], help='kind of augmentation')
            self.parser.add_argument('--normalize_image',    type=str,                choices=['yes', 'no'], default='yes', help='image nomalization: yes, no (Default: yes)')

            # Sampler
            self.parser.add_argument('--sampler',            type=str,  default='no',  choices=['yes', 'no'], help='sample data in traning or not, yes or no')

            # Input channel
            self.parser.add_argument('--in_channel',         type=int,  required=True, choices=[1, 3], help='channel of input image')
            self.parser.add_argument('--vit_image_size',     type=int,  default=0,                     help='input image size for ViT. Set 0 if not used ViT (Default: 0)')

            # Weight saving strategy
            self.parser.add_argument('--save_weight_policy', type=str,  choices=['best', 'each'], default='best', help='Save weight policy: best, or each(ie. save each time loss decreases when multi-label output) (Default: best)')

        else:
            # Directry of weight at traning
            self.parser.add_argument('--weight_dir',         type=str,  default=None, help='directory of weight to be used when test. If None, the latest one is selected')

            # Test bash size
            self.parser.add_argument('--test_batch_size',    type=int,  default=1, metavar='N', help='batch size for test (Default: 1)')

            # Splits for test
            self.parser.add_argument('--test_splits',        type=str, default='train-val-test', help='splits for test: e.g. test, val-test, train-val-test. (Default: train-val-test)')


        self.args = self.parser.parse_args()

        if datetime is not None:
            setattr(self.args, 'datetime', datetime)

        assert isinstance(isTrain, bool), 'isTrain should be bool.'
        setattr(self.args, 'isTrain', isTrain)

    def get_args(self):
        return self.args


class ParamParser:
    def __init__(self, args):
        self.args = args

    def _parse_model(self, model_name: str) -> Tuple[Union[str, None], Union[str, None]]:
        """
        Parse model name.

        Args:
            model_name (str): model name (eg. MLP, ResNey18, or MLP+ResNet18)

        Returns:
            Tuple[str, str]: MLP, CNN or Vision Transformer name
            eg. 'MLP', 'ResNet18', 'MLP+ResNet18' ->
                ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
        """
        _model = model_name.split('+')
        mlp = 'MLP' if 'MLP' in _model else None
        _net = [_n for _n in _model if _n != 'MLP']
        net = _net[0] if _net != [] else None
        return mlp, net

    def _parse_gpu_ids(self, gpu_ids: str) -> List[int]:
        """
        Parse GPU ids concatenated with '-' to list of integers of GPU ids.
        eg. '0-1-2' -> [0, 1, 2], '-1' -> []

        Args:
            gpu_ids (str): GPU Ids

        Returns:
            List[int]: list of GPU ids
        """
        if (gpu_ids == 'cpu') or (gpu_ids == 'cpu\r'):
            str_ids = []
        else:
            str_ids = gpu_ids.split('-')
        _gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                _gpu_ids.append(id)
        return _gpu_ids

    def _get_latest_weight_dir(self, dataset_dir) -> str:
        """
        Return the latest path to directory of weight made at training.

        Returns:
            str: path to directory of the latest weight
            eg. 'materials/docs/[csv_name].csv'
                -> 'materials/results/[csv_name]/sets/2022-09-30-15-56-60/weights'
        """
        _weight_dirs = list(Path(dataset_dir, 'results').glob('*/sets/*/weights'))

        assert (_weight_dirs != []), 'No directory of weight.'
        weight_dir = max(_weight_dirs, key=lambda weight_dir: weight_dir.stat().st_mtime)
        return str(weight_dir)

    def parse(self) -> None:
        """
        Parse options.
        """
        _args = self.args

        # dataset_dir
        _dataset_dir = re.findall('(.*)/docs', _args.csvpath)[0]  # should be unique
        setattr(_args, 'dataset_dir', _dataset_dir)

        # csv_name
        _csv_name = Path(_args.csvpath).stem
        setattr(_args, 'csv_name', _csv_name)

        # gpu_ids
        _gpu_ids = self._parse_gpu_ids(_args.gpu_ids)
        setattr(_args, 'gpu_ids', _gpu_ids)

        #device
        _device = torch.device(f"cuda:{_gpu_ids[0]}") if _gpu_ids != [] else torch.device('cpu')
        setattr(_args, 'device', _device)

        if _args.isTrain:
            # mlp, net
            _mlp, _net = self._parse_model(_args.model)
            setattr(_args, 'mlp', _mlp)
            setattr(_args, 'net', _net)

            # pretrained
            _pretrained = bool(_args.pretrained)  # strtobool('False') = 0 (== False)
            setattr(_args, 'pretrained', _pretrained)

            # save_datetime_dir
            _save_datetime_dir = str(Path(_args.dataset_dir, 'results', _args.csv_name, 'sets', _args.datetime))
            setattr(_args, 'save_datetime_dir', _save_datetime_dir)

        else:
            # weight_dir
            if self.args.weight_dir is None:
                _weight_dir = self._get_latest_weight_dir(_dataset_dir)
                setattr(_args, 'weight_dir',  _weight_dir)

            # save_datetime_dir
            _save_datetime_dir = Path(_args.weight_dir).parents[0]
            setattr(_args, 'save_datetime_dir', _save_datetime_dir)

            # test_splits
            setattr(_args, 'test_splits', self.args.test_splits.split('-'))

        return _args





"""
Allocation_table = {
csvpath

input_list
label_list
mlp_num_inputs
num_outputs_for_label
period_name

gpu_ids
task
model

pretrained
criterion
optimizer
lr
epochs
batch_size
augmentation
normalize_image
sampler
in_channel
vit_image_size
save_weight_policy

save_datetime_dir
scaler_path

csvpath
gpu_ids
weight_dir
test_batch_size
test_splits

datetime
}
"""

"""
opt = Options(datetime='2022-12-26-16-56-00', isTrain=True)
args = opt.get_args()
args = ParamParser(args).parse()

param_handler = ParamHandler(args)
params = param_handler.devide()
params['model_param']
params['dataloader_param']
...
"""

"""
param_type = [
csv
dataloader,
model,
train_conf
]

split_provider_param = [
csv_path,
task
]

dataloader_params = [
df_sources,
isTrain,
batch_size, test_bartch_size,
sampelr,
task,
label_list,
input_list,
period_names,
mlp,
net,
augmentation,
normal_image,
in_channel
]

model_param = [
isTrain,
criterion,
optimizer,
lr,
task,
label_list,
device,
gpu_ids,
mlp,
net,
num_outputs_for_label,
mlp_num_inputs,
in_channel,
vit_image_size,
pretrained
]


train_param =[
epochs,
save_datetime_dir
]
"""

# args ->
# csvpath -> sp
# dataloader_param
# model_param
# train_param
# test_param


class ParamContainer:
    def __init__(self):
        self.table = []

param_container = ParamContainer()


class ParamHandler:
    def __init__(self, args):
        self.args = args
        self.sp = make_split_provider(args.csvpath, args.task)
        self.param_pool = self.gather_args()


    def gather_args(self):
        _param_container = ParamContainer()

        for k, v in self.args.items():
            setattr(_param_container, k, v)

        for k, v in self.sp.items():
            setattr(_param_container, k, v)

        return _param_container

    def devide(self):
        # trainの時は、振り分けだけ
        # testの時は、parameter.json を読み込んで、マージして、振り分け
        if self.args.isTrain:
            pass
        else:
            parameter_path = Path(self.args.save_datetime_dir, 'parameters.json')
            pass

        # gpu_idsが決まった後
        # self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids != [] else torch.device('cpu')


def check_train_options(datetime_name):
    opt = Options(datetime=datetime_name, isTrain=True)
    args = opt.get_args()
    args = ParamParser(args).parse()
    params = ParamHandler(args).devide()
    return params


def check_test_options():
    pass


def set_params():
    pass
