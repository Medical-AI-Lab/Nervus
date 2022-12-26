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

    def get_opt(self):
        return self.args


class ParamParser:
    def __init__(self, args):
        self.args = self.parse(args)

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

    def parse(self, args) -> None:
        """
        Parse options.
        """
        _args = args

        _dataset_dir = re.findall('(.*)/docs', args.csvpath)[0]  # should be unique
        setattr(_args, 'dataset_dir', _dataset_dir)

        _gpu_ids = self._parse_gpu_ids(args.gpu_ids)
        setattr(_args, 'gpu_ids', _gpu_ids)

        _csv_name = Path(self.csvpath).stem
        setattr(_args, 'csv_name', _csv_name)

        if args.isTrain:
            _mlp, _net = self._parse_model(args.model)
            setattr(_args, 'mlp', _mlp)
            setattr(_args, 'net', _net)

            _pretrained = bool(args.pretrained)   # strtobool('False') = 0 (== False)
            setattr(_args, 'pretrained', _pretrained)
        else:
            if args.weight_dir is None:
                _weight_dir = self._get_latest_weight_dir(_dataset_dir)
                setattr(_args, 'weight_dir',  _weight_dir)

            setattr(_args, 'test_splits', args.test_splits.split('-'))
        return _args



"""
args = Options(datetime='2022-12-26-16-56-00', isTrain=True)
param_handler = ParamHandler(args)
params = param_handler.devide()
params['model_param']
params['dataloader_param']
...
"""

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


class ParamHandler:
    def __init__(self, args):
        self.args = args

        if self.args.isTrain:
            _save_datetime_dir = str(Path(self.args.dataset_dir, 'results', self.args.csv_name, 'sets', self.args.datetime))
            self.save_datetime_dir = _save_datetime_dir
        else:
            _save_datetime_dir = Path(self.args.weight_dir).parents[0]
            parameter_path = Path(_save_datetime_dir, 'parameters.json')


    def devide(self):
        # trainの時は、振り分けだけ
        # testの時は、parameter.json を読み込んで、マージ
        if self.args.isTrain:
            pass
        else:
            pass



class ParamContainer:
    def __init__(self):
        pass

# args ->
# csvpath -> csv_param
# dataloader_param
# model_param
# train_param
# test_param
# mutual_param




def check_train_options(datetime_name: str) -> Options:
    """
    Parse options for training.

    Args:
        datetime_name (str): date time

    Returns:
        Options: options
    """
    opt = Options(datetime=datetime_name, isTrain=True)
    opt.parse()

    #! Must opt -> params
    return opt


def check_test_options() -> Options:
    """
    Parse options for test.

    Returns:
        Options: options
    """
    opt = Options(isTrain=False)
    opt.parse()
    return opt




# For PaframHandler
class ParamMixin:
    def print_parameter(self) -> None:
        """
        Print parameters
        """
        no_print = [
                    '_dataset_dir',
                    '_csv_name',
                    'mlp',
                    'net',
                    'input_list',
                    'label_list',
                    'period_name',
                    'mlp_num_inputs',
                    'num_outputs_for_label',
                    'dataloaders',
                    'datetime',
                    'device',
                    'isTrain'
                    ]

        phase = 'Training' if self.isTrain else 'Test'
        message = ''
        message += f"{'-'*25} Options for {phase} {'-'*33}\n"

        for _param, _arg in vars(self).items():
            if _param not in no_print:
                _str_arg = self._arg2str(_param, _arg)
                message += '{:>25}: {:<40}\n'.format(_param, _str_arg)
            else:
                pass

        message += f"{'-'*30} End {'-'*48}\n"
        logger.logger.info(message)


    def _arg2str(self, param: str, arg: Union[str, int, float]) -> str:
        """
        Convert argument to string.

        Args:
            param (str): parameter
            arg (Union[str, int, float]): argument

        Returns:
            str: strings of argument
        """
        if param == 'lr':
            if arg is None:
                str_arg = 'Default'
            else:
                str_arg = str(param)
        elif param == 'gpu_ids':
            if arg == []:
                str_arg = 'CPU selected'
            else:
                str_arg = f"{arg}  (Primary GPU:{arg[0]})"
        else:
            if arg is None:
                str_arg = 'No need'
            else:
                str_arg = str(arg)
        return str_arg

    def print_dataset_info(self) -> None:
        """
        Print dataset size for each split.
        """
        for split, dataloader in self.dataloaders.items():
            total = len(dataloader.dataset)
            logger.logger.info(f"{split:>5}_data = {total}")
        logger.logger.info('')

    def save_parameter(self) -> None:
        """
        Save parameters.
        """
        no_save = [
                    '_dataset_dir',
                    '_csv_name',
                    'dataloaders',
                    'device',  # Need str(self.device) when save
                    'isTrain',
                    'datetime',
                    'save_datetime_dir'
                    ]
        saved = dict()
        for _param, _arg in vars(self).items():
            if _param not in no_save:
                saved[_param] = _arg

        # Save scaler
        if hasattr(self.dataloaders['train'].dataset, 'scaler'):
            scaler = self.dataloaders['train'].dataset.scaler
            saved['scaler_path'] = str(Path(self.save_datetime_dir, 'scaler.pkl'))
            with open(saved['scaler_path'], 'wb') as f:
                pickle.dump(scaler, f)

        # Save parameters
        save_dir = Path(self.save_datetime_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(Path(save_dir, 'parameters.json'))
        with open(save_path, 'w') as f:
            json.dump(saved, f, indent=4)

    def load_parameter(self, parameter_path: Path) -> Dict:
        """
        Return dictionalry of parameters at training.

        Args:
            parameter_path (str): path to parameter_path

        Returns:
            Dict: parameters at training
        """
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters




class BaseParam:
    """
    Set up configure for traning or test.
    Integrate args and parameters.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        for _param, _arg in vars(args).items():
            setattr(self, _param, _arg)





class TrainParam(BaseParam):
    """
    Class for setting parameters for training.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

        sp = make_split_provider(self.csvpath, self.task)
        self.input_list = list(sp.df_source.columns[sp.df_source.columns.str.startswith('input')])
        self.label_list = list(sp.df_source.columns[sp.df_source.columns.str.startswith('label')])
        self.mlp_num_inputs = len(self.input_list)
        self.num_outputs_for_label = self._define_num_outputs_for_label(sp.df_source, self.label_list)
        if self.task == 'deepsurv':
            self.period_name = list(sp.df_source.columns[sp.df_source.columns.str.startswith('period')])[0]

        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids != [] else torch.device('cpu')

        # Directory for saveing paramaters, weights, or learning_curve
        _datetime = self.datetime

        # _save_datetime_dir = str(Path(self._dataset_dir, 'results', self._csv_name, 'sets', _datetime))
        # self.save_datetime_dir = _save_datetime_dir

        # Dataloader
        self.dataloaders = {split: create_dataloader(self, sp.df_source, split=split) for split in ['train', 'val']}

    def _define_num_outputs_for_label(self, df_source: pd.DataFrame, label_list: List[str]) -> Dict[str, int]:
        """
        Define the number of outputs for each label.

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            label_list (List[str]): label list

        Returns:
            Dict[str, int]: dictionary of the number of outputs for each label
            eg.
                classification:       _num_outputs_for_label = {label_A: 2, label_B: 3, ...}
                regression, deepsurv: _num_outputs_for_label = {label_A: 1, label_B: 1, ...}
                deepsurv:             _num_outputs_for_label = {label_A: 1}
        """
        if self.task == 'classification':
            _num_outputs_for_label = {label_name: df_source[label_name].nunique() for label_name in label_list}
        elif (self.task == 'regression') or (self.task == 'deepsurv'):
            _num_outputs_for_label = {label_name: 1 for label_name in label_list}
        else:
            raise ValueError(f"Invalid task: {self.task}.")
        return _num_outputs_for_label


class TestParam(BaseParam):
    """
    Class for setting parameters for test.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

        # Load paramaters
        _save_datetime_dir = Path(self.weight_dir).parents[0]
        parameter_path = Path(_save_datetime_dir, 'parameters.json')
        parameters = self.load_parameter(parameter_path)  # Dict
        required_for_test = [
                            'task',
                            'model',
                            'normalize_image',
                            'in_channel',
                            'vit_image_size',
                            'mlp',
                            'net',
                            'input_list',  # should be used one at trainig
                            'label_list',  # shoudl be used one at trainig
                            'mlp_num_inputs',
                            'num_outputs_for_label',
                            'period_name',
                            'scaler_path'
                            ]
        for _param in required_for_test:
            setattr(self, _param, parameters.get(_param))  # If no exists, set None

        # No need the below at test
        self.augmentation = 'no'
        self.sampler = 'no'
        self.pretrained = False

        sp = make_split_provider(self.csvpath, self.task)
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids != [] else torch.device('cpu')

        # Directory for saving ikelihood
        _datetime = _save_datetime_dir.name
        _save_datetime_dir = str(Path(self._dataset_dir, 'results', self._csv_name, 'sets', _datetime))  # csv_name might be for external dataset
        self.save_datetime_dir = _save_datetime_dir

        # Align splits to be test
        _splits_in_df_source = sp.df_source['split'].unique().tolist()
        self.test_splits = self._align_test_splits(self.test_splits, _splits_in_df_source)

        # Dataloader
        # self.dataloaders = {split: create_dataloader(self, sp.df_source, split=split) for split in self.test_splits}

    def _align_test_splits(self, arg_test_splits: List[str], splits_in_df_source: List[str]) -> List[str]:
        """
        Align splits to be test.

        Args:
            arg_test_splits (List[str]): splits specified by args. Default is ['train', 'val', 'test']
            splits_in_df_source (List[str]): splits includinf csv

        Returns:
            List[str]: splits for test

        args_test_splits  = ['train', 'val', 'test'], ['val', 'test'], or ['test']
        splits_in_df_source = ['train', 'val', 'test'], or ['test']
        Smaller set of splits has priority.
        """
        if set(splits_in_df_source) < set(arg_test_splits):
            _test_splits = splits_in_df_source  # maybe when external dataset
        elif set(arg_test_splits) < set(splits_in_df_source):
            _test_splits = arg_test_splits     # ['val', 'test'], or ['test']
        else:
            _test_splits = arg_test_splits
        return _test_splits


def set_params(args: argparse.Namespace) -> Union[TrainParam, TestParam]:
    """
    Set parameters depending on training or test

    Args:
        args (argparse.Namespace): args

    Returns:
        Union[TrainParam, TestParam]: parameters
    """
    if args.isTrain:
        params = TrainParam(args)
    else:
        params = TestParam(args)
    return params




def check_train_options(datetime_name: str) -> Options:
    """
    Parse options for training.

    Args:
        datetime_name (str): date time

    Returns:
        Options: options
    """
    opt = Options(datetime=datetime_name, isTrain=True)
    opt.parse()

    #! Must opt -> params
    return opt


def check_test_options() -> Options:
    """
    Parse options for test.

    Returns:
        Options: options
    """
    opt = Options(isTrain=False)
    opt.parse()
    return opt