#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from distutils.util import strtobool
from pathlib import Path
import pandas as pd
import json
import torch
from .logger import BaseLogger
from typing import List, Dict, Tuple, Union


logger = BaseLogger.get_logger(__name__)


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

        # CSV
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
            self.args.datetime = datetime

        assert isinstance(isTrain, bool), 'isTrain should be bool.'
        self.args.isTrain = isTrain

    def get_args(self) -> argparse.Namespace:
        return self.args



class CSVParser:
    """
    Class to get information of csv and cast csv.
    """
    def __init__(self, csvpath: str, task:str, isTrain=None) -> None:
        """
        Args:
            csvpath (str): path to csv
            task (str): task
            isTrain (bool): if trainig or not
        """
        self.csvpath = csvpath
        self.task = task

        _df_source = pd.read_csv(self.csvpath)
        _df_excluded = _df_source[_df_source['split'] != 'exclude']

        self.input_list = list(_df_excluded.columns[_df_excluded.columns.str.startswith('input')])
        self.label_list = list(_df_excluded.columns[_df_excluded.columns.str.startswith('label')])
        if task.task == 'deepsurv':
            _period_name_list = list(_df_excluded.columns[_df_excluded.columns.str.startswith('period')])
            assert (len(_period_name_list) == 1), f"Column of period should be one in {self.csvpath} when deepsurv."
            self.period_name = _period_name_list[0]

        self.df_source = self._cast(_df_excluded, self.task)

        # If no columns of group, add it.
        if not('group' in self.df_source.columns):
            self.df_source = self.df_source.assign(group='all')


        if isTrain:
            self.mlp_num_inputs = len(self.input_list)
            self.num_outputs_for_label = self._define_num_outputs_for_label(self.df_source, self.label_list, self.task)


    def _cast(self, df_excluded, task) -> pd.DataFrame:
        """
        Make dictionary of cast depending on task.

        Args:
            df_excluded (pd.DataFrame): Dataframe excluded

        Returns:
            DataFrame: csv excluded and cast depending on task
        """
        if task == 'classification':
            _cast_input = {input_name: float for input_name in self.input_list}
            _cast_label = {label_name: int for label_name in self.label_list}
            _cast_dict = {**_cast_input, **_cast_label}
            _df_excluded_cast = df_excluded.astype(_cast_dict)
            return _df_excluded_cast

        elif task == 'regression':
            _cast_input = {input_name: float for input_name in self.input_list}
            _cast_label = {label_name: float for label_name in self.label_list}
            _cast_dict = {**_cast_input, **_cast_label}
            _df_excluded_cast = df_excluded.astype(_cast_dict)
            return _df_excluded_cast

        elif task == 'deepsurve':
            _cast_input = {input_name: float for input_name in self.input_list}
            _cast_label = {label_name: int for label_name in self.label_list}
            _cast_period = {self.period_name: int}
            _cast_dict = {**_cast_input, **_cast_label, **_cast_period}
            _df_excluded_cast = df_excluded.astype(_cast_dict)
            return _df_excluded_cast

        else:
            raise ValueError(f"Invalid task: {self.task}.")

    def _define_num_outputs_for_label(self, df_source: pd.DataFrame, label_list: List[str], task :str) -> Dict[str, int]:
        """
        Define the number of outputs for each label.

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            label_list (List[str]): label list
            task: str

        Returns:
            Dict[str, int]: dictionary of the number of outputs for each label
            eg.
                classification:       _num_outputs_for_label = {label_A: 2, label_B: 3, ...}
                regression, deepsurv: _num_outputs_for_label = {label_A: 1, label_B: 1, ...}
                deepsurv:             _num_outputs_for_label = {label_A: 1}
        """
        if task == 'classification':
            _num_outputs_for_label = {label_name: df_source[label_name].nunique() for label_name in label_list}
            return _num_outputs_for_label

        elif (task == 'regression') or (task == 'deepsurv'):
            _num_outputs_for_label = {label_name: 1 for label_name in label_list}
            return _num_outputs_for_label

        else:
            raise ValueError(f"Invalid task: {task}.")


def _parse_model(model_name: str) -> Tuple[Union[str, None], Union[str, None]]:
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


def _parse_gpu_ids(gpu_ids: str) -> List[int]:
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


def _get_latest_weight_dir() -> str:
    """
    Return the latest path to directory of weight made at training.

    Returns:
        str: path to directory of the latest weight
        eg. 'results/<project>/trials/2022-09-30-15-56-60/weights'
    """
    _weight_dirs = list(Path('results').glob('*/trials/*/weights'))
    assert (_weight_dirs != []), 'No directory of weight.'
    weight_dir = max(_weight_dirs, key=lambda weight_dir: weight_dir.stat().st_mtime)
    return str(weight_dir)


def _collect_weight(weight_dir: str) -> List[str]:
    """
    Return list of weight paths.

    Args:
        weight_dir (str): path to directory of weights

    Returns:
        List[str]: list of weight paths
    """
    _weight_paths = list(Path(weight_dir).glob('*.pt'))
    assert _weight_paths != [], f"No weight in {weight_dir}."
    _weight_paths.sort(key=lambda path: path.stat().st_mtime)
    __weight_paths = [str(weight_path) for weight_path in _weight_paths]
    return __weight_paths


def save_parameter(args: argparse.Namespace, save_path: str) -> None:
    """
    Save parameters.

    Args:
        args (argparse.Namespace): arguments

        save_path (str): save path for parameters
    """

    """
    no_save = [
                'isTrain',
                'device',  # Need str(self.device) if save
                'df_source',
                'datetime',
                'save_datetime_dir',
                'dataloader_params',
                'model_params',
                'train_conf_params',
                'test_conf_params',
                'mlp',
                'net'
                ]
    """

    save_params = [
                    'project',
                    'csvpath',
                    'task',

                    'model',
                    'pretrained',

                    'criterion',
                    'optimizer',
                    'lr',
                    'epochs',
                    'batch_size',

                    'augmentation',
                    'normalize_image',
                    'sampler',
                    'in_channel',
                    'vit_image_size',

                    'input_list',
                    'label_list',
                    'period_name'
                    'mlp_num_inputs',
                    'num_outputs_for_label',

                    'save_weight_policy',
                    'gpu_ids',
                    ]

    saved = dict()
    for _param, _arg in vars(args).items():
        if _param in save_params:
            saved[_param] = _arg

    save_dir = Path(save_path).parents[0]
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(saved, f, indent=4)


def load_parameter(parameter_path: str) -> Dict[str, Union[str, int, float]]:
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


def print_paramater(args: argparse.Namespace) -> None:
    """
    Print parameters.

    Args:
        args (argparse.Namespace): arugments
    """
    _train_params = [
                    'project',
                    'csvpath',
                    'task',

                    'gpu_ids',
                    'model',
                    'pretrained',

                    'criterion',
                    'optimizer',
                    'lr',
                    'epochs',
                    'batch_size',

                    'normalize_image',
                    'augmentation',
                    'sampler',
                    'in_channel',
                    'vit_image_size',

                    'save_weight_policy',
                    'save_datetime_dir'
                    ]

    _test_params = [
                    'project',
                    'csvpath',
                    'task',

                    'gpu_ids',
                    'model',
                    'weight_dir',

                    'test_batch_size',
                    'test_splits',

                    'normalize_image',
                    'in_channel',
                    'vit_image_size',

                    'scaler_path',
                    'save_datetime_dir'
                    ]

    if args.isTrain:
        phase = 'Training'
        print_params = _train_params
    else:
        phase = 'Test'
        print_params = _test_params

    message = ''
    message += f"{'-'*25} Options for {phase} {'-'*33}\n"

    for _param, _arg in vars(args).items():
        if _param in print_params:
            _str_arg = _arg2str(_param, _arg)
            message += '{:>25}: {:<40}\n'.format(_param, _str_arg)

    message += f"{'-'*30} End {'-'*48}\n"
    logger.info(message)


def _arg2str(param: str, arg: Union[str, int, float, None]) -> str:
        """
        Convert argument to string.

        Args:
            param (str): parameter
            arg (Union[str, int, float, None): argument

        Returns:
            str: strings of argument
        """
        if param == 'lr':
            if arg is None:
                str_arg = 'Default'
            else:
                str_arg = str(param)
            return str_arg
        elif param == 'gpu_ids':
            if arg == []:
                str_arg = 'CPU selected'
            else:
                str_arg = f"{arg}  (Primary GPU:{arg[0]})"
            return str_arg
        else:
            if arg is None:
                str_arg = 'No need'
            else:
                str_arg = str(arg)
        return str_arg


#
# Option -> parse -> add param -> dispatch
#
# args = Options(...).get_args()
#
# Prepare file name when saving in advance
# weight_dir name , weight name
# scaler name
# parameters.json
#if args.mlp is not None:
#    setattr(args, 'scaler_path', str(Path(args.save_datetime_dir, 'scaler.pkl')))
def _train_parse(args) -> None:
    args.project = Path(args.csvpath).stem
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)

    _device = torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids != [] else torch.device('cpu')
    args.device = _device

    _mlp, _net = _parse_model(args.model)
    args.mlp = _mlp
    args.net = _net

    args.pretrained = bool(args.pretrained)   # strtobool('False') = 0 (== False)

    args.save_datetime_dir = str(Path('results', args.project, 'trials', args.datetime))

    csvparser = CSVParser(args.csvpath, args.task, args.isTrain)
    args.df_csv = csvparser.df_csv
    args.input_list = csvparser.input_list
    args.label_list = csvparser.label_list
    args.mlp_num_inputs = csvparser.mlp_num_inputs
    args.num_outputs_for_label = csvparser.num_outputs_for_label
    if args.task == 'deepsurv':
        args.pariod_name = csvparser.period_name

    return args


# Prepare file name when saving in advance
# likelihood_dir name
# likelihood name
def _test_parse(args) -> None:
    args.project = Path(args.csvpath).stem
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)

    _device = torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids != [] else torch.device('cpu')
    args.device = _device

    # eg. args.weight_dir = results/tutorial_src/trials/2023-02-02-14-27-15/weights
    if args.weight_dir is None:
        args.weight_dir = _get_latest_weight_dir()
    args.weight_paths = _collect_weight(args.weight_dir)

    args.test_splits = args.test_splits.split('-')
    #
    # _align_test_splits
    #

    _train_datetime_dir = Path(args.weight_dir).parents[0]
    _train_datetime = _train_datetime_dir.name
    args.save_datetime_dir = str(Path('results', args.project, 'trials', _train_datetime))

    # load parameters
    _parameter_path = str(Path(_train_datetime_dir, 'parameters.json'))
    params = load_parameter(_parameter_path)

    required_params = [
                        'task',
                        'model',
                        #'normalize_image',
                        'in_channel',
                        'vit_image_size',
                        # 'mlp',
                        # 'net',
                        'input_list',  # should be used one at trainig
                        'label_list',  # shoudl be used one at trainig
                        'mlp_num_inputs',
                        'num_outputs_for_label',
                        'period_name'
                        # 'scaler_path'
                        ]

    for _param in required_params:
        if _param in params:
            setattr(args, _param, params[_param])

    # When test, always fix the following
    args.augmentation = 'no'
    args.sampler = 'no'
    args.pretrained = False

    _mlp, _net = _parse_model(args.model)
    args.mlp = _mlp
    args.net = _net

    if args.mlp is not None:
        args.scaler_path = str(Path(_train_datetime_dir, 'scaler.pkl'))

    csvparser = CSVParser(args.csvpath, args.task)
    args.df_csv = csvparser.df_csv
    return args


# Option -> parse -> add param -> dispatch


class ParamGroup:
    """
    Class to register parameter for groups.
    """
    dataloader = [
                'task',
                'isTrain',
                'df_source',
                'label_list',
                'input_list',
                'period_name',
                'batch_size',
                'test_batch_size',
                'mlp',
                'net',
                'scaler_path',
                'in_channel',
                'normalize_image',
                'augmentation',
                'sampler'
                ]

    net = [
            'mlp',
            'net',
            'num_outputs_for_label',
            'mlp_num_inputs',
            'in_channel',
            'vit_image_size',
            'pretrained',
            'gpu_ids'
            ]

    model = \
            net + \
            [
            'task',
            'isTrain',
            'criterion',
            'device',
            'optimizer',
            'lr',
            'label_list',
            ]

    train_conf = [
                'epochs',
                'save_weight_policy',
                'save_datetime_dir'
                ]

    test_conf = [
                'task',
                'weight_dir',
                'weight_paths',
                'num_outputs_for_label',
                'test_splits',
                'save_datetime_dir'
                ]


class ParamSet:
    """
    Class containing parameters for each group.
    """
    @classmethod
    def dispatch_params_by_group(cls, args: argparse.Namespace, group_name: str) -> ParamSet:
        """
        Dispatch parameters depenidng on group.

        Args:
            args (argparse.Namespace): arguments
            group_name (str): group

        Returns:
            ParamStore: class containing parameters for group
        """
        for param_name in getattr(ParamGroup, group_name):
            if hasattr(args, param_name):
                _arg = getattr(args, param_name)
                setattr(cls, param_name, _arg)
        return cls


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
