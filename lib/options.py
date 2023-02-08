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
        """
        Return arguments.

        Returns:
            argparse.Namespace: arguments
        """
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
        if self.task == 'deepsurv':
            _period_name_list = list(_df_excluded.columns[_df_excluded.columns.str.startswith('period')])
            assert (len(_period_name_list) == 1), f"One column of period should be contained in {self.csvpath} when deepsurv."
            self.period_name = _period_name_list[0]

        self.df_source = self._cast(_df_excluded, self.task)

        # If no columns of group, add it.
        if not('group' in self.df_source.columns):
            self.df_source = self.df_source.assign(group='all')

        if isTrain:
            self.mlp_num_inputs = len(self.input_list)
            self.num_outputs_for_label = self._define_num_outputs_for_label(self.df_source, self.label_list, self.task)

    def _cast(self, df_excluded: pd.DataFrame, task: str) -> pd.DataFrame:
        """
        Make dictionary of cast depending on task.

        Args:
            df_excluded (pd.DataFrame): excluded Dataframe
            task: (str): task

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

        elif task == 'deepsurv':
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


def save_parameter(params: ParamSet, save_path: str) -> None:
    """
    Save parameters.

    Args:
        params (ParamSet): parameters

        save_path (str): save path for parameters
    """
    _saved = {_param: _arg for _param, _arg in vars(params).items()}
    save_dir = Path(save_path).parents[0]
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(_saved, f, indent=4)


def _load_parameter(parameter_path: str) -> Dict[str, Union[str, int, float]]:
    """
    Return dictionalry of parameters at training.

    Args:
        parameter_path (str): path to parameter_path

    Returns:
        Dict[str, Union[str, int, float]]: parameters at training
    """
    with open(parameter_path) as f:
        params = json.load(f)
    return params


def print_paramater(params: ParamSet) -> None:
    """
    Print parameters.

    Args:
        params (ParamSet): parameters
    """

    LINE_LENGTH = 82

    if params.isTrain:
        phase = 'Training'
    else:
        phase = 'Test'

    _header = f" Configuration of {phase} "
    _padding = (LINE_LENGTH - len(_header) + 1) // 2  # round up
    header = f"{'-' * _padding}{_header}{'-' * _padding}\n"

    _footer = ' End '
    _padding = (LINE_LENGTH - len(_footer) + 1) // 2
    footer = f"{'-' * _padding}{_footer}{'-' * _padding}\n"

    message = ''
    message += header

    params_dict = vars(params)
    del params_dict['isTrain']
    for _param, _arg in params_dict.items():
        _str_arg = _arg2str(_param, _arg)
        message += '{:>30}: {:<40}\n'.format(_param, _str_arg)

    message += footer
    logger.info(message)


def _arg2str(param: str, arg: Union[str, int, float]) -> str:
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
            return str_arg
        elif param == 'gpu_ids':
            if arg == []:
                str_arg = 'CPU selected'
            else:
                str_arg = f"{arg}  (Primary GPU:{arg[0]})"
            return str_arg
        elif param == 'dataset_info':
            str_arg = ''
            for i, (split, total) in enumerate(arg.items()):
                if i < len(arg) - 1:
                    str_arg += (f"{split}_data={total}, ")
                else:
                    str_arg += (f"{split}_data={total}")
            return str_arg
        else:
            if arg is None:
                str_arg = 'No need'
            else:
                str_arg = str(arg)
        return str_arg


class ParamTable:
    """
    Class to make table to dispatch parameters by group.
    """
    # groups
    groups = {
            'model': 'model',
            'dataloader': 'dataloader',
            'train_conf': 'train_conf',
            'test_conf': 'test_conf',
            'save': 'save',
            'load': 'load',
            'train_print': 'train_print',
            'test_print': 'test_print'
            }

    mo = groups['model']
    dl = groups['dataloader']
    trac = groups['train_conf']
    tesc = groups['test_conf']
    sa = groups['save']
    lo = groups['load']
    trap = groups['train_print']
    tesp = groups['test_print']

    # The below shows that which group each parameter belongs to.
    table = {
            'datetime': [sa],
            'project': [sa, trap, tesp],
            'csvpath': [sa, trap, tesp],
            'task': [mo, dl, tesc, sa, lo, trap, tesp],
            'isTrain': [mo, dl, trap, tesp],

            'model': [sa, lo, trap, tesp],
            'vit_image_size': [mo, sa, lo, trap, tesp],
            'pretrained': [mo, sa, trap],
            'mlp': [mo, dl],
            'net': [mo, dl],

            'weight_dir': [tesc, tesp],
            'weight_paths': [tesc],

            'criterion': [mo, sa, trap],
            'optimizer': [mo, sa, trap],
            'lr': [mo, sa, trap],
            'epochs': [sa, trac, trap],

            'batch_size': [dl, sa, trap],
            'test_batch_size': [dl, tesp],
            'test_splits': [tesc, tesp],

            'in_channel': [mo, dl, sa, lo, trap, tesp],
            'normalize_image': [dl, sa, lo, trap, tesp],
            'augmentation': [dl, sa, trap],
            'sampler': [dl, sa, trap],

            'df_source': [dl],
            'label_list': [mo, dl, sa, lo],
            'input_list': [dl, sa, lo],
            'period_name': [dl, sa, lo],
            'mlp_num_inputs': [mo, sa, lo],
            'num_outputs_for_label': [mo, sa, lo, tesc],

            'save_weight_policy': [sa, trap, trac],
            'scaler_path': [dl, tesp],
            'save_datetime_dir': [trac, tesc, trap, tesp],

            'gpu_ids': [mo, sa, trap, tesp],
            'device': [mo],
            'dataset_info': [sa, trap, tesp]
            }

    @classmethod
    def make_table(cls) -> pd.DataFrame:
        """
        Make table to dispatch parameters by group.

        Returns:
            pd.DataFrame: table which shows that which group each parameter belongs to.
        """
        df_table = pd.DataFrame([], index=cls.table.keys(), columns=cls.groups.keys()).fillna('no')
        for param, grps in cls.table.items():
            for grp in grps:
                df_table.loc[param, grp] = 'yes'

        df_table = df_table.reset_index()
        df_table = df_table.rename(columns={'index': 'parameter'})
        return df_table


PARAM_TABLE = ParamTable.make_table()


class ParamSet:
    """
    Class containing required parameters for each group.
    """
    pass


def _get_param_name_by_group(group_name: str) -> List[str]:
    _df_table = PARAM_TABLE
    _param_names = _df_table[_df_table[group_name] == 'yes']['parameter'].tolist()
    return _param_names


def _dispatch_by_group(args: argparse.Namespace, group_name: str) -> ParamSet:
    """
    Dispatch parameters depenidng on group.

    Args:
        args (argparse.Namespace): arguments
        group_name (str): group

    Returns:
        ParamStore: class containing parameters for group
    """
    _param_names = _get_param_name_by_group(group_name)
    param_set = ParamSet()
    for param_name in _param_names:
        if hasattr(args, param_name):
            _arg = getattr(args, param_name)
            setattr(param_set, param_name, _arg)
    return param_set


def _train_parse(args: argparse.Namespace) -> argparse.Namespace:
    """
    Parse pamaters required at training.

    Args:
        args (argparse.Namespace): arguments

    Returns:
        argparse.Namespace: arguments
    """
    args.project = Path(args.csvpath).stem
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)
    args.device = torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids != [] else torch.device('cpu')
    args.mlp, args.net = _parse_model(args.model)
    args.pretrained = bool(args.pretrained)   # strtobool('False') = 0 (== False)
    args.save_datetime_dir = str(Path('results', args.project, 'trials', args.datetime))

    # Parse csv
    _csvparser = CSVParser(args.csvpath, args.task, args.isTrain)
    args.df_source = _csvparser.df_source
    args.dataset_info = {split: len(args.df_source[args.df_source['split'] == split]) for split in ['train', 'val']}
    args.input_list = _csvparser.input_list
    args.label_list = _csvparser.label_list
    args.mlp_num_inputs = _csvparser.mlp_num_inputs
    args.num_outputs_for_label = _csvparser.num_outputs_for_label
    if args.task == 'deepsurv':
        args.period_name = _csvparser.period_name

    # Dispatch paramaters
    args.model_params = _dispatch_by_group(args, 'model')
    args.dataloader_params = _dispatch_by_group(args, 'dataloader')
    args.conf_params = _dispatch_by_group(args, 'train_conf')
    args.print_params = _dispatch_by_group(args, 'train_print')
    args.save_params = _dispatch_by_group(args, 'save')
    return args


def _test_parse(args: argparse.Namespace) -> argparse.Namespace:
    """
    Parse pamaters required at test.

    Args:
        args (argparse.Namespace): arguments

    Returns:
        argparse.Namespace: arguments
    """
    args.project = Path(args.csvpath).stem
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)
    args.device = torch.device(f"cuda:{args.gpu_ids[0]}") if args.gpu_ids != [] else torch.device('cpu')

    # Collect weight paths
    if args.weight_dir is None:
        args.weight_dir = _get_latest_weight_dir()
    args.weight_paths = _collect_weight(args.weight_dir)

    # Get datetime at training
    _train_datetime_dir = Path(args.weight_dir).parents[0]
    _train_datetime = _train_datetime_dir.name
    args.save_datetime_dir = str(Path('results', args.project, 'trials', _train_datetime))

    # Load parameters
    _parameter_path = str(Path(_train_datetime_dir, 'parameters.json'))
    params = _load_parameter(_parameter_path)

    # Delete parameters which do not need to be passed on at test.
    _required_params = _get_param_name_by_group('load')
    for _param, _arg in params.items():
        if _param in _required_params:
            setattr(args, _param, _arg)

    # When test, always fix the following
    args.augmentation = 'no'
    args.sampler = 'no'
    args.pretrained = False

    args.mlp, args.net = _parse_model(args.model)
    if args.mlp is not None:
        args.scaler_path = str(Path(_train_datetime_dir, 'scaler.pkl'))

    # Parse csv
    _csvparser = CSVParser(args.csvpath, args.task)
    args.df_source = _csvparser.df_source

    # Align test_splits
    args.test_splits = args.test_splits.split('-')
    _splits = args.df_source['split'].unique().tolist()
    if set(_splits) < set(args.test_splits):
        args.test_splits = _splits

    args.dataset_info = {split: len(args.df_source[args.df_source['split'] == split]) for split in args.test_splits}

    # Dispatch paramaters
    args.model_params = _dispatch_by_group(args, 'model')
    args.dataloader_params = _dispatch_by_group(args, 'dataloader')
    args.conf_params = _dispatch_by_group(args, 'test_conf')
    args.print_params = _dispatch_by_group(args, 'test_print')
    return args


def set_options(datetime_name: str = None, phase: str = None) -> argparse.Namespace:
    """
    Parse options for training or test.

    Args:
        datetime_name (str, optional): datetime name. Defaults to None.
        phase (str, optional): train or test. Defaults to None.

    Returns:
        argparse.Namespace: arguments
    """
    if phase == 'train':
        opt = Options(datetime=datetime_name, isTrain=True)
        args = opt.get_args()
        args = _train_parse(args)
        return args
    else:
        opt = Options(isTrain=False)
        args = opt.get_args()
        args = _test_parse(args)
        return args
