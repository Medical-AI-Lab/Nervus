#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import datetime
import argparse
from pathlib import Path
import json
import pandas as pd
from distutils.util import strtobool
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
            self.parser.add_argument('--optimizer', type=str,   default='Adam', choices=['SGD', 'Adadelta', 'RMSprop', 'Adam', 'RAdam'], help='optimizer')
            self.parser.add_argument('--lr',        type=float,                metavar='N', help='learning rate')
            self.parser.add_argument('--epochs',    type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

            # Batch size
            self.parser.add_argument('--batch_size', type=int,  required=True, metavar='N', help='batch size in training')

            # Preprocess for image
            self.parser.add_argument('--augmentation',       type=str,  default='no', choices=['xrayaug', 'trivialaugwide', 'randaug', 'no'], help='kind of augmentation')
            self.parser.add_argument('--normalize_image',    type=str,                choices=['yes', 'no'], default='yes', help='image normalization: yes, no (Default: yes)')

            # Sampler
            self.parser.add_argument('--sampler',            type=str,  required=True, choices=['weighted', 'distributed', 'distweight', 'no'], help='kind of sampler')

            # Image
            self.parser.add_argument('--in_channel',      type=int, required=True, choices=[1, 3],  help='channel of input image')
            self.parser.add_argument('--bit_depth',       type=int, required=True, choices=[8, 16], help='bit depth of input image')
            self.parser.add_argument('--vit_image_size',  type=int, default=0,                      help='input image size for ViT. Set 0 if not used ViT (Default: 0)')

            # Weight saving strategy
            self.parser.add_argument('--save_weight_policy', type=str,  choices=['best', 'each'], default='best',
                                                            help='Save weight policy: best, or each(ie. save each time loss decreases when multi-label output) (Default: best)')

        else:
            # Directory of weight at training
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
    def __init__(self, csvpath: str, task: str, isTrain: bool = None) -> None:
        """
        Args:
            csvpath (str): path to csv
            task (str): task
            isTrain (bool): if training or not
        """
        self.csvpath = csvpath
        self.task = task

        _df_source = pd.read_csv(self.csvpath)
        _df_source = _df_source[_df_source['split'] != 'exclude']

        self.input_list = list(_df_source.columns[_df_source.columns.str.startswith('input')])
        self.label_list = list(_df_source.columns[_df_source.columns.str.startswith('label')])
        if self.task == 'deepsurv':
            _period_name_list = list(_df_source.columns[_df_source.columns.str.startswith('period')])
            assert (len(_period_name_list) == 1), f"Only one column on period should be included in {self.csvpath} when deepsurv."
            self.period_name = _period_name_list[0]

        _df_source = self._cast(_df_source, self.task)

        # If no column of group, add it.
        if 'group' not in _df_source.columns:
            _df_source = _df_source.assign(group='all')

        self.df_source = _df_source

        if isTrain:
            self.mlp_num_inputs = len(self.input_list)
            self.num_outputs_for_label = self._define_num_outputs_for_label(self.df_source, self.label_list, self.task)

    def _cast(self, df_source: pd.DataFrame, task: str) -> pd.DataFrame:
        """
        Make dictionary of cast depending on task.

        Args:
            df_source (pd.DataFrame): excluded DataFrame
            task: (str): task

        Returns:
            DataFrame: csv excluded and cast depending on task
        """
        _cast_input = {input_name: float for input_name in self.input_list}

        if task == 'classification':
            _cast_label = {label_name: int for label_name in self.label_list}
            _casts = {**_cast_input, **_cast_label}
            df_source = df_source.astype(_casts)
            return df_source

        elif task == 'regression':
            _cast_label = {label_name: float for label_name in self.label_list}
            _casts = {**_cast_input, **_cast_label}
            df_source = df_source.astype(_casts)
            return df_source

        elif task == 'deepsurv':
            _cast_label = {label_name: int for label_name in self.label_list}
            _cast_period = {self.period_name: int}
            _casts = {**_cast_input, **_cast_label, **_cast_period}
            df_source = df_source.astype(_casts)
            return df_source

        else:
            raise ValueError(f"Invalid task: {self.task}.")

    def _define_num_outputs_for_label(self, df_source: pd.DataFrame, label_list: List[str], task :str) -> Dict[str, int]:
        """
        Define the number of outputs for each label.

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            label_list (List[str]): list of labels
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
    eg.:
        '0-1-2' -> [0, 1, 2],
        '-1' -> []

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


def _collect_weight_paths(weight_dir: str) -> List[str]:
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
    _weight_paths = [str(weight_path) for weight_path in _weight_paths]
    return _weight_paths


class ParamTable:
    """
    Class to make table to dispatch parameters by group.
    """
    def __init__(self) -> None:
        # groups
        # key is abbreviation, value is group name
        self.groups = {
                        'mo': 'model',
                        'dl': 'dataloader',
                        'trc': 'train_conf',
                        'tsc': 'test_conf',
                        'sa': 'save',
                        'lo': 'load',
                        'trp': 'train_print',
                        'tsp': 'test_print'
                        }

        mo = self.groups['mo']
        dl = self.groups['dl']
        trc = self.groups['trc']
        tsc = self.groups['tsc']
        sa = self.groups['sa']
        lo = self.groups['lo']
        trp = self.groups['trp']
        tsp = self.groups['tsp']

        # The below shows that which group each parameter dispatches to.
        self.dispatch = {
                'datetime': [sa],
                'project': [sa, trp, tsp],
                'csvpath': [sa, trp, tsp],
                'task': [dl, tsc, sa, lo, trp, tsp],
                'isTrain': [dl, trp, tsp],

                'model': [sa, lo, trp, tsp],
                'vit_image_size': [mo, sa, lo, trp, tsp],
                'pretrained': [mo, sa, trp],
                'mlp': [mo, dl],
                'net': [mo, dl],

                'weight_dir': [tsc, tsp],
                'weight_paths': [tsc],

                'criterion': [trc, sa, trp],
                'optimizer': [trc, sa, trp],
                'lr': [trc, sa, trp],
                'epochs': [trc, sa, trp],

                'batch_size': [dl, sa, trp],
                'test_batch_size': [dl, tsp],
                'test_splits': [tsc, tsp],

                'in_channel': [mo, dl, sa, lo, trp, tsp],

                'bit_depth': [dl, sa, lo, trp, tsp],        #! bit_depth

                'normalize_image': [dl, sa, lo, trp, tsp],
                'augmentation': [dl, sa, trp],
                'sampler': [dl, sa, trp],

                'df_source': [dl],
                'label_list': [dl, trc, sa, lo],
                'input_list': [dl, sa, lo],
                'period_name': [dl, sa, lo],
                'mlp_num_inputs': [mo, sa, lo],
                'num_outputs_for_label': [mo, sa, lo, tsc],

                'save_weight_policy': [sa, trp, trc],
                'scaler_path': [dl, tsp],
                'save_datetime_dir': [trc, tsc, trp, tsp],

                'gpu_ids': [dl, trc, tsc, sa, trp, tsp],
                'dataset_info': [sa, trp, tsp]
                }

        self.table = self._make_table()

    def _make_table(self) -> pd.DataFrame:
        """
        Make table to dispatch parameters by group.

        Returns:
            pd.DataFrame: table which shows that which group each parameter belongs to.
        """
        df_table = pd.DataFrame([], index=self.dispatch.keys(), columns=self.groups.values()).fillna('no')
        for param, grps in self.dispatch.items():
            for grp in grps:
                df_table.loc[param, grp] = 'yes'

        df_table = df_table.reset_index()
        df_table = df_table.rename(columns={'index': 'parameter'})
        return df_table

    def get_by_group(self, group_name: str) -> List[str]:
        """
        Return list of parameters which belong to group

        Args:
            group_name (str): group name

        Returns:
            List[str]: list of parameters
        """
        _df_table = self.table
        _param_names = _df_table[_df_table[group_name] == 'yes']['parameter'].tolist()
        return _param_names


Param_Table = ParamTable()


class ParamSet:
    """
    Class to store required parameters for each group.
    """
    pass


def _dispatch_by_group(args: argparse.Namespace, group_name: str) -> ParamSet:
    """
    Dispatch parameters depending on group.

    Args:
        args (argparse.Namespace): arguments
        group_name (str): group

    Returns:
        ParamSet: class containing parameters for group
    """
    _param_names = Param_Table.get_by_group(group_name)
    param_set = ParamSet()
    for param_name in _param_names:
        if hasattr(args, param_name):
            _arg = getattr(args, param_name)
            setattr(param_set, param_name, _arg)
    return param_set


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


def _retrieve_parameter(parameter_path: str) -> Dict[str, Union[str, int, float]]:
    """
    Retrieve only parameters required at test from parameters at training.

    Args:
        parameter_path (str): path to parameter_path

    Returns:
        Dict[str, Union[str, int, float]]: parameters at training
    """
    with open(parameter_path) as f:
        params = json.load(f)

    _required = Param_Table.get_by_group('load')
    params = {p: v for p, v in params.items() if p in _required}
    return params


def print_parameter(params: ParamSet) -> None:
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
    _header = ('-' * _padding) + _header + ('-' * _padding) + '\n'

    _footer = ' End '
    _padding = (LINE_LENGTH - len(_footer) + 1) // 2
    _footer = ('-' * _padding) + _footer + ('-' * _padding) + '\n'

    message = ''
    message += _header

    _params_dict = vars(params)
    del _params_dict['isTrain']
    for _param, _arg in _params_dict.items():
        _str_arg = _arg2str(_param, _arg)
        message += f"{_param:>30}: {_str_arg:<40}\n"

    message += _footer
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
        elif param == 'test_splits':
            str_arg = ', '.join(arg)
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


def _check_if_valid_sampler(sampler: str, gpu_ids: List[int]) -> None:
    """
    Check if sampler is valid for the number of GPU, or
    depending on distributed learning or not.

    Args:
        sampler (str): sampler
        gpu_ids (List[str]): list og GPU ids, where [] means CPU.
    """
    dist_sampler = ['distributed', 'distweight']
    no_dist_sampler = ['weighted', 'no']
    isDistributed = (len(gpu_ids) >= 1)
    if isDistributed:
        assert (sampler in dist_sampler), \
                f"Invalid sampler: {sampler}, Specify distributed or distweight when using GPU."
    else:
        assert (sampler in no_dist_sampler), \
                f"Invalid sampler: {sampler}, No need of sampler for distributed learning when using CPU."


def _check_if_valid_criterion(criterion: str, task: str) -> None:
    """
    Check if criterion is valid for task at training.

    Args:
        criterion (str): criterion
        task (str): task
    """
    valid_criterion = {
        'classification': ['CEL'],
        'regression': ['MSE', 'RMSE', 'MAE'],
        'deepsurv': ['NLL']
    }
    assert criterion in valid_criterion[task], \
            f"Invalid criterion for task: task={task}, criterion={criterion}. Specify any of {valid_criterion[task]}."


def _train_parse(args: argparse.Namespace) -> Dict[str, ParamSet]:
    """
    Parse parameters required at training.

    Args:
        args (argparse.Namespace): arguments

    Returns:
        Dict[str, ParamSet]: parameters dispatched by group
    """
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)

    #print('Now, Distributed learning on CPU is supposed to be OK.\n')
    # Check validity of sampler
    _check_if_valid_sampler(args.sampler, args.gpu_ids)

    # Check validity of criterion
    _check_if_valid_criterion(args.criterion, args.task)

    args.project = Path(args.csvpath).stem
    args.mlp, args.net = _parse_model(args.model)
    args.pretrained = bool(args.pretrained)  # strtobool('False') = 0 (== False)
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

    # Dispatch parameters
    return {
            'args_model': _dispatch_by_group(args, 'model'),
            'args_dataloader': _dispatch_by_group(args, 'dataloader'),
            'args_conf': _dispatch_by_group(args, 'train_conf'),
            'args_print': _dispatch_by_group(args, 'train_print'),
            'args_save': _dispatch_by_group(args, 'save')
            }


def _test_parse(args: argparse.Namespace) -> Dict[str, ParamSet]:
    """
    Parse parameters required at test.

    Args:
        args (argparse.Namespace): arguments

    Returns:
        Dict[str, ParamSet]: parameters dispatched by group
    """
    args.project = Path(args.csvpath).stem
    args.gpu_ids = _parse_gpu_ids(args.gpu_ids)

    # Collect weight paths
    if args.weight_dir is None:
        args.weight_dir = _get_latest_weight_dir()
    args.weight_paths = _collect_weight_paths(args.weight_dir)

    # Get datetime at training
    _train_datetime_dir = Path(args.weight_dir).parents[0]
    _train_datetime = _train_datetime_dir.name

    args.save_datetime_dir = str(Path('results', args.project, 'trials', _train_datetime))

    # Retrieve only parameters required at test
    _parameter_path = str(Path(_train_datetime_dir, 'parameters.json'))
    params = _retrieve_parameter(_parameter_path)
    for _param, _arg in params.items():
        setattr(args, _param, _arg)

    # When test, the followings are always fixed.
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

    # Dispatch parameters
    return {
            'args_model': _dispatch_by_group(args, 'model'),
            'args_dataloader': _dispatch_by_group(args, 'dataloader'),
            'args_conf': _dispatch_by_group(args, 'test_conf'),
            'args_print': _dispatch_by_group(args, 'test_print')
            }


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
        _args = opt.get_args()
        args = _train_parse(_args)
        return args
    else:
        opt = Options(isTrain=False)
        _args = opt.get_args()
        args = _test_parse(_args)
        return args


def set_world_size(gpu_ids: List[int]) -> int:
    """
    Set world_size, ie, total number of processes.

    Args:
        gpu_ids (List[int]): GPU ids
        is_dist_on_cpu (bool): True when distributed learning 1CPU/N-Process (N>1) on CPU.

    Returns:
        int: world_size

    Note:
        1-CPU/1-Process: Even if using CPU, run 1 process, but DistributedDataParallel is not used.
        1-GPU/1-Process
    """
    if gpu_ids == []:
        # When using CPU, world_size is set as 1.
        return 1
    else:
        return len(gpu_ids)


def setenv() -> None:
    """
    Set environment variables.

    Note:
        GLOO_SOCKET_IFNAME is required when using CPU for setting backend='gloo'.
    """

    import platform
    _system = platform.system()

    if _system == 'Darwin':
        os.environ['GLOO_SOCKET_IFNAME'] = 'lo0'
    elif _system == 'Linux':
        os.environ['GLOO_SOCKET_IFNAME'] = 'lo'
    else:
        raise ValueError(f'Not supported system: {_system}')

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'


def get_elapsed_time(
                    start_datetime: datetime.datetime,
                    end_datetime: datetime.datetime
                    ) -> str:
    """
    Return elapsed time.

    Args:
        start_datetime (datetime.datetime): start datetime
        end_datetime (datetime.datetime): end datetime

    Returns:
        str: elapsed time
    """
    _elapsed_datetime = (end_datetime - start_datetime)
    _days = _elapsed_datetime.days
    _hours = _elapsed_datetime.seconds // 3600
    _minutes = (_elapsed_datetime.seconds // 60) % 60
    _seconds = _elapsed_datetime.seconds % 60
    elapsed_datetime_name = f"{_days} days, {_hours:02}:{_minutes:02}:{_seconds:02}"
    return elapsed_datetime_name
