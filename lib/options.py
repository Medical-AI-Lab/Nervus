#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import argparse
import pandas as pd
from .logger import Logger as logger
from typing import Dict, List, Tuple, Union


class Options:
    """
    Class to parse options.
    """
    def __init__(self, isTrain: bool = None) -> None:
        """
        Args:
            isTrain (bool, optional): Variable indicating whether training. True when traning. Defaults to None.
        """
        assert isinstance(isTrain, bool), 'isTrain should be bool.'

        self.parser = argparse.ArgumentParser(description='Options for training or test')

        # The blow is common argument both at training and test.
        self.parser.add_argument('--csvpath',   type=str, required=True, help='path to csv for training or test')

        if isTrain:
            # Task
            self.parser.add_argument('--task',  type=str, required=True, choices=['classification', 'regression', 'deepsurv'], help='Task')

            # Model
            self.parser.add_argument('--model', type=str, required=True, help='model: MLP, CNN, ViT, or MLP+(CNN or ViT)')

            # Training and Internal validation
            self.parser.add_argument('--criterion',       type=str,  required=True, choises=['CEL', 'MSE', 'RMSE', 'MAE', 'NLL'], help='criterion')
            self.parser.add_argument('--optimizer',       type=str,  required=True, choices=['SGD', 'Adadelta', 'RMSprop', 'Adam', 'RAdam'], help='optimzer')
            self.parser.add_argument('--lr',              type=float,               metavar='N', help='learning rate')
            self.parser.add_argument('--epochs',          type=int,  default=10,    metavar='N', help='number of epochs (Default: 10)')

            # Batch size
            self.parser.add_argument('--batch_size',      type=int,  required=True, metavar='N', help='batch size in training')

            # Preprocess for image
            self.parser.add_argument('--augmentation',    type=str,  required=True, choices=['xrayaug', 'trivialaugwide', 'randaug', 'no'], help='kind of augmentation')
            self.parser.add_argument('--normalize_image', type=str,                 choices=['yes', 'no'], default='yes', help='image nomalization: yes, no (Default: yes)')

            # Sampler
            self.parser.add_argument('--sampler',         type=str,  required=True, choices=['yes', 'no'], help='sample data in traning or not, yes or no')

            # Input channel
            self.parser.add_argument('--in_channel',      type=int,  required=True, choices=[1, 3], help='channel of input image')
            self.parser.add_argument('--vit_image_size',  type=int,                 default=0,      help='input image size for ViT. Set 0 if not used ViT (Default: 0)')

            # Weight saving strategy
            self.parser.add_argument('--save_weight',     type=str,  choices=['best', 'each'], default='best', help='Save weight: best, or each(ie. save each time loss decreases when multi-label output) (Default: best)')

            # GPU Ids
            self.parser.add_argument('--gpu_ids',         type=str,  default='-1', help='gpu ids: e.g. 0, 0-1-2, 0-2. Use -1 for CPU (Default: -1)')

        else:
            # Directry of weight at traning
            self.parser.add_argument('--weight_dir',      type=str,  default=None, help='directory of weight to be used when test. If None, the latest one is selected')

            # Test bash size
            self.parser.add_argument('--test_batch_size', type=int,  default=64, metavar='N', help='batch size for test (Default: 64)')

        self.args = self.parser.parse_args()
        setattr(self.args, 'isTrain', isTrain)

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
        assert (model_name is not None), 'Specify model.'
        _model = model_name.split('+')
        mlp = 'MLP' if 'MLP' in _model else None
        _net = [_n for _n in _model if _n != 'MLP']
        net = _net[0] if _net != [] else None
        return mlp, net

    def _parse_gpu_ids(self, gpu_ids: str) -> List[int]:
        """
        Parse comma-separated GPU ids strings to list of integers to list of GPU ids.
        eg. '0-1-2' -> [0, 1, 2], '-1' -> []

        Args:
            gpu_ids (str): GPU Ids

        Returns:
            List[int]: list of GPU ids
        """
        str_ids = gpu_ids.split('-') if gpu_ids != '-1' else ['-1']
        _gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                _gpu_ids.append(id)
        return _gpu_ids

    def _unparse_gpu_ids(self, gpu_ids_list: List[int]) -> str:
        """
        Unpasrse gpu_ids,
        ie, convert list of integers to list of GPU ids to
            strings of GPU ids concatenating with '-'.
            eg. [0, 1, 2] -> '0-1-2', [] -> '-1'

        Args:
            gpu_ids (List[int]): list of GPU ids

        Returns:
            str : strings of GPU ids concatenating '-'
        """
        if gpu_ids_list == []:
            return '-1'
        else:
            _gpu_ids = [str(i) for i in gpu_ids_list]
            _gpu_ids = '-'.join(_gpu_ids)
            return _gpu_ids

    def _get_latest_weightdir(self, base_dir: str) -> str:
        """
        Return the latest path to directory of weight made at training.

        Args:
            dase_dir (str): directory below in which imgs/ and docs/ are included.

        Returns:
            str: the latest path to directory of weight
            eg. 'materials/results/[csv_name]/sets/2022-09-30-15-56-60/weights'

        Note that parameter.json is in the same directory with the diretory of weight.
        """
        _pattern = 'results/*/sets/*/weights'
        _dirs = list(path for path in Path(base_dir).glob(_pattern))
        assert (_dirs != []), f"No weight in below in {base_dir}."
        weight_dir = max(_dirs, key=lambda datename: datename.stat().st_mtime)
        return str(weight_dir)

    def parse(self) -> None:
        """
        Parse options.
        """
        # eg. 'materials/docs/[csv_name].csv' -> 'materials'
        #     'materials/covid/docs/[csv_name].csv' -> 'materials/covid'
        _base_dir = re.findall('(.*)/docs', )[0]
        setattr(self.args, 'base_dir', _base_dir)

        if self.args.isTrain:
            _mlp, _net = self._parse_model(self.args.model)
            setattr(self.args, 'mlp', _mlp)
            setattr(self.args, 'net', _net)

            _gpu_ids = self._parse_gpu_ids(self.args.gpu_ids)
            setattr(self.args, 'gpu_ids', _gpu_ids)
        else:
            if self.args.weight_dir is None:
                _weight_dir = self._get_latest_weight_dir(self.args.dase_dir)
                setattr(self.args, 'weight_dir',  _weight_dir)

    def _get_args(self) -> Dict[str, Union[str, float, int, None]]:
        """
        Return dictionary of option name and its parameter.

        Returns:
            Dict[str, Union[str, float, int, None]]: dictionary of option name and its parameter
        """
        return vars(self.args)

    def print_options(self) -> None:
        """
        Print options.
        """
        no_print = [
                    'mlp',
                    'net',
                    'isTrain'
                    ]
        phase = 'Training' if self.args.isTrain else 'Test'
        message = ''
        message += f"--------------- Options for {phase} --------------------------\n"
        _args = self._get_args()
        for k, v in _args.items():
            if k in no_print:
                pass
            else:
                comment = ''
                default = self.parser.get_default(k)
                if k == 'csv_name':
                    str_v = str(Path(v).name)
                elif k == 'weight_dir':
                    str_v = str(Path(v).parents[4])
                elif k == 'test_datetime':
                    str_v = str(Path(v).name)
                elif k == 'gpu_ids':
                    if v == []:
                        str_v = 'CPU selected'
                    else:
                        str_v = f"{str(v)}  (Primary GPU:{v[0]})"
                else:
                    str_v = str(v)
                comment = f"\t[Default: {default}]" if k != 'gpu_ids' else '\t[Default: CPU]'
                message += '{:>25}: {:<40}{}\n'.format(str(k), str_v, comment)

        message += '------------------------ End -------------------------------\n'
        logger.logger.info(message)

    def save_parameter(self, date_name: str) -> None:
        """
        Save parameters.

        Args:
            date_name (str): diractory name for saving
        """
        _args = self._get_args()
        no_save = ['mlp', 'net', 'isTrain']
        saved_args = dict()
        for option, parameter in _args.items():
            if option in no_save:
                pass
            else:
                if option == 'csv_name':
                    saved_args['csv_name'] = parameter.name
                elif option == 'gpu_ids':
                    saved_args['gpu_ids'] = self._unparse_gpu_ids(parameter) # [] -> '-1', [0, 1, 2] -> '0-1-2'
                elif parameter is not None:
                    saved_args[option] = parameter
                else:
                    saved_args[option] = 'None'

        df_parameter = pd.DataFrame(saved_args.items(), columns=['option', 'parameter'])
        save_dir = Path(self.args.baseset_dir, 'results', saved_args['csv_name'].replace('.csv', ''), 'sets', date_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'parameter.csv')
        df_parameter.to_csv(save_path, index=False)

    #! train時のparameterとの整合整合性を取るのは、framework.py でする。
    def setup_parameter_for_test(self) -> None:
        """
        Set up paramters for test by aligning parameters at trainig.

        Note: self.args.weight_dir is directory not only
            in which weights are store but also parameter.csv at training.
        """
        # self.args.weight_dir = Path('baseset/results/int_cla_multi_output_multi_class/sets/2022-10-04-10-16-27/weights')
        _paramater_path = Path(Path(self.args.weight_dir).parents[0], 'parameter.csv')
        df_args = pd.read_csv(_paramater_path, index_col=0)
        no_need_at_test = [
                            'baseset_dir',
                            'criterion',
                            'optimizer',
                            'lr',
                            'epochs',
                            'batch_size',
                            'save_weight'
                            ]
        df_args = df_args.drop(no_need_at_test)

        # Set up just options required for test
        for option, s_parameter in df_args.iterrows():
            _parameter = s_parameter['parameter']
            if option == 'csv_name':
                if self.args.csv_name is None:
                    # csv_name at the latest training is used,
                    _csv_name = Path(self.args.testset_dir, 'docs', _parameter)
                else:
                    _csv_name = Path(self.args.testset_dir, 'docs', self.args.csv_name)
                assert _csv_name.exists(), f"No such csv: {_csv_name}."
                setattr(self.args, 'csv_name', _csv_name)

            elif option == 'model':
                _mlp, _net = self._parse_model(_parameter)
                setattr(self.args, 'model', _parameter)
                setattr(self.args, 'mlp', _mlp)
                setattr(self.args, 'net', _net)

            elif option == 'task':
                setattr(self.args, 'task', _parameter)

            elif option == 'in_channel':
                setattr(self.args, 'in_channel', int(_parameter))

            elif option == 'vit_image_size':
                setattr(self.args, 'vit_image_size', int(_parameter))

            elif option == 'gpu_ids':
                _train_gpu_ids = self._parse_gpu_ids(_parameter)
                setattr(self.args, 'gpu_ids', _train_gpu_ids)

            elif option == 'normalize_image':
                setattr(self.args, 'normalize_image', _parameter)

            else:
                _parameter
                if _parameter == 'None':
                    setattr(self.args, option, None)
                else:
                    setattr(self.args, option, _parameter)

            # The below should be 'no' when test.
            setattr(self.args, 'augmentation', 'no')
            setattr(self.args, 'sampler', 'no')

        _csv_name = self.args.csv_name.stem
        _datetime = Path(self.args.weight_dir).parents[0].name  # -> '2022-10-04-10-16-27'
        _test_datetime = Path(self.args.testset_dir, 'results', _csv_name, 'sets', _datetime)
        setattr(self.args, 'test_datetime', str(_test_datetime))


def check_train_options() -> Options:
    """
    Parse and print options.

    Returns:
        Options: Object of Options
    """
    opt = Options(isTrain=True)
    opt.parse()
    opt.print_options()
    return opt


def check_test_options() -> Options:
    """
    Parse, set up for test and print options.

    Returns:
        Options: Object of Options
    """
    opt = Options(isTrain=False)
    opt.parse()
    # opt.setup_parameter_for_test()
    opt.print_options()
    return opt
