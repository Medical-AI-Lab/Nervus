#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from distutils.util import strtobool
from pathlib import Path
import re
from typing import List, Tuple, Union


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
            setattr(self.args, 'datetime', datetime)

        assert isinstance(isTrain, bool), 'isTrain should be bool.'
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

    def _get_latest_weight_dir(self) -> str:
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

    def parse(self) -> None:
        """
        Parse options.
        """
        _project = Path(self.args.csvpath).stem
        setattr(self.args, 'project', _project)

        _gpu_ids = self._parse_gpu_ids(self.args.gpu_ids)
        setattr(self.args, 'gpu_ids', _gpu_ids)

        if self.args.isTrain:
            _mlp, _net = self._parse_model(self.args.model)
            setattr(self.args, 'mlp', _mlp)
            setattr(self.args, 'net', _net)

            _pretrained = bool(self.args.pretrained)   # strtobool('False') = 0 (== False)
            setattr(self.args, 'pretrained', _pretrained)
        else:
            if self.args.weight_dir is None:
                _weight_dir = self._get_latest_weight_dir()
                setattr(self.args, 'weight_dir',  _weight_dir)

            setattr(self.args, 'test_splits', self.args.test_splits.split('-'))


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
