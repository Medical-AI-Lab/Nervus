#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import re
from pathlib import Path
import argparse
import pandas as pd

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('models.options')


class Options:
    def __init__(self, isTrain=None):
        assert isinstance(isTrain, bool), 'isTrain should be bool.'

        self.parser = argparse.ArgumentParser(description='Options for training or test')

        if isTrain:
            # Materials
            self.parser.add_argument('--csv_name',        type=str,   default=None,   help='csv filename(Default: None)')
            self.parser.add_argument('--image_dir',       type=str,   default=None,   help='directory name contaning images(Default: None)')

            # Task
            self.parser.add_argument('--task',            type=str,   choices=['classification', 'regression', 'deepsurv'], default=None, help='Task: classification or regression (Default: None)')

            # Model
            self.parser.add_argument('--model',           type=str,   default=None,   help='model: MLP, CNN, MLP+CNN (Default: None)')

            # Training and Internal validation
            self.parser.add_argument('--criterion',       type=str,   default=None,  help='criterion: CEL, MSE, RMSE, MAE, NLL (Default: None)')
            self.parser.add_argument('--optimizer',       type=str,   default=None,  help='optimzer: SGD, Adadelta, RMSprop, Adam, RAdam (Default: None)')
            self.parser.add_argument('--lr',              type=float, default=0.001, metavar='N', help='learning rate: (Default: 0.001)')
            self.parser.add_argument('--epochs',          type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

            # Batch size
            self.parser.add_argument('--batch_size',      type=int,   default=None,  metavar='N', help='batch size in training (Default: None)')

            # Preprocess for image
            self.parser.add_argument('--augmentation',    type=str,   default=None,  help='Automatic Augmentation: randaug, trivialaugwide, augmix, no')
            self.parser.add_argument('--normalize_image', type=str,   default='yes', help='image nomalization: yes, no (Default: yes)')

            # Sampler
            self.parser.add_argument('--sampler',         type=str,   default=None,  help='sample data in traning or not, yes or no (Default: None)')

            # Input channel
            self.parser.add_argument('--in_channel',      type=int,   default=None,  help='channel of input image (Default: None)')
            self.parser.add_argument('--vit_image_size',  type=int,   default=None,  help='input image size for ViT(Default: None)')

            # Weight saving strategy
            self.parser.add_argument('--save_weight',     type=str,   choices=['best', 'each'], default='best', help='Save weight: best, or each(ie. save each time loss decreases when multi-label output) (Default: best)')

            # GPU
            self.parser.add_argument('--gpu_ids',         type=str,   default='-1',  help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')

        else:
            # Test
            self.parser.add_argument('--test_datetime',   type=str,   default=None,  help='date time when trained(Default: None)')
            self.parser.add_argument('--test_batch_size', type=int,   default=64,    metavar='N', help='batch size for test (Default: 64)')

        self.args = self.parser.parse_args()
        self.args.isTrain = isTrain

    def _parse_model(self, model_name):
        assert (model_name is not None), 'Specify model.'
        _model = model_name.split('+')  # 'MLP', 'ResNet18', 'MLP+ResNet18' -> ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
        mlp = 'MLP' if 'MLP' in _model else None
        _net = [_n for _n in _model if _n != 'MLP']
        net = _net[0] if _net != [] else None
        return mlp, net

    def _parse_gpu_ids(self, gpu_ids):
        str_ids = gpu_ids.split(',')
        _gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                _gpu_ids.append(id)
        return _gpu_ids

    def _get_latest_test_datetime(self):
        date_names = [path for path in Path('./results/sets/').glob('*') if re.search(r'\d+', str(path))]
        latest = max(date_names, key=lambda date_name: date_name.stat().st_mtime).name
        return latest

    def parse(self):
        if self.args.isTrain:
            # model
            mlp, net = self._parse_model(self.args.model)
            self.args.mlp = mlp
            self.args.net = net

            # split path
            assert (self.args.csv_name is not None), 'Specify csv_name.'
            self.args.csv_name = Path('./materials/splits', self.args.csv_name)

            # image directory
            if self.args.image_dir is not None:
                self.args.image_dir = Path('./materials/images', self.args.image_dir)

            # GPU IDs
            self.args.gpu_ids = self._parse_gpu_ids(self.args.gpu_ids)

        else:
            if self.args.test_datetime is None:
                self.args.test_datetime = self._get_latest_test_datetime()

    def _get_args(self):
        return vars(self.args)

    def print_options(self):
        phase = 'Training' if self.args.isTrain else 'Test'

        message = ''
        message += f"--------------- Options for {phase} --------------------------\n"

        ignored = ['isTrain', 'mlp', 'net']
        for k, v in self._get_args().items():
            if k not in ignored:
                comment = ''
                default = self.parser.get_default(k)

                str_default = str(default) if str(default) != '' else 'None'
                str_v = str(v) if str(v) != '' else 'Not specified'

                if k == 'csv_name':
                    str_v = Path(v).name
                elif k == 'image_dir':
                    str_v = Path(v).name
                elif k == 'gpu_ids':
                    if str_v == '[]':
                        str_v = 'CPU selected'
                    else:
                        str_v = f"{str_v}  (Primary GPU:{v[0]})"

                comment = f"\t[Default: {str_default}]" if k != 'gpu_ids' else '\t[Default: CPU]'
                message += '{:>25}: {:<40}{}\n'.format(str(k), str_v, comment)
            else:
                pass
        message += '------------------------ End -------------------------------'
        logger.info(message)

    def save_parameter(self, date_name):
        saved_args = self._get_args()

        ignored = ['isTrain']
        for ignore in ignored:
            del saved_args[ignore]

        for option, parameter in saved_args.items():
            if option == 'gpu_ids':
                if parameter == []:
                    saved_args['gpu_ids'] = 'CPU'
                else:
                    _gpu_ids = [str(i) for i in saved_args['gpu_ids']]
                    _gpu_ids = '-'.join(_gpu_ids)   # ['0', '1', '2'] -> 0-1-2
                    saved_args['gpu_ids'] = _gpu_ids
            else:
                if parameter is None:
                    saved_args[option] = 'None'

        df_parameter = pd.DataFrame(saved_args.items(), columns=['option', 'parameter'])
        save_dir = Path('./results/sets', date_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'parameter.csv')
        df_parameter.to_csv(save_path, index=False)

    def setup_parameter_for_test(self):
        parameter_path = Path('./results/sets', self.args.test_datetime, 'parameter.csv')
        df_args = pd.read_csv(parameter_path)

        ignored = ['criterion', 'optimizer', 'lr', 'epochs', 'batch_size', 'save_weight']  # no need when test
        # DataFrame -> Dict except ignored
        args_dict = dict()
        for each in df_args.to_dict(orient='records'):  # eg. each = {'option': 'model', 'parameter': 'ResNet18'}
            if each['option'] not in ignored:
                args_dict[each['option']] = each['parameter']

        for option, parameter in args_dict.items():
            if option == 'in_channel':
                setattr(self.args, 'in_channel', int(parameter))

            elif option == 'vit_image_size':
                if parameter.isnumeric():
                    setattr(self.args, option, int(parameter))
                else:
                    setattr(self.args, option, None)

            elif option == 'gpu_ids':
                if parameter == 'CPU':
                    setattr(self.args, option, [])
                else:
                    _gpu_ids = [int(i) for i in parameter.split('-')]  # 0-1-2 -> [0, 1, 2]
                    setattr(self.args, option, _gpu_ids)

            else:
                if parameter == 'None':
                    setattr(self.args, option, None)
                else:
                    setattr(self.args, option, parameter)

        # The below should be 'no' when test.
        setattr(self.args, 'augmentation', 'no')
        setattr(self.args, 'sampler', 'no')


def check_train_options():
    opt = Options(isTrain=True)
    opt.parse()
    opt.print_options()
    return opt


def check_test_options():
    opt = Options(isTrain=False)
    opt.parse()
    opt.setup_parameter_for_test()
    opt.print_options()
    return opt
