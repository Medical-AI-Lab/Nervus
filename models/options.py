#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import argparse

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('options')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Options for training or test')

        # Materials
        self.parser.add_argument('--csv_name',        type=str,  default=None,   help='csv filename(Default: None)')
        self.parser.add_argument('--image_dir',       type=str,  default=None,   help='directory name contaning images(Default: None)')

        # Task
        self.parser.add_argument('--task',            type=str,  default=None,   help='Task: classification or regression (Default: None)')

        # Model
        self.parser.add_argument('--model',           type=str,  default=None,   help='model: MLP, CNN, MLP+CNN (Default: None)')

        # Training and Internal validation
        self.parser.add_argument('--criterion',       type=str,   default=None,  help='criterion: CEL, MSE, RMSE, MAE, NLL (Default: None)')
        self.parser.add_argument('--optimizer',       type=str,   default=None,  help='optimzer:SGD, Adadelta, RMSprop, Adam, RAdam (Default: None)')
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
        self.parser.add_argument('--input_channel',   type=int,   default=None,  help='channel of input image (Default: None)')

        # Weight saving strategy
        self.parser.add_argument('--save_weight',     type=str,   choices=['best', 'each'], default='best', help='Save weight: best, or each(ie. save each time loss decreases when multi-label output) (Default: None)')

        # GPU
        self.parser.add_argument('--gpu_ids',         type=str,   default='-1',  help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')

        # Test
        self.parser.add_argument('--test_batch_size', type=int,   default=64,    metavar='N', help='batch size for test (Default: 64)')
        self.parser.add_argument('--test_datetime',   type=str,   default=None,  help='datetime when trained(Default: None)')

    def parse(self):
        self.args = self.parser.parse_args()

        # model
        assert (self.args.model is not None), 'Specify model.'
        _model = self.args.model.split('+')  # 'MLP', 'ResNet18', 'MLP+ResNet18' -> ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
        if 'MLP' in _model:
            self.args.mlp = 'MLP'
        else:
            self.args.mlp = None

        _net = [m for m in _model if m != 'MLP']
        if _net != []:
            self.args.net = _net[0]
        else:
            self.args.net = None

        # split path
        self.args.csv_name = Path('./materials/splits', self.args.csv_name)

        # image directory
        if (self.args.net is not None):
            self.args.image_dir = Path('./materials/images', self.args.image_dir)

        # GPU IDs
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

    def _get_args(self):
        return vars(self.args)

    def print_options(self):
        ignore = ['mlp', 'net', 'test_batch_size', 'test_datetime']

        message = ''
        message += '------------------------ Options --------------------------\n'

        for k, v in self._get_args().items():
            if k not in ignore:
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

    def check_options(self):
        self.parse()
        self.print_options()
        return self.args
