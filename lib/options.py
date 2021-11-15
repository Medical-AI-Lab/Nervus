#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse
import torch
from . import static



class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Project in PyTorch')

        # Project
        self.parser.add_argument('--project_name', default=None, help='Project name.')
        self.parser.add_argument('--project_task', default=None, help='classification or regresssion (Default: None).')

        # Model
        self.parser.add_argument('--model', default=None, help='model: MLP, CNN, MLP+CNN (Default: None)')

        # For CNN, MLP+CNN
        self.parser.add_argument('--image_set', default=None, help='image name (Default: None)')
        self.parser.add_argument('--resize_size', type=int, default=None, help='resize size (Default: None)')
        self.parser.add_argument('--crop_size', type=int, default=None, help='crop size (Default: None)')
        self.parser.add_argument('--normalize_image', default=None, help='image nomalization (Default: None)')

        # Batch size for each phase
        self.parser.add_argument('--batch_size', type=int, default=None, metavar='N', help='batch size for training (Default: None)')
        self.parser.add_argument('--val_batch_size', type=int, default=64, metavar='N', help='batch size for internal validation (Default: 64)')
        self.parser.add_argument('--test_batch_size', type=int, default=64, metavar='N', help='batch size for test (Default: 64)')

        # Sampler
        self.parser.add_argument('--sampler', default=None, help='yes or no: sample data in traning or not (Default: None)')

        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')

        # Training and Internal validation
        self.parser.add_argument('--criterion', default=None, help='criterion: CrossEntropyLoss, MSE, RMSE, MAE (Default: None)')
        self.parser.add_argument('--optimizer', default=None, help='optimzer:SGD, Adadelta, Adam, RMSprop (Default: None)')
        self.parser.add_argument('--lr', type=float, default=0.001, metavar='N', help='learning rate: (Default: 0.001)')
        self.parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs (Default: 10)')

        # External validation
        self.parser.add_argument('--test_weight', type=str, default=None, help='weight for test (Default: None)')

        # Plot ROC or yy-graph
        self.parser.add_argument('--likelihood', type=str, default=None, help='likelihood plotted in ROC or yy-plot (Default: None)')

        # Visualization
        self.parser.add_argument('--visualization_weight', type=str, default=None, help='weight for visualization (Default: None)')
        self.parser.add_argument('--visualization_split', type=str, default=None, help='split to be visualized: eg. train,val,test, val,test. (Default: None)')


    def parse(self):
        self.args = self.parser.parse_args()

        # Add project name and task to args
        self.args.project_name = static.project_name
        self.args.project_task = static.project_task

        # Get MLP and CNN name only when training
        # 'MLP', 'ResNet18', 'MLP+ResNet18' -> ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
        if not(self.args.model is None):
            self.args.model = self.args.model.split('+')
            self.args.cnn = list(filter(lambda model_name: model_name != 'MLP', self.args.model))

            if 'MLP' in self.args.model:
                self.args.mlp = 'MLP'
            else:
                self.args.mlp = None

            # Check validyty of specification of CNN
            if len(self.args.cnn) >= 2:
                print('Specify a single CNN ' + str(self.args.cnn))
                exit()
            elif len(self.args.cnn) == 1:
                self.args.cnn = self.args.cnn[0]
            else:
                self.args.cnn = None

            # Set load_image when used CNN
            if (self.args.cnn is None):
                self.args.load_image = 'no'
            else:
                self.args.load_image = 'yes'

            # Revert
            self.args.model = '+'.join(self.args.model)

        else:
            pass


        # Align gpu_ids
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []

        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        return vars(self.args)



    def is_option_valid(self, opt:dict):
        self.args = self.parser.parse_args()
        response = False

        # Check if must options is None.
        must_opts = ['project_task', 'model', 'image_set', 'epochs', 'batch_size', 'criterion', 'optimizer', 'sampler', 'normalize_image']
        for op in must_opts:
            if opt[op] is None:
                print('Specify {}.'.format(op))
                exit()
            else:
                pass


        # Check validity of options.
        project_task = opt['project_task']
        criterion = opt['criterion']

        if (project_task == 'classification') or (project_task == 'multi_classification'):
            if criterion != 'CrossEntropyLoss':
                print('Project task is classification.')
                print('Specify criterion: CrossEntropyLoss!')
                response = False
            else:
                response = True

        elif (project_task == 'regression') or (project_task == 'multi_regression'):
            if criterion not in ['MSE', 'RMSE', 'MAE']:
                print('Project task is regression.')
                print('Specify criterion: MSE, RMSE, or MAE!')
                response = False
            else:
                response = True

        else:
            print('Wrong projection task: {}'.format(project_task))
            exit()

        return response



    def print_options(self, opt:dict):
        self.args = self.parser.parse_args()

        ignore = ['test_weight', 'likelihood', 'visualization_weight']

        message = ''
        message += '----------------- Options ---------------\n'

        for k, v in (opt.items()):
            if k not in ignore:
                comment = ''
                default = self.parser.get_default(k)

                str_default = str(default) if str(default) != '' else 'None'
                str_v = str(v) if str(v) != '' else 'Not specified'

                if k == 'gpu_ids':
                    if str_v == '[]':
                        str_v = 'CPU selected'
                    else:
                        str_v = str_v + ' (Primary GPU:{})'.format(v[0])

                comment = ('\t[Default: %s]' % str_default) if k != 'gpu_ids' else '\t[Default: CPU]'
                message += '{:>25}: {:<30}{}\n'.format(str(k), str_v, comment)

            else:
                pass

        message += '----------------- End -------------------'
        print(message)



        # Save options
        #os.makedirs(static.expr_dir, exist_ok=True)


# ----- EOF -----
