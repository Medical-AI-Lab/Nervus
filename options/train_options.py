#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse



class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Train options')
        
        # Materials
        self.parser.add_argument('--csv_name',  type=str, default=None, help='csv namefilename(Default: None)')
        self.parser.add_argument('--image_dir', type=str, default=None, help='directory name contaning images(Default: None)')

        # Task
        self.parser.add_argument('--task',  type=str, default=None, help='Task: classification or regression (Default: None)')

        # Model
        self.parser.add_argument('--model', type=str, default=None, help='model: MLP, CNN, MLP+CNN (Default: None)')

        # Training and Internal validation
        self.parser.add_argument('--criterion', type=str,   default=None,               help='criterion: CrossEntropyLoss, MSE, RMSE, MAE (Default: None)')
        self.parser.add_argument('--optimizer', type=str,   default=None,               help='optimzer:SGD, Adadelta, Adam, RMSprop (Default: None)')
        self.parser.add_argument('--lr',        type=float, default=0.001, metavar='N', help='learning rate: (Default: 0.001)')
        self.parser.add_argument('--epochs',    type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

        # Batch size
        self.parser.add_argument('--batch_size',      type=int, default=None, metavar='N', help='batch size for training (Default: None)')

        # Preprocess for image
        self.parser.add_argument('--preprocess',             type=str, default=None,  help='preprocess for image, yes or no (Default: None)')
        self.parser.add_argument('--random_horizontal_flip', type=str, default=None,  help='RandomHorizontalFlip, yes or no (Default: None)')
        self.parser.add_argument('--random_rotation',        type=str, default=None,  help='RandomRotation, yes or no (Default: None)')
        self.parser.add_argument('--color_jitter',           type=str, default=None,  help='ColorJitter, yes or no (Default: None)')
        self.parser.add_argument('--random_apply',           type=str, default=None,  help='transform randomly applies, yes or no (Default: None)')
        self.parser.add_argument('--normalize_image',        type=str, default='yes', help='image nomalization, yes no no (Default: yes)')

        # Sampler
        self.parser.add_argument('--sampler', type=str, default=None, help='sample data in traning or not, yes or no (Default: None)')

        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')


    def parse(self):
        self.args = self.parser.parse_args()

        # Get MLP and CNN name when training
        # 'MLP', 'ResNet18', 'MLP+ResNet18' -> ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
        #
        # Add: mlp, cnn, load_input, load_image
        if not(self.args.model is None):
            _model = self.args.model.split('+')
            self.args.cnn = list(filter(lambda model_name: model_name != 'MLP', _model))

            if 'MLP' in _model:
                self.args.mlp = 'MLP'
            else:
                self.args.mlp = None

            # Check validyty of specification of CNN
            if len(self.args.cnn) >= 2:
                print('Cannot identify a single CNN ' + str(self.args.cnn) + '\n')
                exit()
            elif len(self.args.cnn) == 1:
                self.args.cnn = self.args.cnn[0]
            else:
                self.args.cnn = None

        else:
            pass
            #Check the case of when no specifying model later


        # Align gpu_ids
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []

        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)

        return vars(self.args)



    def is_option_valid(self, args:dict):
        # Check must options
        must_base_opts = ['csv_name', 'task', 'model', 'epochs', 'batch_size', 'criterion', 'optimizer', 'sampler']
        if not(args['cnn'] is None):
            must_base_opts = must_base_opts + ['image_dir', 'normalize_image']
        
        for opt in must_base_opts:
            if args[opt] is None:
                print('\nSpecify {}.\n'.format(opt))
                exit()
            else:
                pass
        print('\nOptions have been cheked.\n')



    def print_options(self, opt:dict):
        self.args = self.parser.parse_args()

        ignore = ['mlp', 'cnn', 'load_input', 'load_image']

        message = ''
        message += '-------------------- Options --------------------\n'

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

        message += '------------------- End --------------------------'
        print(message)


# ----- EOF -----