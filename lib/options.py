#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import argparse



class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Project in PyTorch')
        
        # Materials
        self.parser.add_argument('--csv_name',  type=str, default=None, help='csv namefilename(Default: None)')
        self.parser.add_argument('--image_dir', type=str, default=None, help='directory name contaning images(Default: None)')

        # Model
        self.parser.add_argument('--model', type=str, default=None, help='model: MLP, CNN, MLP+CNN (Default: None)')

        # Training and Internal validation
        self.parser.add_argument('--criterion', type=str,   default=None,               help='criterion: CrossEntropyLoss, MSE, RMSE, MAE (Default: None)')
        self.parser.add_argument('--optimizer', type=str,   default=None,               help='optimzer:SGD, Adadelta, Adam, RMSprop (Default: None)')
        self.parser.add_argument('--lr',        type=float, default=0.001, metavar='N', help='learning rate: (Default: 0.001)')
        self.parser.add_argument('--epochs',    type=int,   default=10,    metavar='N', help='number of epochs (Default: 10)')

        # Batch size for each phase
        self.parser.add_argument('--batch_size',      type=int, default=None, metavar='N', help='batch size for training (Default: None)')
        self.parser.add_argument('--val_batch_size',  type=int, default=64,   metavar='N', help='batch size for internal validation (Default: 64)')
        self.parser.add_argument('--test_batch_size', type=int, default=64,   metavar='N', help='batch size for test (Default: 64)')

        # Input normalize
        #self.parser.add_argument('--normalize_input', type=str, default=None, help='input nomalization, yes or no (Default: None)')

        # Preprocess for image
        self.parser.add_argument('--preprocess',             type=str, default=None, help='preprocess for image, yes or no (Default: None)')
        self.parser.add_argument('--random_horizontal_flip', type=str, default=None, help='RandomHorizontalFlip, yes or no (Default: None)')
        self.parser.add_argument('--random_rotation',        type=str, default=None, help='RandomRotation, yes or no (Default: None)')
        self.parser.add_argument('--color_jitter',           type=str, default=None, help='ColorJitter, yes or no (Default: None)')
        self.parser.add_argument('--random_apply',           type=str, default=None, help='transform randomly applies, yes or no (Default: None)')
        self.parser.add_argument('--normalize_image',        type=str, default=None, help='image nomalization, yes no no (Default: None)')

        # Sampler
        self.parser.add_argument('--sampler', type=str, default=None, help='sample data in traning or not, yes or no (Default: None)')

        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU (Default: -1)')

        # External validation
        self.parser.add_argument('--test_datetime', type=str, default=None, help='datetime when training (Default: None)')
        #self.parser.add_argument('--test_weight', type=str, default=None, help='weight for test (Default: None)')

        # Plot ROC or yy-graph
        self.parser.add_argument('--likelihood_datetime', type=str, default=None, help='datetime of likelihodd (Default: None)')
        #self.parser.add_argument('--likelihood', type=str, default=None, help='likelihood plotted for ROC or yy-plot (Default: None)')

        # Visualization
        #self.parser.add_argument('--visualization_weight', type=str, default=None, help='weight for visualization (Default: None)')
        self.parser.add_argument('--visualization_datetime', type=str, default=None, help='datetime for visualization (Default: None)')
        self.parser.add_argument('--visualization_split',  type=str, default='train,val,test', help='split to be visualized: eg. train,val,test, val,test. (Default: train,val,test)')



    def parse(self):
        self.args = self.parser.parse_args()
                                
        # Get MLP and CNN name when training
        # 'MLP', 'ResNet18', 'MLP+ResNet18' -> ['MLP'], ['ResNet18'], ['MLP', 'ResNet18']
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

            # Set load_input
            if not(self.args.mlp is None):
                self.args.load_input = 'yes'
            else:
                self.args.load_input = 'no'

            # Set load_image
            if not(self.args.cnn is None):
                self.args.load_image = 'yes'
            else:
                self.args.load_image = 'no'
        else:
            pass
            #Check the case of when no specyfing model 


        # Align gpu_ids
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []

        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)


        args_dict = vars(self.args)

        return args_dict


    def is_option_valid(self, args:dict):
        # Check must options
        must_base_opts = ['csv_name', 'model', 'epochs', 'batch_size', 'criterion', 'optimizer', 'sampler']
        if args['load_image'] == 'yes':
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

        ignore = ['test_weight', 'likelihood', 'visualization_weight', 'visualization_split']

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