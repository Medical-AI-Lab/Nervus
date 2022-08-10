#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
import pandas as pd

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchinfo import summary

from .net import create_net
from .criterion import set_criterion
from .optimizer import set_optimizer
from .loss import create_loss_reg

import sys

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('models.framework')


class BaseModel(ABC):
    def __init__(self, args, split_provider):
        self.args = args
        self.sp = split_provider

        self.task = args.task
        self.mlp = self.args.mlp
        self.net = self.args.net
        self.internal_label_list = self.sp.internal_label_list
        self.num_classes_in_internal_label = self.sp.num_classes_in_internal_label
        self.mlp_num_inputs = len(self.sp.input_list)
        self.in_channel = self.args.in_channel
        self.vit_image_size = self.args.vit_image_size
        self.lr = self.args.lr
        self.criterion_name = args.criterion
        self.optimizer_name = args.optimizer

        self.gpu_ids = self.args.gpu_ids
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids else torch.device('cpu')

        self.network = create_net(self.mlp, self.net, self.num_classes_in_internal_label, self.mlp_num_inputs, self.in_channel, self.vit_image_size, self.gpu_ids)
        if self.gpu_ids != []:
            self.network.to(self.device)
            self.network = nn.DataParallel(self.network, device_ids=self.gpu_ids)

        self.criterion = set_criterion(self.criterion_name, self.device)
        self.optimizer = set_optimizer(self.optimizer_name, self.network, self.lr)
        self.loss_reg = create_loss_reg(self.task, self.criterion, self.internal_label_list, self.device)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    @abstractmethod
    def set_data(self, data):
        pass
        # data = {
        #        'Filename': filename,
        #        'ExamID': examid,
        #        'Institution': institution,
        #        'raw_labels': raw_label_dict,
        #        'internal_labels': internal_label_dict,
        #        'inputs': inputs_value,
        #        'image': image,
        #        'period': period,
        #        'split': split
        #        }

    def multi_label_to_device(self, multi_label):
        for internal_label_name, each_data in multi_label.items():
            multi_label[internal_label_name] = each_data.to(self.device)
        return multi_label

    @abstractmethod
    def forward(self):
        pass

    def backward(self):
        self.loss = self.loss_reg.batch_loss['total']
        self.loss.backward()

    def optimize_paramters(self):
        self.optimizer.step()

    @abstractmethod
    def cal_batch_loss(self):
        pass

    def cal_running_loss(self, batch_size=None):
        self.loss_reg.cal_running_loss(batch_size)

    def cal_epoch_loss(self, epoch, phase, dataset_size=None):
        self.loss_reg.cal_epoch_loss(epoch, phase, dataset_size)

    def is_total_val_loss_updated(self):
        _total_epoch_loss = self.loss_reg.epoch_loss['total']
        is_updated = _total_epoch_loss.is_val_loss_updated()
        return is_updated

    def print_epoch_loss(self, num_epochs, epoch):
        self.loss_reg.print_epoch_loss(num_epochs, epoch)


class SaveLoadMixin:
    sets_dir = 'results/sets'
    csv_parameter = 'parameter.csv'
    weight_dir = 'weights'
    weight_name = 'weight'
    learning_curve_dir = 'learning_curves'
    csv_learning_curve = 'learning_curve'
    likelihood_dir = 'likelihoods'
    csv_likelihood = 'likelihood'

    # variables to keep best_weight and best_epoch
    acting_best_weight = None
    acting_best_epoch = None

    def save_parameter(self, date_name):
        saved_args = copy.deepcopy(vars(self.args))

        for option, parameter in saved_args.items():
            if parameter is None:
                saved_args[option] = 'NONE'
        saved_args['test_batch_size'] = 'NO_USED_IN_TRAINING'
        saved_args['test_datetime'] = 'NO_USED_IN_TRAINING'
        # del saved_args['test_batch_size']
        # del saved_args['test_datetime']

        if saved_args['gpu_ids'] == []:
            saved_args['gpu_ids'] = 'CPU'
        else:
            _gpu_ids = [str(i) for i in saved_args['gpu_ids']]
            _gpu_ids = 'GPU-' + '-'.join(_gpu_ids)
            saved_args['gpu_ids'] = _gpu_ids

        df_parameter = pd.DataFrame(saved_args.items(), columns=['option', 'parameter'])
        save_dir = Path(self.sets_dir, date_name)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, self.csv_parameter)
        df_parameter.to_csv(save_path, index=False)

    def load_parameter(self, raw_args, date_name=None):
        pass
        """
        load_path = Path(cls.sets_dir, datetime, cls.csv_parameter)
        df_parameter = pd.read_csv(load_path, index_col=0)
        df_parameter = df_parameter.fillna(np.nan).replace([np.nan], [None])
        parameter = df_parameter.to_dict()['value']

        parameter['augmentation'] = 'no'  # No need of augmentation when inference

        # Cast
        parameter['lr'] = float(parameter['lr'])
        parameter['epochs'] = int(parameter['epochs'])
        parameter['batch_size'] = int(parameter['batch_size'])
        parameter['in_channel'] = int(parameter['input_channel'])
        """
        # raw_args = Options().parse()
        # raw_args と 読み込んだ parameter をマージ
        # 'NONE_IN_TRAINING' -> Noneに置き換える
        # gpu_ids = CPU -> []
        # gpu_ids = GPU-0-1-2-3 -> [0,1,2,3]

    def store_weight(self):
        self.acting_best_epoch = self.loss_reg.epoch_loss['total'].get_best_epoch()
        _network = copy.deepcopy(self.network)
        if hasattr(_network, 'module'):
            # When DataParallel used
            self.acting_best_weight = copy.deepcopy(_network.module.to(torch.device('cpu')).state_dict())
        else:
            self.acting_best_weight = copy.deepcopy(_network.state_dict())

    def save_weight(self, date_name, as_best=None):
        assert isinstance(as_best, bool), 'Argument as_best should be bool.'

        save_dir = Path(self.sets_dir, date_name, self.weight_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = self.weight_name + '_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
        save_path = Path(save_dir, save_name)

        if as_best:
            save_name_as_best = self.weight_name + '_epoch-' + str(self.acting_best_epoch).zfill(3) + '_best' + '.pt'
            save_path_as_best = Path(save_dir, save_name_as_best)
            if save_path.exists():
                # Check if best weight already saved. If so, just rename with '-best'
                save_path.rename(save_path_as_best)
            else:
                torch.save(self.acting_best_weight, save_path_as_best)
        else:
            save_name = self.weight_name + '_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
            torch.save(self.acting_best_weight, save_path)


    def load_weight(self, date_name=None, which_epoch='best'):
        pass
        """
        load_path = Path(cls.sets_dir, datetime, cls.weight_dir, weight_name)
        cls.model.load_state_dict(load_path)
        """

    def save_learning_curve(self, data_name):
        save_dir = Path(self.sets_dir, data_name, self.learning_curve_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        epoch_loss = self.loss_reg.epoch_loss
        for internal_label_name in self.internal_label_list + ['total']:
            each_epoch_loss = epoch_loss[internal_label_name]
            df_each_epoch_loss = pd.DataFrame({
                                                'train_loss': each_epoch_loss.train,
                                                'val_loss': each_epoch_loss.val
                                            })
            label_name = internal_label_name.replace('internal_', '') if internal_label_name.startswith('internal') else 'total'
            best_epoch = str(each_epoch_loss.get_best_epoch()).zfill(3)
            best_val_loss = f"{each_epoch_loss.get_best_val_loss():.4f}"
            save_name = self.csv_learning_curve + '_' + label_name + '_val-best-epoch-' + best_epoch + '_val-best-loss-' + best_val_loss + '.csv'
            save_path = Path(save_dir, save_name)
            df_each_epoch_loss.to_csv(save_path, index=False)


class ModelWidget(BaseModel, SaveLoadMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class MLPModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.inputs)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class CVModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.image)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class FusionModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.inputs, self.image)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class MLPDeepSurv(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])
        self.period = data['period'].float().to(self.device)

    def forward(self):
        self.multi_output = self.network(self.inputs)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.period, self.network)


class CVDeepSurv(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])
        self.period = data['period'].float().to(self.device)

    def forward(self):
        self.multi_output = self.network(self.image)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.period, self.network)


class FusionDeepSurv(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])
        self.period = data['period'].float().to(self.device)

    def forward(self):
        self.multi_output = self.network(self.inputs, self.image)

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.period, self.network)


def create_model(args, split_provider):
    task = args.task
    mlp = args.mlp
    net = args.net

    if (task == 'classification') or (task == 'regression'):
        if (mlp is not None) and (net is None):
            model = MLPModel(args, split_provider)
        elif (mlp is None) and (net is not None):
            model = CVModel(args, split_provider)
        elif (mlp is not None) and (net is not None):
            model = FusionModel(args, split_provider)
        else:
            logger.error(f"Cannot identify model type for {task}.")

    elif task == 'deepsurv':
        if (mlp is not None) and (net is None):
            model = MLPDeepSurv(args, split_provider)
        elif (mlp is None) and (net is not None):
            model = CVDeepSurv(args, split_provider)
        elif (mlp is not None) and (net is not None):
            model = FusionDeepSurv(args, split_provider)
        else:
            logger.error(f"Cannot identify model type for {task}.")

    else:
        logger.error(f"Invalid task: {task}.")

    return model
