#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from pathlib import Path
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torchinfo import summary

from .net import BaseNet
from .criterion import Criterion
from .optimizer import Optimizer
from .loss import LossRegistory

import sys
from pathlib import Path

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('models.framework')


class MultiMixin:
    def multi_forward(self, out_features):
        output = dict()
        for internal_label_name, classifier in self.multi_classifier.items():
            output[internal_label_name] = classifier(out_features)
        return output


class MultiWidget(nn.Module, BaseNet, MultiMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class MultiNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channel=None, vit_image_size=None):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channel = in_channel
        self.vit_image_size = vit_image_size

        self.extractor = self.constuct_extractor(self.net_name, mlp_num_inputs=self.mlp_num_inputs, in_channel=self.in_channel, vit_image_size=self.vit_image_size)
        self.multi_classifier = self.construct_multi_classifier(self.net_name, self.num_classes_in_internal_label)

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiNetFusion(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channel=None, vit_image_size=None):
        assert (net_name != 'MLP'), 'net_name should not be MLP.'

        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channel = in_channel
        self.vit_image_size = vit_image_size

        # Extractor of MLP and Net
        self.extractor_mlp = self.constuct_extractor('MLP', mlp_num_inputs=self.mlp_num_inputs)
        self.extractor_net = self.constuct_extractor(self.net_name, in_channel=self.in_channel, vit_image_size=self.vit_image_size)
        self.aux_module = self.construct_aux_module(self.net_name)

        # Intermediate MLP
        self.in_featues_from_mlp = self.get_classifier_in_features('MLP')
        self.in_features_from_net = self.get_classifier_in_features(self.net_name)
        self.inter_mlp_in_feature = self.in_featues_from_mlp + self.in_features_from_net
        self.inter_mlp = self.MLPNet(self.inter_mlp_in_feature, inplace=False)  # ! If inplace==True, cannot backweard  Check!

        # Multi classifier
        self.multi_classifier = self.construct_multi_classifier('MLP', num_classes_in_internal_label)

    def forward(self, x_mlp, x_net):
        out_mlp = self.extractor_mlp(x_mlp)
        out_net = self.extractor_net(x_net)
        out_net = self.aux_module(out_net)

        out_features = torch.cat([out_mlp, out_net], dim=1)
        out_features = self.inter_mlp(out_features)
        output = self.multi_forward(out_features)
        return output


def create_net(mlp, net, num_classes_in_internal_label, mlp_num_inputs, in_channel, vit_image_size, gpu_ids):
    if (mlp is not None) and (net is None):
        multi_net = MultiNet('MLP', num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channel=in_channel, vit_image_size=vit_image_size)
    elif (mlp is None) and (net is not None):
        multi_net = MultiNet(net, num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channel=in_channel, vit_image_size=vit_image_size)
    elif (mlp is not None) and (net is not None):
        multi_net = MultiNetFusion(net, num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channel=in_channel, vit_image_size=vit_image_size)
    else:
        logger.error('Cannot identify net type.')
    # DataParallel
    return multi_net


def set_criterion(criterion, device):
    criterion = Criterion.set_criterion(criterion, device)
    return criterion


def set_optimizer(optimizer, network, lr):
    optimizer = Optimizer.set_optimizer(optimizer, network, lr)
    return optimizer


def create_loss_reg(task, criterion, internal_label_list, device):
    loss_reg = LossRegistory(task, criterion, internal_label_list, device)
    return loss_reg


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
        #        'split': split
        #        }

    def multi_label_to_device(self, multi_label):
        for internal_label_name, each_data in multi_label.items():
            multi_label[internal_label_name] = each_data.to(self.device)
        return multi_label

    @abstractmethod
    def forward(self):
        pass

    def cal_batch_loss(self):
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)

    def cal_running_loss(self, batch_size):
        self.loss_reg.cal_running_loss(batch_size)

    def cal_epoch_loss(self, phase, dataset_size):
        self.loss_reg.cal_epoch_loss(phase, dataset_size)

    def print_epoch_loss(self, num_epochs, epoch):
        self.loss_reg.print_epoch_loss(num_epochs, epoch)

    def backward(self):
        self.loss = self.loss_reg.batch_loss['total']
        self.loss.backward()

    def optimize_paramters(self):
        self.optimizer.step()


class ModelMixin:
    sets_dir = 'results/sets'
    csv_parameter = 'parameter.csv'

    weight_dir = 'weights'
    weight_name = 'weight'

    learning_curve_dir = 'learning_curves'
    csv_learning_curve = 'learning_curve'

    likelihood_dir = 'likelihoods'
    csv_likelihood = 'likelihood'

    @classmethod
    def save_parameter(cls, args, datetime):
        pass
        """
        df_parameter = pd.DataFrame(list(args.items()), columns=['option', 'value'])
        save_path = Path(cls.sets_dir, datetime, cls.csv_parameter)
        df_parameter.to_csv(save_path, index=False)
        """

    @classmethod
    def load_parameter(cls, datetime=None):
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

    @classmethod
    def save_weight(cls, datetime, epoch):
        pass
        # weight_epoch-065-best.pt
        # if best:
        """
        weight_name = cls.weight_name + '_epoch-' + str(epoch).zfill(3) + best + '.pt'
        save_path = Path(cls.sets_dir, datetime, cls.weight_dir, weight_name)
        weight = copy.deepcopy(cls.model.state_dict()).to('cpu')  #
        torch.save(weight, save_path)
        """

    @classmethod
    def load_wight(cls, datetime=None, weight_name = None):
        pass
        """
        load_path = Path(cls.sets_dir, datetime, cls.weight_dir, weight_name)
        cls.model.load_state_dict(load_path)
        """

    @classmethod
    def save_learning_curve(cls, datetime, loss_dict):
        pass
        """
        save_name = cls.csv_learning_curve + (label | overall) +'.csv'
        save_path = Path(cls.sets_dir, datetime, cls.learning_curve_dir, save_name)
        df_learning_curve = pd.DataFrame(loss_dict)
        df_learning_curve.to_csv(save_path, index=False)
        """


class ModelWidget(BaseModel, ModelMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class CVModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.image)


class MLPModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.inputs)


class FusionModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'])

    def forward(self):
        self.multi_output = self.network(self.inputs, self.image)


def create_model(args, split_provider):
    mlp = args.mlp
    net = args.net
    if (mlp is not None) and (net is None):
        model = MLPModel(args, split_provider)
    elif (mlp is None) and (net is not None):
        model = CVModel(args, split_provider)
    elif (mlp is not None) and (net is not None):
        model = FusionModel(args, split_provider)
    else:
        logger.error('Cannot identify model type.')
    return model
