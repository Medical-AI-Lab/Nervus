#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy as sp
import torch
import torch.nn as nn
from torchinfo import summary

from abc import ABC, abstractmethod

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
        for internal_label_name, unit in self.multi_classifier.items():
            output[internal_label_name] = unit(out_features)
        return output

    # def get_label_output(self, multi_output, internal_label_name):
    #    return multi_output[internal_label_name]


class MultiWidget(nn.Module, BaseNet, MultiMixin):
    """
    Class for a widght to inherit multiple classes simultaneously
    """
    pass


class MultiNet(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channels=None, vit_image_size=None):
        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channels = in_channels
        self.vit_image_size = vit_image_size

        self.extractor = self.constuct_extractor(self.net_name, mlp_num_inputs=self.mlp_num_inputs, in_channels=self.in_channels, vit_image_size=self.vit_image_size)
        self.multi_classifier = self.construct_multi_classifier(self.net_name, self.num_classes_in_internal_label)

    def forward(self, x):
        out_features = self.extractor(x)
        output = self.multi_forward(out_features)
        return output


class MultiNetFusion(MultiWidget):
    def __init__(self, net_name, num_classes_in_internal_label, mlp_num_inputs=None, in_channels=None, vit_image_size=None):
        assert (net_name != 'MLP'), 'net_name should not be MLP.'

        super().__init__()

        self.net_name = net_name
        self.num_classes_in_internal_label = num_classes_in_internal_label
        self.mlp_num_inputs = mlp_num_inputs
        self.in_channels = in_channels
        self.vit_image_size = vit_image_size

        # Extractor of MLP and Net
        self.extractor_mlp = self.constuct_extractor('MLP', mlp_num_inputs=self.mlp_num_inputs)
        self.extractor_net = self.constuct_extractor(self.net_name, in_channels=self.in_channels, vit_image_size=self.vit_image_size)
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


def create_model(mlp, net, num_classes_in_internal_label, mlp_num_inputs, in_channels, vit_image_size, gpu_ids):
    if (mlp is not None) and (net is None):
        multi_net = MultiNet('MLP', num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channels=in_channels, vit_image_size=vit_image_size)

    elif (mlp is None) and (net is not None):
        multi_net = MultiNet(net, num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channels=in_channels, vit_image_size=vit_image_size)

    elif (mlp is not None) and (net is not None):
        multi_net = MultiNetFusion(net, num_classes_in_internal_label, mlp_num_inputs=mlp_num_inputs, in_channels=in_channels, vit_image_size=vit_image_size)

    else:
        logger.error('Cannot identify net type.')

    return multi_net


class ModelMixin:
    def multi_label_to_device(self, multi_label, device):
        for internal_label_name, each_data in multi_label.items():
            multi_label[internal_label_name] = each_data.to(device)
        return multi_label

    def print_epoch_loss(self, num_epochs, epoch):
        train_loss = self.loss_reg.get_epoch_loss('train')
        val_loss = self.loss_reg.get_epoch_loss('val')
        epoch_comm = f"epoch [{epoch+1:>3}/{num_epochs:<3}]"
        train_comm = f"train_loss: {train_loss:.4f}"
        val_comm = f"val_loss: {val_loss:.4f}"
        updated_commemt = self.loss_reg.check_loss_updated()
        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_commemt
        logger.info(comment)

    def save_weight(self, strategy):
        pass

    def load_wight(self, datetime=None):
        pass

    def save_oprtions(self):
        pass

    def load_options(self):
        # train_parameter.augmantation = 'no'
        pass


def create_loss_reg(criterion, internal_label_list, device):
    loss_reg = LossRegistory(criterion, internal_label_list, device)
    return loss_reg


def set_criterion(criterion, device):
    criterion = Criterion.set_criterion(criterion, device)
    return criterion


def set_optimizer(optimizer, network, lr):
    optimizer = Optimizer.set_optimizer(optimizer, network, lr)
    return optimizer


class BaseModel(ABC):
    def __init__(self, args, split_provider):
        self.args = args
        self.sp = split_provider

        self.mlp = self.args.mlp
        self.net = self.args.net
        self.internal_label_list = self.sp.internal_label_list
        self.num_classes_in_internal_label = self.sp.num_classes_in_internal_label
        self.mlp_num_inputs = len(self.sp.input_list)
        self.in_channels = self.args.in_channels
        self.vit_image_size = self.args.vit_image_size
        self.lr = self.args.lr

        self.gpu_ids = self.args.gpu_ids
        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids else torch.device('cpu')

        # Pass to GPU
        # DataParallel
        self.network = create_model(self.mlp, self.net, self.num_classes_in_internal_label, self.mlp_num_inputs, self.in_channels, self.vit_image_size, self.gpu_ids)
        self.criterion = set_criterion(args.criterion, self.device)
        self.optimizer = set_optimizer(args.optimizer, self.network, self.lr)
        self.loss_reg = create_loss_reg(self.criterion, self.internal_label_list, self.device)

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

    @abstractmethod
    def forward(self):
        pass
        # self.multi_output = self.network(self.image)

    def store_raw_loss(self):
        self.loss_reg.store_raw_loss(self.multi_output, self.multi_label)

    def backward(self):
        self.loss = self.loss_reg.get_raw_loss()
        self.loss.backward()

    def optimize_paramters(self):
        self.optimizer.step()

    def store_iter_loss(self):
        self.loss_reg.store_iter_loss()

    def store_epoch_loss(self, phase, len_dataloader):
        self.loss_reg.store_epoch_loss(phase, len_dataloader)


class ModelWidget(BaseModel, ModelMixin):
    """
    Class for a widght to inherit multiple classes simultaneously
    """
    pass


class CVModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)
        # self.network = create_model(self.mlp, self.net, self.num_classes_in_internal_label, self.mlp_num_inputs, self.in_channels, self.vit_image_size, self.gpu_ids)
        # self.optimizer = set_optimizer(args.optimizer, self.network, self.lr)

    def set_data(self, data):
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'], self.device)

    def forward(self):
        self.multi_output = self.network(self.image)


class MLPModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)
        # self.network = create_model(self.mlp, self.net, self.num_classes_in_internal_label, self.mlp_num_inputs, self.in_channels, self.vit_image_size, self.gpu_ids)
        # self.optimizer = set_optimizer(args.optimizer, self.network, self.lr)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'], self.device)

    def forward(self):
        self.multi_output = self.network(self.inputs)


class FusionModel(ModelWidget):
    def __init__(self, args, split_provider):
        super().__init__(args, split_provider)
        # self.network = create_model(self.mlp, self.net, self.num_classes_in_internal_label, self.mlp_num_inputs, self.in_channels, self.vit_image_size, self.gpu_ids)
        # self.optimizer = set_optimizer(args.optimizer, self.network, self.lr)

    def set_data(self, data):
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['internal_labels'], self.device)

    def forward(self):
        self.multi_output = self.network(self.inputs, self.image)
