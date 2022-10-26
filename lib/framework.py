#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import re
import copy
from abc import ABC, abstractmethod
import pandas as pd
import torch
import torch.nn as nn
import json
import pickle
from .component import (
                make_split_provider,
                create_dataloader,
                create_net
                )
from .logger import Logger as logger
from typing import List, Dict, Union
import argparse


"""
#! testの時は、paramater.jsonの情報だけを使う
#!
# self.conf = args  # !!!!!!!!!!!!!
self.args = args
self.csv_name = Path(self.args.csv_name)
self.task = self.args.task
self.mlp = self.args.mlp
self.net = self.args.net
self.in_channel = self.args.in_channel
self.vit_image_size = self.args.vit_image_size
self.gpu_ids = self.args.gpu_ids
self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids != [] else torch.device('cpu')

csvpath = self.args.csvpath
task = self.args.task
model = self.args.model
criterion
optimizer
epochs
batch_size
augmentation
normalize_image
sampler
in_channel
vit_imgae_size
save_weight
gpu_ids
self.mlp = self.args.mlp
self.net = self.args.net


#! testの時(internalもexternalも)、input_listも、label_listも args.csv_nameで指定されるcsvの情報は使わないから
self.sp = make_split_provider(self.csv_name, self.task)  #! cast だけする
self.df_source = self.sp.df_source
self.label_list = list(self.df_source.columns[self.df_source.columns.str.startswith('label')])
self.input_list = list(self.df_source.columns[self.df_source.columns.str.startswith('input')])
self.mlp_num_inputs = len(self.input_list)
self.num_outputs_for_label = self._define_num_outputs_for_label()
if self.task == 'deepsurv':
    self.period_name = list(self.df_source.columns[self.df_source.columns.str.startswith('period')])[0]

#! testの時、ここで、sp とconfの整合性を合わす
#! self.setup_config(.....)
! testの時は、paramater.jsonの情報だけを使う
# materials/covid/results/int_cla_single_output_bin_class_128/sets/2022-10-21-17-18-27/weights ->
# materials/covid/results/int_cla_single_output_bin_class_128/sets/2022-10-21-17-18-27
_parameter_path = (Path(self.args.weight_dir).parents[0] / 'parameters').with_suffix('.json')
#! testの時、ここで、sp とconfの整合性を合わす
#! self.setup_config(.....)
#! testの時(internalもexternalも)、input_listも、label_listも args.csv_nameで指定されるcsvの情報は使わないから
# base_dir
# augmentation
# ampler
_paramater = self.load_paramater(_parameter_path)
"""


class ModelParam:
    """
    Set up configure for traning or test.
    Integrate args and parameters.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        self._setup_parameters(args)

    def _setup_parameters(self, args) -> None:
        # Copy from args to param
        # No overlap between options for training and test
        for _param, _arg in vars(args).items():
            setattr(self, _param, _arg)

        if self.isTrain:
            sp = make_split_provider(self.csvpath, self.task)
            self.input_list = list(sp.df_source.columns[sp.df_source.columns.str.startswith('input')])
            self.label_list = list(sp.df_source.columns[sp.df_source.columns.str.startswith('label')])
            self.mlp_num_inputs = len(self.input_list)
            self.num_outputs_for_label = self._define_num_outputs_for_label(sp.df_source, self.label_list)
            if self.task == 'deepsurv':
                self.period_name = list(sp.df_source.columns[sp.df_source.columns.str.startswith('period')])[0]

            self.dataloaders = {split: create_dataloader(self, sp.df_source, split=split) for split in ['train', 'val']}
        else:
            # Load paramaters
            # self.weight_dir = 'materials/covid/results/int_cla_single_output_bin_class_128/sets/2022-10-21-17-18-27/weights'
            # -> 'materials/covid/results/int_cla_single_output_bin_class_128/sets/2022-10-21-17-18-27'
            _save_datetime_dir = Path(self.weight_dir).parents[0]
            parameter_path = Path(_save_datetime_dir, 'parameters.json')
            parameters = self.load_parameter(parameter_path)
            no_need_for_test = [
                                'csvpath',
                                'criterion',
                                'optimizer',
                                'lr',
                                'epochs',
                                'batch_size',
                                'save_weight_policy',
                                ]

            # for _param, _arg in parameters.items():
            #    if _param not in no_need_for_test:
            #        setattr(self, _param, _arg)

            # Align because of no need to apply the below at test
            self.augmentation = 'no'
            self.sampler = 'no'


            # ! ここから整合性取る
            # ! internalの情報とexternalの情報で

            # Split_provider
            # csvpathはオプションから、taskはparameterから取ってきたもの
            sp = make_split_provider(self.csvpath, self.task)

            # dataloaderへはtestのcsvの情報が優先?????
            self.input_list = self.params.input_list  # ! testの時は、load_parameterから取ったものになる
            self.label_list = self.params.label_list

            self.dataloaders = {split: create_dataloader(self.params, split=split) for split in self.test_splits}

        # --- Common ---
        # dataset_dir
        # csvpath is specified train.py or test.py
        # eg. csvpath = 'materials/docs/[csv_name].csv' -> 'materials'
        # csvpath exists both at training and test.
        _dataset_dir = re.findall('(.*)/docs', self.csvpath)[0]
        # self.dataset_dir = _dataset_dir
        _csv_name = Path(self.csvpath).stem

        # save_datetime_dir for saveing paramaters, weights, learning_curve, or likelihood
        _save_datetime_dir = str(Path(_dataset_dir, 'results', _csv_name, 'sets', self.datetime))
        self.save_datetime_dir = _save_datetime_dir

        self.device = torch.device(f"cuda:{self.gpu_ids[0]}") if self.gpu_ids != [] else torch.device('cpu')

    def _define_num_outputs_for_label(self, df_source: pd.DataFrame, label_list: List[str]) -> Dict[str, int]:
        """
        Define the number of outputs for each label.

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            label_list (List[str]): label list

        Returns:
            Dict[str, int]: dictionary of the number of outputs for each label
            eg.
                classification:       _num_outputs_for_label = {label_A: 2, label_B: 3, ...}
                regression, deepsurv: _num_outputs_for_label = {label_A: 1, label_B: 1, ...}
        """
        if self.task == 'classification':
            _num_outputs_for_label = {label_name: df_source[label_name].nunique() for label_name in label_list}
        elif (self.task == 'regression') or (self.task == 'deepsurv'):
            _num_outputs_for_label = {label_name: 1 for label_name in label_list}
        else:
            raise ValueError(f"Invalid task: {self.task}.")
        return _num_outputs_for_label

    def print_parameter(self):
        """
        csvpath materials/covid/docs/int_cla_single_output_bin_class_128.csv
        task classification
        model MLP+ResNet18
        criterion CEL
        optimizer Adam
        lr None
        epochs 3
        batch_size 64
        augmentation xrayaug
        normalize_image yes
        sampler no
        in_channel 1
        vit_image_size 0
        save_weight each
        gpu_ids []
        datetime 2022-10-26-14-04-56
        isTrain True
        mlp MLP
        net ResNet18
        input_list ['input_HR', 'input_sBP', 'input_RR', 'input_Sat', 'input_Leukocytes', 'input_CRP', 'input_Sodium', 'input_Potassium', 'input_AST', 'input_ALT', 'input_BUN', 'input_Cre', 'input_Lac', 'input_BNP', 'input_d-dimer', 'input_BMI', 'input_htn_v', 'input_dm_v', 'input_ckd_v', 'input_malignancies_v', 'input_age_splits_1', 'input_age_splits_2', 'input_lung_disease', 'input_heart_disease']
        label_list ['label_last_status_1']
        mlp_num_inputs 24
        num_outputs_for_label {'label_last_status_1': 2}
        dataloaders {'train': <torch.utils.data.dataloader.DataLoader object at 0x117cfcb10>, 'val': <torch.utils.data.dataloader.DataLoader object at 0x1625bfdd0>}
        dataset_dir materials/covid
        save_datetime_dir materials/covid/results/int_cla_single_output_bin_class_128/sets/2022-10-26-14-04-56
        # device cpu
        """
        no_print = ['device']
        for _param, _arg in vars(self).items():
            print(_param, _arg)

    def print_dataset_info(self) -> None:
        """
        Print dataset size for each split.
        """
        for split, dataloader in self.dataloaders.items():
            total = len(dataloader.dataset)
            logger.logger.info(f"{split:>5}_data = {total}")
        logger.logger.info('')

    def save_parameter(self) -> None:
        """
        Save parameters.
        """
        # Delete params not to be saved.
        # str(self.device) if saved.
        no_save_params = [
                        'dataloaders',
                        'device',
                        'save_datetime_dir',
                        'datetime',
                        'isTrain'
                        ]
        saved = dict()
        for _param, _arg in vars(self).items():
            if _param not in no_save_params:
                saved[_param] = _arg

        # weight_dir
        saved['weight_dir'] = str(Path(self.save_datetime_dir, 'weights'))

        # Save scaler
        if hasattr(self.dataloaders['train'].dataset, 'scaler'):
            scaler = self.dataloaders['train'].dataset.scaler
            saved['scaler_path'] = str(Path(self.save_datetime_dir, 'scaler.pkl'))
            with open(saved['scaler_path'], 'wb') as f:
                pickle.dump(scaler, f)

        # Save
        save_dir = Path(self.save_datetime_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = str(Path(save_dir, 'parameters.json'))
        with open(save_path, 'w') as f:
            json.dump(saved, f, indent=4)

    def load_parameter(self, parameter_path: Path) -> Dict:
        """
        Return dictionalry of parameters at training.

        Args:
            parameter_path (Path): path to parameter_path

        Returns:
            Dict: parameters at training
        """
        with open(parameter_path) as f:
            parameters = json.load(f)
        return parameters

    def align_paramaters_after_loading(self):
        pass


class BaseModel(ABC):
    """
    Class to construct model. This class is the base class to construct model.
    """
    def __init__(self, params: ModelParam) -> None:
        """
        Class to define Model

        Args:
            param (ModelParam): parameters for model
        """
        self.params = params

        self.network = create_net(
                                self.params.mlp,
                                self.params.net,
                                self.params.num_outputs_for_label,
                                self.params.mlp_num_inputs,
                                self.params.in_channel,
                                self.params.vit_image_size
                                )

        if self.params.isTrain:
            from .component import set_criterion, set_optimizer, create_loss_reg
            self.criterion = set_criterion(self.params.criterion, self.params.device)
            self.optimizer = set_optimizer(self.params.optimizer, self.network, self.params.lr)
            self.loss_reg = create_loss_reg(self.params.task, self.criterion, self.params.label_list, self.params.device)
        else:
            from .component import set_likelihood
            self.likelihood = set_likelihood(self.params.task, self.params.num_outputs_for_label, self.params.save_datetime_dir)  # Grad-CAMの時はいらない


        # Copy class varialbles refered below or outside.
        self.label_list = self.params.label_list
        self.epochs = self.params.epochs
        self.device = self.params.device
        self.gpu_ids = self.params.gpu_ids
        self.dataloaders = self.params.dataloaders
        self.save_datetime_dir = self.params.save_datetime_dir
        self.save_weight_policy = self.params.save_weight_policy

        # Only when test
        if hasattr(self.params, 'weight_dir'):
            self.weight_dir = self.params.weight_dir


    def print_parameter(self) -> None:
        """
        Print parameters.
        """
        self.params.print_parameter()

    def print_dataset_info(self) -> None:
        """
        Print dataset size for each split.
        """
        self.params.print_dataset_info()

    def train(self) -> None:
        """
        Make self.network training mode.
        """
        self.network.train()

    def eval(self) -> None:
        """
        Make self.network evaluation mode.
        """
        self.network.eval()

    def _enable_on_gpu_if_available(self) -> None:
        """
        Make model compute on the GPU.
        """
        if self.gpu_ids != []:
            assert torch.cuda.is_available(), 'No avalibale GPU on this machine.'
            self.network.to(self.device)
            self.network = nn.DataParallel(self.network, device_ids=self.gpu_ids)
        else:
            pass

    @abstractmethod
    def set_data(self, data):
        pass
        # data = {
        #         'imgpath': imgpath,
        #         'inputs': inputs_value,
        #         'image': image,
        #         'labels': label_dict,
        #         'periods': periods,
        #         'split': split
        #        }: Dict[str, Union[str, torch.Tensor, torch.Tensor, Dict[str, Union[int, float]], int, str]]

    def multi_label_to_device(self, multi_label: Dict[str, Union[int, float]]) -> Dict[str, Union[int, float]]:
        """
        Pass the value of each label to the device

        Args:
            multi_label (Dict[str, Union[int, float]]): dictionary of each label and its value

        Returns:
            Dict[str, Union[int, float]]: dictionary of each label and its value which is on devide
        """
        for label_name, each_data in multi_label.items():
            multi_label[label_name] = each_data.to(self.device)
        return multi_label

    @abstractmethod
    def forward(self):
        pass

    def get_output(self) -> Dict[str, torch.Tensor]:
        """
        Return output of model.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        return self.multi_output

    def backward(self) -> None:
        """
        Backward
        """
        self.loss = self.loss_reg.batch_loss['total']
        self.loss.backward()

    def optimize_parameters(self) -> None:
        """
        Update parameters
        """
        self.optimizer.step()

    # Loss
    @abstractmethod
    def cal_batch_loss(self):
        pass

    def cal_running_loss(self, batch_size: int = None) -> None:
        """
        Calculate loss for each iteration.

        Args:
            batch_size (int): batch size. Defaults to None.
        """
        self.loss_reg.cal_running_loss(batch_size)

    def cal_epoch_loss(self, epoch: int, phase: str, dataset_size: int = None) -> None:
        """
        Calculate loss for each epoch.

        Args:
            epoch (int): epoch number
            phase (str): phase, ie. 'train' or 'val'
            dataset_size (int): dataset size. Defaults to None.
        """
        self.loss_reg.cal_epoch_loss(epoch, phase, dataset_size)

    def is_total_val_loss_updated(self) -> bool:
        """
        Check if val loss updated or not.

        Returns:
            bool: True if val loss updated, otherwise False.
        """
        _total_epoch_loss = self.loss_reg.epoch_loss['total']
        is_updated = _total_epoch_loss.is_val_loss_updated()
        return is_updated

    def print_epoch_loss(self, epoch: int) -> None:
        """
        Print loss for each epoch.

        Args:
            epoch (int): current epoch number
        """
        # self.epochs is total number pf epochs
        self.loss_reg.print_epoch_loss(self.epochs, epoch)

    # Lieklihood
    def make_likelihood(self, data: Dict) -> None:
        """
        Make DataFrame of likelihood.

        Args:
            data (Dict): dictionary of each label and its value which is on devide
        """
        self.likelihood.make_likehood(data, self.get_output())


class SaveLoadMixin:
    """
    Class including methods for save or load weight, learning_curve, or likelihood.
    """
    # variables to keep best_weight and best_epoch temporarily.
    acting_best_weight = None
    acting_best_epoch = None

    # For weight
    def store_weight(self) -> None:
        """
        Store weight.
        """
        self.acting_best_epoch = self.loss_reg.epoch_loss['total'].get_best_epoch()
        _network = copy.deepcopy(self.network)
        if hasattr(_network, 'module'):
            # When DataParallel used, move weight to CPU.
            self.acting_best_weight = copy.deepcopy(_network.module.to(torch.device('cpu')).state_dict())
        else:
            self.acting_best_weight = copy.deepcopy(_network.state_dict())

    def save_weight(self, as_best: bool = None) -> None:
        """
        Save weight.

        Args:
            as_best (bool): True if weight is saved as best, otherise False. Defaults to None.
        """
        assert isinstance(as_best, bool), 'Argument as_best should be bool.'
        save_dir = Path(self.save_datetime_dir, 'weights')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_name = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
        save_path = Path(save_dir, save_name)

        if as_best:
            save_name_as_best = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '_best' + '.pt'
            save_path_as_best = Path(save_dir, save_name_as_best)
            if save_path.exists():
                # Check if best weight already saved. If exists, rename with '_best'
                save_path.rename(save_path_as_best)
            else:
                torch.save(self.acting_best_weight, save_path_as_best)
        else:
            save_name = 'weight_epoch-' + str(self.acting_best_epoch).zfill(3) + '.pt'
            torch.save(self.acting_best_weight, save_path)

    def load_weight(self, weight_path: Path) -> None:
        """
        Load wight from weight_path.

        Args:
            weight_path (Path): path to weight
        """
        weight = torch.load(weight_path)
        self.network.load_state_dict(weight)

        # Make model compute on GPU after loading weight.
        self._enable_on_gpu_if_available()

    # For learning curve
    def save_learning_curve(self) -> None:
        """
        Save leraning curve.

        Args:
            date_name (str): save name for learning curve
        """
        save_dir = Path(self.save_datetime_dir, 'learning_curve')
        save_dir.mkdir(parents=True, exist_ok=True)
        epoch_loss = self.loss_reg.epoch_loss
        for label_name in self.label_list + ['total']:
            each_epoch_loss = epoch_loss[label_name]
            df_each_epoch_loss = pd.DataFrame({
                                                'train_loss': each_epoch_loss.train,
                                                'val_loss': each_epoch_loss.val
                                            })
            best_epoch = str(each_epoch_loss.get_best_epoch()).zfill(3)
            best_val_loss = f"{each_epoch_loss.get_best_val_loss():.4f}"
            save_name = 'learning_curve_' + label_name + '_val-best-epoch-' + best_epoch + '_val-best-loss-' + best_val_loss + '.csv'
            save_path = Path(save_dir, save_name)
            df_each_epoch_loss.to_csv(save_path, index=False)

    # For save parameters
    def save_parameter(self) -> None:
        """
        Save parameters.
        """
        self.params.save_parameter()

    # For likelihood
    def save_likelihood(self, save_name: str = None) -> None:
        """
        Save likelihood.

        Args:
            save_name (str): save name for likelihood. Defaults to None.
        """
        self.likelihood.save_likelihood(save_name=save_name)


class ModelWidget(BaseModel, SaveLoadMixin):
    """
    Class for a widget to inherit multiple classes simultaneously
    """
    pass


class MLPModel(ModelWidget):
    """
    Class for MLP model
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of MLP.

        Args:
            data (Dict): dictionary of data
        """
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.inputs)

    def cal_batch_loss(self) -> None:
        """
        Calculate loss for bach bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class CVModel(ModelWidget):
    """
    Class for CNN or ViT model
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of CNN or ViT Model.

        Args:
            data (Dict): dictionary of data
        """
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.image)

    def cal_batch_loss(self):
        """
        Calculate loss for each bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class FusionModel(ModelWidget):
    """
    Class for MLP+CNN or MLP+ViT model.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of MLP+CNN or MLP+ViT.

        Args:
            data (Dict): dictionary of data
        """
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.inputs, self.image)

    def cal_batch_loss(self) -> None:
        """
        Calculate loss for bach bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label)


class MLPDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with MLP
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of DeepSurv model with MLP

        Args:
            data (Dict): dictionary of data
        """
        self.inputs = data['inputs'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])
        self.periods = data['periods'].float().to(self.device)

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.inputs)

    def cal_batch_loss(self) -> None:
        """
        Calculate loss for each bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.periods, self.network)


class CVDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with CNN or ViT
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of DeepSurv model with with CNN or ViT

        Args:
            data (Dict): dictionary of data
        """
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])
        self.periods = data['periods'].float().to(self.device)

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.image)

    def cal_batch_loss(self) -> None:
        """
        Calculate loss for each bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.periods, self.network)


class FusionDeepSurv(ModelWidget):
    """
    Class for DeepSurv model with MLP+CNN or MLP+ViT model.
    """
    def __init__(self, args: argparse.Namespace) -> None:
        """
        Args:
            args (argparse.Namespace): options
        """
        super().__init__(args)

    def set_data(self, data: Dict) -> None:
        """
        Unpack data for forwarding of DeepSurv with MLP+CNN or MLP+ViT.

        Args:
            data (Dict): dictionary of data
        """
        self.inputs = data['inputs'].to(self.device)
        self.image = data['image'].to(self.device)
        self.multi_label = self.multi_label_to_device(data['labels'])
        self.periods = data['periods'].float().to(self.device)

    def forward(self) -> None:
        """
        Forward.
        """
        self.multi_output = self.network(self.inputs, self.image)

    def cal_batch_loss(self) -> None:
        """
        Calculate loss for bach bach.
        """
        self.loss_reg.cal_batch_loss(self.multi_output, self.multi_label, self.periods, self.network)


def create_model(args: argparse.Namespace) -> nn.Module:
    """
    Construct model

    Args:
        args (argparse.Namespace): options

    Returns:
        nn.Module: model
    """
    params = ModelParam(args)
    task = params.task
    _isMLPModel = (params.mlp is not None) and (params.mlp is None)
    _isCVModel = (params.mlp is None) and (params.mlp is not None)
    _isFusion = (params.mlp is not None) and (params.mlp is not None)

    if (task == 'classification') or (task == 'regression'):
        if _isMLPModel:
            model = MLPModel(params)
        elif _isCVModel:
            model = CVModel(params)
        elif _isFusion:
            model = FusionModel(params)
        else:
            raise ValueError(f"Invalid model type: mlp={mlp}, net={net}.")

    elif task == 'deepsurv':
        if _isMLPModel:
            model = MLPDeepSurv(params)
        elif _isCVModel:
            model = CVDeepSurv(params)
        elif _isFusion:
            model = FusionDeepSurv(params)
        else:
            raise ValueError(f"Invalid model type: mlp={mlp}, net={net}.")

    else:
        raise ValueError(f"Invalid task: {task}.")

    if params.isTrain:
        model._enable_on_gpu_if_available()
    # When test, execute model._enable_on_gpu_if_available() in load_weight(),
    # ie. after loadding weight.
    return model
