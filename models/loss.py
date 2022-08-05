
import torch

import sys
from pathlib import Path

sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger

logger = Logger.get_logger('models.loss')


"""
from typing import Tuple, Dict, List
import dataclasses
@dataclasses.dataclass
class LossStorage:
    train: List[float] = dataclasses.field(default_factory=list)
    val: List[float] = dataclasses.field(default_factory=list)
"""

"""
def criterion_regression(criterion, multi_output, multi_label):
    multi_loss = dict()
    for internal_label_name in multi_label.keys():
        _output = multi_output[internal_label_name].sqeeze()
        _label = multi_label[internal_label_name].float
        multi_loss[internal_label_name] = criterion(_output, _label)
    return multi_loss
"""

"""
def criterion_deepsurv(criterion, multi_output, multi_label, period, model):
    # len(self.internal_label_list) should be 1
    multi_loss = dict()
    internal_label_name = list(multi_label.keys())[0]
    _pred = multi_output[internal_label_name]
    _period = period.reshape(-1, 1)
    _label = multi_label[internal_label_name].reshape(-1, 1)
    multi_loss[internal_label_name] = criterion(_pred, _period, _label, model)
    return multi_loss
"""

"""
def criterion_classification(criterion, multi_output, multi_label, device):
    _multi_batch_loss = dict()
    # each label
    for internal_label_name in multi_label.keys():
        _output = multi_output[internal_label_name]
        _label = multi_label[internal_label_name]
        _multi_batch_loss[internal_label_name] = criterion(_output, _label)

    # total
    _total = torch.tensor([0.0]).to(device)
    for internal_label_name in multi_label.keys():
        _total = torch.add(_total, _multi_batch_loss[internal_label_name])

    _multi_batch_loss['total'] = _total
    return _multi_batch_loss
"""

"""
def apply_criterion(task):
    if task == 'classification':
        return criterion_classification
    elif task == 'regression':
        return criterion_regression
    else:
        logger.error(f"Cannot identify task: {task}.")
"""


# raw_loss -> iter_loss^* -> epoch_loss
class LossRegistory:
    def __init__(self, task, criterion, internal_label_list, device):
        self.task = task
        self.criterion = criterion
        self.internal_label_list = internal_label_list
        self.device = device

        self.batch_loss = self._init_batch_loss()       # For every batch
        self.running_loss = self._init_running_loss()   # accumlates bach loss
        self.epoch_loss = self._init_epoch_loss()       # For every epoch

        self.best_loss = None

    def _init_batch_loss(self):
        _batch_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _batch_loss[internal_label_name] = None
        return _batch_loss

    def _init_running_loss(self):
        _running_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _running_loss[internal_label_name] = 0.0
        return _running_loss

    def _init_epoch_loss(self):
        _epoch_loss = dict()
        for internal_label_name in self.internal_label_list + ['total']:
            _epoch_loss[internal_label_name] = {'train': [], 'val': []}
        return _epoch_loss

    def cal_batch_loss(self, multi_output, multi_label):
        _total = torch.tensor([0.0]).to(self.device)
        for internal_label_name in self.internal_label_list:
            _output = multi_output[internal_label_name]
            _label = multi_label[internal_label_name]
            #
            # ! task で　criterion の引数の渡し方違う
            # _loss = self.criterion(_output, _label, self.device)
            # 一旦、CELで
            _loss = self.criterion(_output, _label)
            self.batch_loss[internal_label_name] = _loss
            _total = torch.add(_total, _loss)

        self.batch_loss['total'] = _total

    # batch_loss is accumated in runnning_loss
    def cal_running_loss(self, batch_size):
        for internal_label_name in self.internal_label_list:
            _running_loss = self.running_loss[internal_label_name] + (self.batch_loss[internal_label_name].item() * batch_size)
            self.running_loss[internal_label_name] = _running_loss
            self.running_loss['total'] = self.running_loss['total'] + _running_loss

    def cal_epoch_loss(self, phase, dataset_size):
        for internal_label_name in self.internal_label_list:
            _epoch_loss = self.running_loss[internal_label_name] / dataset_size
            self.epoch_loss[internal_label_name][phase].append(_epoch_loss)

        _total = self.running_loss[internal_label_name] / (dataset_size * len(self.internal_label_list))
        self.epoch_loss['total'][phase].append(_total)

        self.batch_loss = self._init_batch_loss()
        self.running_loss = self._init_running_loss()

    def get_epoch_loss(self, phase):
        latest_loss = self.epoch_loss['total'][phase][-1]
        return latest_loss

    def check_if_loss_updated(self):
        self.latest_loss = self.epoch_loss['total']['val'][-1]
        updated_commemt = ''

        if self.best_loss is None:
            self.best_loss = self.latest_loss

        if self.best_loss > self.latest_loss:
            self.best_loss = self.latest_loss
            updated_commemt = '   Updated!'

        return updated_commemt

    def print_epoch_loss(self, num_epochs, epoch):
        train_loss = self.get_epoch_loss('train')
        val_loss = self.get_epoch_loss('val')
        epoch_comm = f"epoch [{epoch+1:>3}/{num_epochs:<3}]"
        train_comm = f"train_loss: {train_loss:.4f}"
        val_comm = f"val_loss: {val_loss:.4f}"
        updated_commemt = self.check_if_loss_updated()
        comment = epoch_comm + ', ' + train_comm + ', ' + val_comm + updated_commemt
        logger.info(comment)
