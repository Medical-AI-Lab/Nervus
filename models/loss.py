
import torch

"""
from typing import Tuple, Dict, List
import dataclasses
@dataclasses.dataclass
class LossStorage:
    train: List[float] = dataclasses.field(default_factory=list)
    val: List[float] = dataclasses.field(default_factory=list)
"""

# raw_loss -> iter_loss^* -> epoch_loss
class LossRegistory():
    def __init__(self, criterion, internal_label_list, device):
        super().__init__()

        self.criterion = criterion
        self.internal_label_list = internal_label_list
        self.device = device
        self.raw_loss = self._init_raw_loss()
        self.iter_loss = self._init_iter_loss()
        self.epoch_loss = self._init_epoch_loss()  # iter を 平均　
        self.best_loss = None    # epoch==0 以降は、Noneでない

    def _init_raw_loss(self):
        return dict()

    def _init_iter_loss(self):
        _loss_store = dict()
        for internal_label_name in self.internal_label_list:
            _loss_store[internal_label_name] = 0.0
        return _loss_store

    def _init_epoch_loss(self):
        _epoch_loss = dict()
        for internal_label_name in self.internal_label_list:
            _epoch_loss[internal_label_name] = {'train': [], 'val': []}

        _epoch_loss['total'] = {'train': [], 'val': []}
        return _epoch_loss

    # multi_output, internal_labels は、batch_size ごと
    # ! regression の時、output.squeeze() いる?
    def store_raw_loss(self, multi_output, multi_label):
        for internal_label_name in self.internal_label_list:
            _each_raw_loss = self.criterion(multi_output[internal_label_name], multi_label[internal_label_name])
            self.raw_loss[internal_label_name] = _each_raw_loss

    # For backward only when train
    def get_raw_loss(self):
        _raw_loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        for internal_label_name in self.internal_label_list:
            _raw_loss = torch.add(_raw_loss, self.raw_loss[internal_label_name])
        return _raw_loss

    # iterごとに加算していくだけ
    def store_iter_loss(self):
        for internal_label_name in self.internal_label_list:
            self.iter_loss[internal_label_name] = self.iter_loss[internal_label_name] + self.raw_loss[internal_label_name].item()
        # Initialize
        self.raw_loss = self._init_raw_loss()

    def store_epoch_loss(self, phase, len_dataloader):
        _total_avg = 0.0
        for internal_label_name in self.internal_label_list:
            _each_avg_iter_loss = self.iter_loss[internal_label_name] / len_dataloader
            self.epoch_loss[internal_label_name][phase].append(_each_avg_iter_loss)
            _total_avg = _total_avg + _each_avg_iter_loss

        _total_avg = _total_avg / len(self.internal_label_list)
        self.epoch_loss['total'][phase].append(_total_avg)
        # Initialize
        self.iter_loss = self._init_iter_loss()

    def get_epoch_loss(self, phase):
        latest_loss = self.epoch_loss['total'][phase][-1]
        return latest_loss

    def check_loss_updated(self):
        self.latest_loss = self.epoch_loss['total']['val'][-1]
        updated_commemt = ''

        if self.best_loss is None:
            self.best_loss = self.latest_loss

        if self.best_loss > self.latest_loss:
            self.best_loss = self.latest_loss
            updated_commemt = '   Updated!'

        return updated_commemt
