#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image


class LoadDataSet(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, args, split_provider, split):
        """_summary_

        Args:
            args (_type_): _description_
            split_provider (_type_): _description_
            split (_type_): _description_
        """
        super().__init__()

        self.args = args
        self.split_provider = split_provider
        self.split = split

        self.df_source = self.split_provider.df_source
        self.df_split = self.df_source[self.df_source['split'] == self.split]

        self.raw_label_list = self.split_provider.raw_label_list
        self.internal_label_list = self.split_provider.internal_label_list
        self.input_list = self.split_provider.input_list

        self.col_index_dict = {col_name: self.df_split.columns.get_loc(col_name) for col_name in self.df_split.columns}

        if (self.args.net is not None):
            self.transform = self._make_transforms()
            self.augmentation = self._make_augmentations()

    def _make_transforms(self):
        assert ((self.args.in_channel == 1) or (self.args.in_channel == 3)), f"Invalid input channel: {self.args.in_channel}."

        _transforms = []
        _transforms.append(transforms.ToTensor())

        if self.args.normalize_image == 'yes':
            # transforms.Normalize accepts only Tensor.
            if self.args.in_channel == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))
            else:
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            pass

        _transforms = transforms.Compose(_transforms)
        return _transforms

    def _make_augmentations(self):
        _aug_list = ['randaug', 'trivialaugwide', 'augmix', 'no']
        assert (self.args.augmentation in _aug_list), f"Invalid augmentation: {self.args.augmentation}."

        _augmentation = []

        if self.args.augmentation == 'randaug':
            _augmentation.append(transforms.RandAugment())
        elif self.args.augmentation == 'trivialaugwide':
            _augmentation.append(transforms.TrivialAugmentWide())
        elif self.args.augmentation == 'augmix':
            _augmentation.append(transforms.AugMix())
        else:
            pass

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation

    def _input_value_to_single_tensor_if_mlp(self, idx):
        inputs_value = ''

        if self.args.mlp is None:
            return inputs_value

        index_input_list = [self.col_index_dict[input] for input in self.input_list]
        s_inputs_value = self.df_split.iloc[idx, index_input_list]
        inputs_value = np.array(s_inputs_value, dtype=np.float64)
        inputs_value = torch.from_numpy(inputs_value.astype(np.float32)).clone()
        return inputs_value

    def _load_image_if_cnn(self, idx):
        image = ''

        if self.args.net is None:
            return image

        filepath = self.df_split.iat[idx, self.col_index_dict['filepath']]
        image_path = Path(self.args.image_dir, filepath)

        if self.args.in_channel == 1:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.augmentation(image)
        image = self.transform(image)
        return image

    def _load_priods_if_deepsurv(self, idx):
        period = ''
        if self.args.task != 'deepsurv':
            return period

        assert (self.args.task == 'deepsurv') and (len(self.internal_label_list)==1), 'Deepsurv cannot work in multi-label.'
        period_key = [key for key in self.col_index_dict if key.startswith('period')][0]
        period = self.df_split.iat[idx, self.col_index_dict[period_key]]
        period = np.array(period, dtype=np.float64)
        period = torch.from_numpy(period.astype(np.float32)).clone()
        return period

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        filename = Path(self.df_split.iat[idx, self.col_index_dict['filepath']]).name
        examid = self.df_split.iat[idx, self.col_index_dict['ExamID']]
        institution = self.df_split.iat[idx, self.col_index_dict['Institution']]
        raw_label_dict = {raw_label_name: self.df_split.iat[idx, self.col_index_dict[raw_label_name]] for raw_label_name in self.raw_label_list}
        internal_label_dict = {internal_label_name: self.df_split.iat[idx, self.col_index_dict[internal_label_name]] for internal_label_name in self.internal_label_list}
        inputs_value = self._input_value_to_single_tensor_if_mlp(idx)
        image = self._load_image_if_cnn(idx)
        period = self._load_priods_if_deepsurv(idx)
        split = self.df_split.iat[idx, self.col_index_dict['split']]
        return {
                'Filename': filename,
                'ExamID': examid,
                'Institution': institution,
                'raw_labels': raw_label_dict,
                'internal_labels': internal_label_dict,
                'inputs': inputs_value,
                'image': image,
                'period': period,
                'split': split
                }


def _make_sampler(split_data):
    _target = []
    for _, data in enumerate(split_data):
        _target.append(list(data['internal_labels'].values())[0])

    class_sample_count = np.array([len(np.where(_target == t)[0]) for t in np.unique(_target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in _target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def create_dataloader(args, split_provider, split=None):
    split_data = LoadDataSet(args, split_provider, split)

    if args.sampler == 'yes':
        assert ((args.task == 'classification') or (args.task == 'deepsurv')), 'Cannot make sampler in regression.'
        assert (len(split_provider.raw_label_list) == 1), 'Cannot make sampler for multi-label.'
        shuffle = False
        sampler = _make_sampler(split_data)
    else:
        shuffle = True
        sampler = None

    split_loader = DataLoader(
                            dataset=split_data,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            sampler=sampler
                            )
    return split_loader
