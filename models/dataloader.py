#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

sys.path.append((Path().resolve() / '../').name)
from logger.logger import NervusLogger


logger = NervusLogger.get_logger('dataloder')


class NervusDataSet(Dataset):
    def __init__(self, args, split_provider, split):
        super().__init__()

        self.args = args
        self.split_provider = split_provider
        self.split = split

        self.raw_label_list = self.split_provider.raw_label_list
        self.internal_label_list = self.split_provider.internal_label_list
        self.input_list = self.split_provider.input_list

        self.df_source = self.split_provider.df_source
        self.df_split = self.df_source[self.df_source['split'] == self.split].copy()  # index is not a serial number.
        self.df_split.reset_index(inplace=True, drop=True)                            # Without reset_index, nan occurrs in iteration. This may be bacause the index is missing in some place.


        # Nomalize inputs
        if (self.args.mlp is not None):
            self.df_train = self.df_source[self.df_source['split'] == 'train'].copy()  # should be normalized with min and max of training data
            self.scaler = MinMaxScaler()
            _ = self.scaler.fit(self.df_train[self.input_list])  # only fit
            self.normed_input_list = ['normed_' + input for input in self.input_list]
            _normed_inputs = self.scaler.transform(self.df_split[self.input_list])
            _df_normed_inputs = pd.DataFrame(_normed_inputs, columns=self.normed_input_list)
            self.df_split = pd.concat([self.df_split, _df_normed_inputs], axis=1)

        # Preprocess for image
        if (self.args.cnn is not None):
            self.transform = self._make_transforms()
            self.augmentation = self._make_augmentations()

        self.index_dict = {col_name: self.df_split.columns.get_loc(col_name) for col_name in self.df_split.columns}  # should be after nomalization inputs

    def _make_transforms(self):
        _transforms = []
        _transforms.append(transforms.ToTensor())

        if self.args.normalize_image == 'yes':
            # transforms.Normalize accepts Tensor only.
            if self.args.input_channel == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))
            elif self.args.input_channel == 3:
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            else:
                logger.error(f"Invalid input channel: {self.args.input_channel}.")
        else:
            pass

        _transforms = transforms.Compose(_transforms)
        return _transforms

    def _make_augmentations(self):
        _augmentation = []

        if self.args.augmentation == 'randaug':
            _augmentation.append(transforms.RandAugment())
        elif self.args.augmentation == 'trivialaugwide':
            _augmentation.append(transforms.TrivialAugmentWide())
        elif self.args.augmentation == 'augmix':
            _augmentation.append(transforms.AugMix())   # ? Cannot find in transforms ?
        elif (self.args.augmentation == 'no'):
            pass
        else:
            logger.error(f"Invalid augmentation: {self.args.augmentation}.")

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation

    def _normalized_value_to_single_tensor_if_mlp(self, idx):
        normed_inputs_value_normed = ""

        if (self.args.mlp is None):
            return normed_inputs_value_normed

        index_normed_input_list = [self.index_dict[normed_input] for normed_input in self.normed_input_list]
        s_normed_inputs_value = self.df_split.iloc[idx, index_normed_input_list]
        normed_inputs_value_normed = np.array(s_normed_inputs_value, dtype=np.float64)
        normed_inputs_value_normed = torch.from_numpy(normed_inputs_value_normed.astype(np.float32)).clone()
        return normed_inputs_value_normed

    def _load_image_if_cnn(self, idx):
        image = ""

        if self.args.cnn is None:
            return image

        filepath = self.df_split.iat[idx, self.index_dict['filepath']]
        image_path = Path(self.args.image_dir, filepath)

        if self.args.input_channel == 1:
            image = Image.open(image_path).convert('L')
        else:
            image = Image.open(image_path).convert('RGB')

        image = self.augmentation(image)
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.df_split)

    def __getitem__(self, idx):
        institution = self.df_split.iat[idx, self.index_dict['Institution']]
        examid = self.df_split.iat[idx, self.index_dict['ExamID']]
        filepath = self.df_split.iat[idx, self.index_dict['filepath']]
        raw_label_dict = {raw_label_name: self.df_split.iat[idx, self.index_dict[raw_label_name]] for raw_label_name in self.raw_label_list}
        internal_label_dict = {internal_label_name: self.df_split.iat[idx, self.index_dict[internal_label_name]] for internal_label_name in self.internal_label_list}
        normed_inputs_value = self._normalized_value_to_single_tensor_if_mlp(idx)
        image = self._load_image_if_cnn(idx)
        split = self.df_split.iat[idx, self.index_dict['split']]
        return {
                'Filename': Path(filepath).name,
                'ExamID': examid,
                'Institution': institution,
                'raw_labels': raw_label_dict,
                'internal_labels': internal_label_dict,
                'normed_inputs': normed_inputs_value,
                'image': image,
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
    split_data = NervusDataSet(args, split_provider, split)

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
