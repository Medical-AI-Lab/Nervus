#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import *

from PIL import Image
from sklearn.preprocessing import MinMaxScaler

from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib import static



class LoadDataSet_MLP_CNN(Dataset):
    def __init__(self, image_dir, label_name, input_list, split_list, resize_size, crop_size, normalize_image, load_image):
        #super(LoadDataSet_MLP_CNN, self).__init__()
        super().__init__()

        self.image_dir = image_dir
        self.label_name = label_name
        self.input_list = input_list
        self.input_list_normed = static.input_list_normed
        self.split_list = split_list  # ['train'], ['val'], ['val', 'test'], ['train', 'val', 'test']

        self.resize_size = resize_size
        self.crop_size = crop_size
        self.normalize_image = normalize_image
        self.load_image = load_image

        self.df_source = static.df_source
        self.df_split = get_column_value(self.df_source, 'split', self.split_list)

        # Nomalize explanatory variables
        self.df_split_normed = self._normalize_input()


        # index of each column
        self.index_dict = { column_name: self.df_split_normed.columns.get_loc(column_name)
                            for column_name in self.df_split_normed.columns
                          }


        # Image tranformation
        _transforms = []

        if not(self.resize_size is None):
            _transforms.append(transforms.Resize((self.resize_size, self.resize_size)))


        if not(self.crop_size is None):
            _transforms.append(transforms.RandomCrop(size=self.crop_size))


        _transforms.append(transforms.ToTensor())


        if not(self.normalize_image is None):
            _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))


        self.transform = transforms.Compose(_transforms)


    # Nomalize explanatory variables
    def _normalize_input(self):
        scaler = MinMaxScaler()
        inputs_normed = scaler.fit_transform(self.df_split[self.input_list])  # np.array (1384, 48)
        df_inputs_normed = pd.DataFrame(inputs_normed, columns=self.input_list_normed)
        df_split_normed = pd.concat([self.df_split, df_inputs_normed], axis=1)

        return df_split_normed



    def __len__(self):
        return len(self.df_split)



    def __getitem__(self, idx):
        # Load imgae when CNN or MLP+CNN
        if self.load_image == 'yes':
            image_path_name = self.df_split.iat[idx, self.index_dict['path_to_img']]
            image_path = os.path.join(self.image_dir, image_path_name)
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)

        else:
            image_path_name = ''
            image = ''


        # Align explanatory values
        label = self.df_split_normed.iat[idx, self.index_dict[self.label_name]]

        # Convert original values to a single Tensor, although might be no need
        s_inputs_value = self.df_split_normed.loc[idx, self.input_list]                         # Series
        inputs_value = np.array(s_inputs_value, dtype=np.float64)                               # -> numpy.float64 (48,)    (np.float64: numpy default type)
        inputs_value = torch.from_numpy(inputs_value.astype(np.float32)).clone()                # -> torch.float32 torch.Size([48]) <- torch default type

        # Convert normalized values to a single Tensor
        s_inputs_value_normed = self.df_split_normed.loc[idx, self.input_list_normed]           # Series
        inputs_value_normed = np.array(s_inputs_value_normed, dtype=np.float64)                 # -> numpy.float64 (48,)    (np.float64: numpy default type)
        inputs_value_normed = torch.from_numpy(inputs_value_normed.astype(np.float32)).clone()  # -> torch.float32 torch.Size([48]) <- torch default type

        pid = self.df_split.iat[idx, self.index_dict['PID']]
        split = self.df_split.iat[idx, self.index_dict['split']]


        return pid, label, inputs_value, inputs_value_normed, image, image_path_name, split



def MakeDataLoader_MLP_CNN_with_WeightedRandomSampler(image_dir, label_name, input_list, split_list=None, resize_size=None, crop_size=None, normalize_image=None, load_image=None, batch_size=None, is_sampler=None):

    if split_list is None:
        print('Specify split to make dataloader.')
        exit()


    split_data = LoadDataSet_MLP_CNN(image_dir, label_name, input_list, split_list, resize_size, crop_size, normalize_image, load_image)


    # Make sampler
    target = []
    for i, (pid, label, inputs_values, inputs_values_normed, image, image_path_name, split) in enumerate(split_data):
         target = target + [label]

    class_sample_count = np.array( [ len(np.where(target == t)[0]) for t in np.unique(target) ] )
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))


    if is_sampler == 'yes':
        split_loader = DataLoader(
                        dataset = split_data,
                        batch_size = batch_size,
                        #shuffle = False,        # Default: False, Note: must not be specified when sampler is not None
                        num_workers = 0,
                        sampler = sampler
                        )

    else:
        split_loader = DataLoader(
                            dataset = split_data,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0,
                            sampler = None
                            )

    return split_loader


# ----- EOF -----
