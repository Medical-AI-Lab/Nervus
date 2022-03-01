#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import os
import sys
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import *
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.util import *


class NervusDataSet(Dataset, ABC):
    def __init__(self, args, csv_dict, image_dir, split_list, multi_label=False):
        # super(LoadDataSet_MLP_CNN, self).__init__()
        super().__init__()

        self.args = args
        self.csv_dict = csv_dict
        self.image_dir = image_dir
        self.split_list = split_list  # ['train'], ['val'], ['val', 'test'], ['train', 'val', 'test']

        self.df_source = self.csv_dict["source"]
        self.id_column = self.csv_dict["id_column"]

        if multi_label:
            self.output_list = self.csv_dict["output_list"]
            self.label_list = self.csv_dict["label_list"]
        else:
            self.output_name = csv_dict["output_list"][0] # should be one because single-output
            self.label_name = self.csv_dict["label_list"][0] # should be one because single-output

        self.input_list = self.csv_dict["input_list"]
        self.filepath_column = self.csv_dict["filepath_column"]
        self.split_column = self.csv_dict["split_column"]
        self.df_split = get_column_value(self.df_source, self.split_column, self.split_list)

        # Nomalize input variables
        if not (self.args["mlp"] is None):
            self.input_list_normed = ["normed_" + input for input in self.input_list]
            self.scaler = MinMaxScaler()
            self.df_train = get_column_value(self.df_source, self.split_column, ["train"])  # should be normalized with min and max of training data
            _ = self.scaler.fit(self.df_train[self.input_list])  # fit only

            inputs_normed = self.scaler.transform(self.df_split[self.input_list])
            df_inputs_normed = pd.DataFrame(inputs_normed, columns=self.input_list_normed)
            self.df_split = pd.concat([self.df_split, df_inputs_normed], axis=1)

        # Preprocess for image
        if not (self.args["cnn"] is None):
            self.transform = self._make_transforms()
            self.augmentation = self._make_augmentations()

        # index of each column
        self.index_dict = {column_name: self.df_split.columns.get_loc(column_name) for column_name in self.df_split.columns}


    def _make_transforms(self):
        _transforms = []

        # MUST: Always convert to Tensor
        _transforms.append(transforms.ToTensor())  # PIL -> Tensor

        if self.args["normalize_image"] == "yes":
            # Normalize accepts Tensor only
            _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        _transforms = transforms.Compose(_transforms)
        return _transforms


    def _make_augmentations(self):
        _augmentation = []

        if self.args['preprocess'] == 'yes':
            if self.args['random_horizontal_flip'] == 'yes':
                _augmentation.append(transforms.RandomHorizontalFlip())

            if self.args['random_rotation'] == 'yes':
                _augmentation.append(transforms.RandomRotation((-10, 10)))

            if self.args['color_jitter'] == 'yes':
                # If img is PIL Image, mode “1”, “I”, “F” and modes with transparency (alpha channel) are not supported.
                # When open Grayscle with "RGB" mode
                _augmentation.append(transforms.ColorJitter())

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation


    def __len__(self):
        return len(self.df_split)


    # Convert normalized values to a single Tensor
    def _normalized_value_to_single_tensor_if_mlp(self, idx):
        inputs_value_normed = ""

        if self.args["mlp"] is None:
            return inputs_value_normed

        index_input_list_normed = [
            self.index_dict[input_normed] for input_normed in self.input_list_normed
        ]
        s_inputs_value_normed = self.df_split.iloc[idx, index_input_list_normed]
        inputs_value_normed = np.array(s_inputs_value_normed, dtype=np.float64)
        inputs_value_normed = torch.from_numpy(inputs_value_normed.astype(np.float32)).clone()

        return inputs_value_normed


    # Load imgae when CNN or MLP+CNN
    def _load_image_if_cnn(self, idx):
        image = ""

        if self.args["cnn"] is None:
            return image

        filepath = self.df_split.iat[idx, self.index_dict[self.filepath_column]]
        image_path = os.path.join(self.image_dir, filepath)
        image = Image.open(image_path).convert("RGB")
        image = self.augmentation(image)   # augmentation
        image = self.transform(image)      # transform, ie. To_Tensor() and Normalization

        return image


    @abstractmethod
    def __getitem__(self, idx):
        pass
