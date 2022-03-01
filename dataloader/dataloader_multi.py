#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

from torch.utils.data.dataloader import DataLoader
from dataloader.nervus_dataloader import NervusDataSet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *


class LoadDataSet_MLP_CNN(NervusDataSet):
    def __init__(self, args, split_provider, image_dir, split_list):
        multi_label = True
        super().__init__(args, split_provider, image_dir, split_list, multi_label)


    def __getitem__(self, idx):
        id = self.df_split.iat[idx, self.index_dict[self.id_column]]
        raw_label_dict = {row_label_name: self.df_split.iat[idx, self.index_dict[row_label_name]] for row_label_name in self.raw_label_list}
        internal_label_dict = {internal_label_name: self.df_split.iat[idx, self.index_dict[internal_label_name]] for internal_label_name in self.internal_label_list}
        split = self.df_split.iat[idx, self.index_dict[self.split_column]]

        # Convert normalized values to a single Tensor
        inputs_value_normed = self._normalized_value_to_single_tensor_if_mlp(idx)
        # Load imgae when CNN or MLP+CNN
        image = self._load_image_if_cnn(idx)

        return id, raw_label_dict, internal_label_dict, inputs_value_normed, image, split


def dataloader_mlp_cnn(args, split_provider, images_dir, split_list=None, batch_size=None, sampler=None):
    assert (split_list is not None), 'Specify split to make dataloader.'
    assert (sampler == 'no'), 'samper should be no when multi-ouputs classification or multi-outputs regresson, but yes was specified.'

    split_data = LoadDataSet_MLP_CNN(args, split_provider, images_dir, split_list)
    split_loader = DataLoader(
                            dataset = split_data,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0,
                            sampler = None)
    return split_loader
