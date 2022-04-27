#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import *

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from dataloader.nervus_dataloader import NervusDataSet

class DeepSurvDataSet(NervusDataSet):
    def __init__(self, args, csv_dict, image_dir, split_list):
        super().__init__(args, csv_dict, image_dir, split_list)
        self.period_columns = self.split_provider.period_column


    def __getitem__(self, idx):
        id = self.df_split.iat[idx, self.index_dict[self.id_column]]
        institution = self.df_split.iat[idx, self.index_dict[self.institution_column]]
        examid = self.df_split.iat[idx, self.index_dict[self.examid_column]]
        raw_label = self.df_split.iat[idx, self.index_dict[self.raw_label_name]]
        internal_label = self.df_split.iat[idx, self.index_dict[self.internal_label_name]]
        split = self.df_split.iat[idx, self.index_dict[self.split_column]]
        period = self.df_split.iloc[idx, self.index_dict[self.period_columns]]

        internal_label = np.array(internal_label, dtype=np.float64)
        internal_label = torch.from_numpy(internal_label.astype(np.float32)).clone()
        #internal_label = internal_label.reshape(-1, 1)

        period = np.array(period, dtype=np.float64)
        period = torch.from_numpy(period.astype(np.float32)).clone()
        #period = period.reshape(-1, 1)

        # Convert normalized values to a single Tensor
        inputs_value_normed = self._normalized_value_to_single_tensor_if_mlp(idx)
        # Load imgae when CNN or MLP+CNN
        image = self._load_image_if_cnn(idx)

        return id, institution, examid, raw_label, internal_label, period, inputs_value_normed, image, split


    @classmethod
    def create_dataloader(cls, args, split_provider, images_dir, split_list=None, batch_size=None, sampler=None):
        assert (split_list is not None), 'Specify split to make dataloader.'

        split_data = cls(args, split_provider, images_dir, split_list)

        # Make sampler
        if sampler == 'yes':
            target = []
            for _, (id, institution, examid, raw_label, internal_label, period, inputs_value_normed, image, split) in enumerate(split_data):
                target.append(internal_label)

            target = [int(t.item()) for t in target]   # Tensor -> int
            class_sample_count = np.array( [ len(np.where(target == t)[0]) for t in np.unique(target) ] )
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target])
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            split_loader = DataLoader(
                                dataset = split_data,
                                batch_size = batch_size,
                                #shuffle = False,        # Default: False, Note: must not be specified when sampler is not None
                                num_workers = 0,
                                sampler = sampler)
        else:
            split_loader = DataLoader(
                                dataset = split_data,
                                batch_size = batch_size,
                                shuffle = True,
                                num_workers = 0,
                                sampler = None)
        return split_loader
