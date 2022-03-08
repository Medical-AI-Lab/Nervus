#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import *
from dataloader.nervus_dataloader import NervusDataSet

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *


class SingleLabelDataSet(NervusDataSet):
    def __init__(self, args, csv_dict, image_dir, split_list):
        #super(LoadDataSet_MLP_CNN, self).__init__()
        super().__init__(args, csv_dict, image_dir, split_list)


    def __getitem__(self, idx):
        id = self.df_split.iat[idx, self.index_dict[self.id_column]]
        raw_output = self.df_split.iat[idx, self.index_dict[self.output_name]]
        label = self.df_split.iat[idx, self.index_dict[self.label_name]]
        split = self.df_split.iat[idx, self.index_dict[self.split_column]]

        # Convert normalized values to a single Tensor
        inputs_value_normed = self._normalized_value_to_single_tensor_if_mlp(idx)
        # Load imgae when CNN or MLP+CNN
        image = self._load_image_if_cnn(idx)

        return id, raw_output, label, inputs_value_normed, image, split


    @classmethod
    def create_dataloader(cls, args, csv_dict, images_dir, batch_size=None, sampler=None, split_list=None):
        assert (split_list is not None), 'Specify split to make dataloader.'
        if (args['task'] == 'regression'):
            assert (sampler == 'no'), 'samper should be no when regression, but yes was specified.'

        split_data = cls(args, csv_dict, images_dir, split_list)

        # Make sampler
        if sampler == 'yes':
            target = []

            for _, (_, _, label, _, _, _) in enumerate(split_data):
                target.append(label)

            class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target])
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            split_loader = DataLoader(
                                dataset = split_data,
                                batch_size = batch_size,
                                #shuffle = False,        # Default: False, Note: must not be True when sampler is not None.
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
