#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataloader.dataloader_deepsurv import DeepSurvDataSet
from dataloader.dataloader_multi import MultiLabelDataSet
from dataloader.dataloader_single import SingleLabelDataSet

class DataLoaderFactory():
    def __init__(self, dataloader_type='', args={}, csv_dict={}, image_dir="", batch_size=0, sampler="", split_list=[]) -> None:
        self.dataloader_type = dataloader_type
        self.args = args
        self.csv_dict = csv_dict
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.sampler = sampler
        self.split_list = split_list


    def create(self, dataloader_type='', args={}, csv_dict={}, image_dir="", batch_size=0, sampler="", split_list=[]):
        # replace initial args if exit method args
        _dataloader_type = dataloader_type if dataloader_type else self.dataloader_type
        _args = args if args else self.args
        _csv_dict = csv_dict if csv_dict else self.csv_dict
        _image_dir = image_dir if image_dir else self.image_dir
        _batch_size = batch_size if batch_size else self.batch_size
        _sampler = sampler if sampler else self.sampler
        _split_list = split_list if split_list else self.split_list

        if _dataloader_type == 'deepsurv':
            _dataloader_handler = DeepSurvDataSet
        elif _dataloader_type == 'singlelabel':
            _dataloader_handler = SingleLabelDataSet
        elif _dataloader_type == 'multilabel':
            _dataloader_handler = MultiLabelDataSet

        _dataloader = _dataloader_handler.create_dataloader(_args, _csv_dict, _image_dir, _batch_size, _sampler, _split_list)

        return _dataloader
