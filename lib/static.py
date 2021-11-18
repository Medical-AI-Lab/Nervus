#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd


# data_root is a directory contains directries of csvs and images
data_root = '../materials'
csvs_dir = 'csvs'
images_dir = 'images'


# After reverting train_opt when test
class Static():
    def __init__(self, args, status):
        """
        # Memo
        status:
               train, before_revert, after_revert, evaluate
        """
        self.args = args          # dict
        self.phase = status
            
        self.static = self.args   # Copy

        # When test, the below is needed pior to other to revert train_opt, ie cvs_name from train_opt
        self.static['train_opt_log_dir'] = './train_opt_logs'
        self.static['weight_dir'] = './weights'

        self.csv_name = self.static['csv_name']

        if not(self.csv_name is None):
            self.data_root = data_root
            self.csvs_dir = csvs_dir
            self.images_dir = images_dir

            self.static['data_root'] = self.data_root
            self.static['csvs_dir'] = self.csvs_dir
            self.static['images_dir'] = self.images_dir

            # Fixed column nama in csv
            self.static['filename_column'] = 'finename'
            self.static['dir_to_img_column'] = 'dir_to_img'
            self.static['split_column'] = 'split'

            self.csv_name = self.static['csv_name']           # csv_name = 'XXX.csv'

            if (status == 'train') or (status == 'after_revert'):
                self.image_dir = self.static['image_dir']     # 128, covid, png256   [data_root]/[images_dir]/[image_dir]
                self.load_input = self.static['load_input']
                self.load_image = self.static['load_image']
            else:
                pass
                # When test, the avobe will be reverted from train_opt later

            # Directory for saving
            self.static['result_dir'] = './results'
            self.static['learning_curve_dir'] = os.path.join(self.static['result_dir'], 'learning_curve')
            self.static['likelihood_dir'] = os.path.join(self.static['result_dir'], 'likelihood')
            self.static['roc_dir'] = os.path.join(self.static['result_dir'], 'roc')
            self.static['yy_dir'] = os.path.join(self.static['result_dir'], 'yy')
            self.static['visualization_dir'] = os.path.join(self.static['result_dir'], 'visualization')

        else:
            pass
            # When test, the avobe will be reverted from train_opt later


    def parse(self):
        # CSV
        if not(self.csv_name is None):
            self.csv_path = os.path.join(self.data_root, self.csvs_dir, self.csv_name)
            self.df_source = pd.read_csv(self.csv_path)

            self.column_names = list(self.df_source.columns)
            self.id_column_list = [ column_name for column_name in self.column_names if column_name.startswith('id_') ]
            self.static['id_column'] = self.id_column_list[0]   # must be one when single label

            self.label_list = [ column_name for column_name in self.column_names if column_name.startswith('label_') ]
            self.label_name = self.label_list[0]
            self.static['label_name'] = self.label_name

            self.class_names = [ str(i) for i in self.df_source[self.label_name].unique() ]  # must be ['0', '1']
            self.static['class_names'] = self.class_names
            self.static['num_classes'] = len(self.class_names)                               # 2, when single label classification

            # Input
            if self.load_input == 'yes':
                self.input_list = [ column_name for column_name in self.column_names if column_name.startswith('input_') ]
            else:
                self.input_list = []

            self.static['input_list'] = self.input_list
            self.static['num_inputs'] = len(self.input_list)
            
            # NOTE
            # label_* : int
            # input_* : float
            self.cast_input_dict = { input: float for input in self.input_list }
            self.cast_label_dict = { label: int for label in self.label_list }
            self.df_source = self.df_source.astype(self.cast_input_dict)
            self.df_source = self.df_source.astype(self.cast_label_dict)
            self.static['df_source'] = self.df_source

        else:
            pass

        return self.static


# ----- EOF -----
