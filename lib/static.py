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
    def __init__(self, args):
        self.static = args   # Copy

        if self.static is None:
            self.static = {}
            self.static['train_opt_log_dir'] = './train_opt_logs'                              # Needed before test a
            self.result_dir = './results'
            self.static['likelihood_dir'] = os.path.join(self.result_dir, 'likelihood')        # Needed before plotting roc
            self.static['roc_dir'] = os.path.join(self.result_dir, 'roc')                      # Needed to save roc
            self.static['visualization_dir'] = os.path.join(self.result_dir, 'visualization')  # Needed to save saliency map

        else:
            self.data_root = data_root
            self.csvs_dir = csvs_dir
            self.images_dir = images_dir

            self.csv_name = self.static['csv_name']        # csv_name = 'XXX.csv'
            self.load_input = self.static['load_input']
            #self.image_dir = self.static['image_dir']     # 128, covid, png256   [data_root]/[images_dir]/[image_dir]
            #self.load_image = self.static['load_image']

            # 
            # The below, addinf required information
            #
            self.static['data_root'] = self.data_root
            self.static['csvs_dir'] = self.csvs_dir
            self.static['images_dir'] = self.images_dir

            # Directory for saving
            self.static['train_opt_log_dir'] = './train_opt_logs'
            self.static['weight_dir'] = './weights'
            self.result_dir = './results'
            self.static['learning_curve_dir'] = os.path.join(self.result_dir, 'learning_curve')
            self.static['likelihood_dir'] = os.path.join(self.result_dir, 'likelihood')
            self.static['roc_dir'] = os.path.join(self.result_dir, 'roc')
            self.static['yy_dir'] = os.path.join(self.result_dir, 'yy')
            self.static['visualization_dir'] = os.path.join(self.result_dir, 'visualization')

            # No need to save in train_opt
            ignore_opts = ['val_batch_size', 'test_batch_size', 'split_column', 'test_datetime', 'likelihood_datetime', 'visualization_datetime', 'visualization_split']
            ignore_dir_fixed = ['data_root', 'csvs_dir', 'images_dir']
            ignore_dir_for_saving = ['train_opt_log_dir', 'weight_dir', 'learning_curve_dir', 'likelihood_dir', 'roc_dir', 'yy_dir', 'visualization_dir']
            ignore_csv_info = ['filename_column', 'dir_to_img_column', 'id_column', 'label_name', 'class_names', 'num_classes','input_list', 'num_inputs', 'df_source']
            args_no_save  = ignore_opts + ignore_dir_fixed + ignore_dir_for_saving + ignore_csv_info
            
            # Args to be saved after training            
            self.static['args_saved'] = [ arg for arg in list(self.static.keys()) if not(arg in args_no_save) ]

            if not(self.static['csv_name'] is None):
                self.static = self.parse()  # add infomation wrt. csv
            else:
                pass


    def parse(self):
        self.csv_path = os.path.join(self.data_root, self.csvs_dir, self.csv_name)
        self.df_source = pd.read_csv(self.csv_path)

        self.static['filename_column'] = 'finename'
        self.static['dir_to_img_column'] = 'dir_to_img'
        self.static['split_column'] = 'split'

        self.column_names = list(self.df_source.columns)
        self.id_column_list = [ column_name for column_name in self.column_names if column_name.startswith('id_') ]
        self.static['id_column'] = self.id_column_list[0]   # must be one when single label

        self.label_list = [ column_name for column_name in self.column_names if column_name.startswith('label_') ]
        self.label_name = self.label_list[0]
        self.static['label_name'] = self.label_name

        self.class_names = [ str(i) for i in self.df_source[self.label_name].unique() ]  # must be ['0', '1']
        self.static['class_names'] = self.class_names
        self.static['num_classes'] = len(self.class_names)                               # 2, when single label classification


        # Need input list or not ?
        if self.load_input == 'yes':
           self.input_list = [ column_name for column_name in self.column_names if column_name.startswith('input_') ]
        else:
            self.input_list = []

        self.static['input_list'] = self.input_list
        self.static['num_inputs'] = len(self.input_list)

        # Cast columns
        # NOTE
        # label_* : int
        # input_* : float
        self.cast_input_dict = { input: float for input in self.input_list }
        self.cast_label_dict = { label: int for label in self.label_list }
        self.df_source = self.df_source.astype(self.cast_input_dict)
        self.df_source = self.df_source.astype(self.cast_label_dict)
        self.static['df_source'] = self.df_source
        
        return self.static


    def align(self):
        return self.static


# ----- EOF -----
