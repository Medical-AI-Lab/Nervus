#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import pandas as pd

from lifelines.utils import concordance_index


sys.path.append((Path().resolve() / '../').name)
from logger.logger import Logger


logger = Logger.get_logger('metrics.c_index')


class C_Index:
    def __init__(self):
        self.c_index = None

    def set_c_index(self, c_index):
        self.c_index = c_index


class InstC_Index:
    def __init__(self):
        self.val = C_Index()
        self.test = C_Index()

    def set_split_c_index(self, split, c_index):
        if split == 'val':
            self.val.set_c_index(c_index)
        elif split == 'test':
            self.test.set_c_index(c_index)
        else:
            logger.error('Invalid split.')

    def cal_inst_c_index(self, df_inst):
        raw_label = list(df_inst.columns[df_inst.columns.str.startswith('label')])[0]  # should be unique
        pred_labe_name = 'pred_' + raw_label
        internal_label_name = 'internal_' + raw_label
        for split in ['val', 'test']:
            df_split = df_inst.query('split == @split')
            _periods = df_split['period'].values
            _preds = df_split[pred_labe_name].values
            _internal_labels = df_split[internal_label_name].values
            c_index = concordance_index(_periods, (-1)*_preds, _internal_labels)
            self.set_split_c_index(split, c_index)


def cal_c_index(likelihood_path):
    df_likelihood = pd.read_csv(likelihood_path)
    whole_c_index = dict()
    for inst in df_likelihood['Institution'].unique():
        df_inst = df_likelihood.query('Institution == @inst')
        inst_c_index = InstC_Index()
        inst_c_index.cal_inst_c_index(df_inst)
        whole_c_index[inst] = inst_c_index
    return whole_c_index


def make_summary(whole_c_index, datetime, likelihood_path):
    df_summary = pd.DataFrame()
    for inst, inst_c_index in whole_c_index.items():
        _new = dict()
        _new['datetime'] = [datetime]
        _new['weight'] = [likelihood_path.name.replace('likelihood_', '')]
        _new['Institution'] = [inst]
        _new['val_c_index'] = [f"{inst_c_index.val.c_index:.2f}"]
        _new['test_c_index'] = [f"{inst_c_index.test.c_index:.2f}"]
        df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

    df_summary = df_summary.sort_values('Institution')
    return df_summary


def print_c_index(df_summary):
    _df = df_summary[['Institution', 'val_c_index', 'test_c_index']]
    logger.info(_df.to_string(index=False))


def make_c_index(datetime, likelihood_path):
    whole_c_index = cal_c_index(likelihood_path)
    df_summary = make_summary(whole_c_index, datetime, likelihood_path)
    print_c_index(df_summary)
    return df_summary
