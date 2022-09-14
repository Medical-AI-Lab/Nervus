#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from lifelines.utils import concordance_index
from lib.logger import Logger as logger


class C_Index:
    def __init__(self):
        self.c_index = None

    def set_c_index(self, periods, preds, internal_labels):
        self.c_index = concordance_index(periods, (-1)*preds, internal_labels)


class Label_C_Index:
    def __init__(self):
        self.val = C_Index()
        self.test = C_Index()

    def cal_label_c_index(self, raw_label_name, df_label):
        for split in ['val', 'test']:
            df_split = df_label.query('split == @split')
            periods = df_split['period']
            preds = df_split['pred_' + raw_label_name]
            internal_labels = df_split['internal_' + raw_label_name]
            if split == 'val':
                self.val.set_c_index(periods, preds, internal_labels)
            elif split == 'test':
                self.test.set_c_index(periods, preds, internal_labels)
            else:
                logger.logger.error('Invalid split.')
                exit()


def cal_inst_c_index(df_inst):
    raw_label_list = list(df_inst.columns[df_inst.columns.str.startswith('label')])  # may be one label
    inst_c_index = dict()
    for raw_label_name in raw_label_list:
        required_columns = list(df_inst.columns[df_inst.columns.str.contains(raw_label_name)]) + ['period', 'split']
        df_label = df_inst[required_columns]
        label_c_index = Label_C_Index()
        label_c_index.cal_label_c_index(raw_label_name, df_label)
        inst_c_index[raw_label_name] = label_c_index
    return inst_c_index


def cal_c_index(likelihood_path):
    df_likelihood = pd.read_csv(likelihood_path)
    whole_c_index = dict()
    for inst in df_likelihood['Institution'].unique():
        df_inst = df_likelihood.query('Institution == @inst')
        whole_c_index[inst] = cal_inst_c_index(df_inst)
    return whole_c_index


def make_summary(whole_c_index, datetime, likelihood_path):
    df_summary = pd.DataFrame()
    for inst, inst_c_index in whole_c_index.items():
        _new = dict()
        _new['datetime'] = [datetime]
        _new['weight'] = [likelihood_path.stem.replace('likelihood_', '') + '.pt']
        _new['Institution'] = [inst]
        for raw_label_name, inst_c_index in inst_c_index.items():
            _new[raw_label_name + '_val_c_index'] = [f"{inst_c_index.val.c_index:.2f}"]
            _new[raw_label_name + '_test_c_index'] = [f"{inst_c_index.test.c_index:.2f}"]
        df_summary = pd.concat([df_summary, pd.DataFrame(_new)], ignore_index=True)

    df_summary = df_summary.sort_values('Institution')
    return df_summary


def print_c_index(df_summary):
    label_list = list(df_summary.columns[df_summary.columns.str.startswith('label')])
    num_splits = len(['val', 'test'])
    _column_list = [label_list[i:i+num_splits] for i in range(0, len(label_list), num_splits)]
    for _, row in df_summary.iterrows():
        logger.logger.info(row['Institution'])
        for _column in _column_list:
            label_name = _column[0].replace('_val_c_index', '')
            logger.logger.info(f"{label_name:<25} val_c_index: {row[_column[0]]:>7}, test_c_index: {row[_column[1]]:>7}")


def make_c_index(datetime, likelihood_path):
    whole_c_index = cal_c_index(likelihood_path)
    df_summary = make_summary(whole_c_index, datetime, likelihood_path)
    print_c_index(df_summary)
    return df_summary
