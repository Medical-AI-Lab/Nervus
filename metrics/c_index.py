#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import dataclasses

import pandas as pd
from lifelines.utils import concordance_index

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib import *
from options import MetricsOptions

logger = NervusLogger.get_logger('evaluation.c_index')

nervusenv = NervusEnv()
args = MetricsOptions().parse()
datetime_dir = get_target(nervusenv.sets_dir, args['likelihood_datetime'])   # args['likelihood_datetime'] if exists or the latest
#likelihood_dir = os.path.join(datetime_dir, nervusenv.likelihood_dir)
#likelihood_path =  glob.glob(likelihood_dir + '/*.csv')[0]   # should be one
#df_likelihood = pd.read_csv(likelihood_path)


@dataclasses.dataclass
class LabelCIndex:
    val: float
    test: float


def cal_c_index(df_likelihood, label_name, period_name):
    pred_name = 'pred_' + label_name
    internal_label_name = 'internal_label' + label_name.replace('label', '')   #   label_XXX -> label_internal_XXX
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        periods_split = df_likelihood_split[period_name].values
        preds_split = df_likelihood_split[pred_name].values
        internal_labels_split = df_likelihood_split[internal_label_name].values
        c_index_split = concordance_index(periods_split, (-1)*preds_split, internal_labels_split)
        if split == 'val':
            c_index_val = c_index_split
        else:
            c_index_test = c_index_split

    label_c_index = LabelCIndex(val=c_index_val, test=c_index_test)
    logger.info(f"{label_name}, val: {label_c_index.val:.2f}, test: {label_c_index.test:.2f}")
    return label_c_index



df_summary_new = pd.DataFrame([])

# Make c_index for each weight
likelihood_dir = os.path.join(datetime_dir, nervusenv.likelihood_dir)
likelihood_path_list = sorted(glob.glob(likelihood_dir + '/*.csv'))


for likelihood_path in likelihood_path_list:
    likelihood_name = os.path.basename(likelihood_path)   # likelihood_weight_epoch-004-best.csv
    logger.info(f"\nLikelihood: {likelihood_name}")
    df_likelihood = pd.read_csv(likelihood_path)

    for institution in df_likelihood['Institution'].unique():
        df_institution_likelihood = df_likelihood.query('Institution == @institution')

        # Calculate c-index
        label_name = list(df_institution_likelihood.columns[df_institution_likelihood.columns.str.startswith('label')])[0]
        period_name = list(df_institution_likelihood.columns[df_institution_likelihood.columns.str.startswith('periods')])[0]

        logger.info(f"{institution}:")
        label_c_index = cal_c_index(df_institution_likelihood, label_name, period_name)

        # Save c-index in summary.csv
        datetime = os.path.basename(datetime_dir)
        weight_name = os.path.basename(likelihood_path).replace(nervusenv.csv_name_likelihood + '_', '').replace('.csv', '.pt')   # weight_epoch-004-best.pt
        summary_tmp = dict()
        summary_tmp['datetime'] = [datetime]
        summary_tmp['weight'] = [weight_name]
        summary_tmp['Institution'] = [institution]
        summary_tmp[label_name + '_val_c_index'] = [f"{label_c_index.val:.2f}"]
        summary_tmp[label_name + '_test_c_index'] = [f"{label_c_index.test:.2f}"]

        df_summary_tmp = pd.DataFrame(summary_tmp)
        df_summary_new = pd.concat([df_summary_new, df_summary_tmp], ignore_index=True)


update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)

# Save c-index in c_index.csv
c_index_path = os.path.join(datetime_dir, nervusenv.csv_name_c_index)
df_summary_new.to_csv(c_index_path, index=False)
