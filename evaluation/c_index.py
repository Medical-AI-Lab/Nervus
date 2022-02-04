#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import dataclasses

from lifelines.utils import concordance_index

from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.util import *
from lib.align_env import *
from options.metrics_options import MetricsOptions


args = MetricsOptions().parse()
nervusenv = NervusEnv()
datetime_dir = get_target(nervusenv.sets_dir, args['likelihood_datetime'])   # args['likelihood_datetime'] if exists or the latest
likelihood_path = os.path.join(datetime_dir, nervusenv.csv_likelihood)
df_likelihood = pd.read_csv(likelihood_path)
output_name = [column_name for column_name in df_likelihood.columns if column_name.startswith('output')][0]
period_name = [column_name for column_name in df_likelihood.columns if column_name.startswith('periods')][0]

@dataclasses.dataclass
class LabelCIndex:
    val: float
    test: float

def cal_c_index(df_likelihood, output_name, period_name):
    pred_name = 'pred_' + output_name
    label_name = 'label_' + output_name
    for split in ['val', 'test']:
        df_likelihood_split = get_column_value(df_likelihood, 'split', [split])
        periods_split = df_likelihood_split[period_name].values
        preds_split = df_likelihood_split[pred_name].values
        #outputs_split = df_likelihood_split[output_name].values
        labels_split = df_likelihood_split[label_name].values
        c_index_split = concordance_index(periods_split, (-1)*preds_split, labels_split)
        if split == 'val':
            c_index_val = c_index_split
        else:
            c_index_test = c_index_split
    output_c_index = LabelCIndex(val=c_index_val, test=c_index_test)
    print(f"{output_name}, val: {output_c_index.val:.2f}, test: {output_c_index.test:.2f}")
    return output_c_index

output_c_index = cal_c_index(df_likelihood, output_name, period_name)


# Save c-index in summary.csv
datetime = os.path.basename(datetime_dir)
summary_new = dict()
summary_new['datetime'] = [datetime]
summary_new[output_name+'_val_c_index'] = [f"{output_c_index.val:.2f}"]
summary_new[output_name+'_test_c_index'] = [f"{output_c_index.test:.2f}"]
df_summary_new = pd.DataFrame(summary_new)
update_summary(nervusenv.summary_dir, nervusenv.csv_summary, df_summary_new)


# Save c-index in c_index.csv
c_index_path = os.path.join(datetime_dir, nervusenv.csv_c_index)
df_summary_new.to_csv(c_index_path, index=False)

# ----- EOF -----
