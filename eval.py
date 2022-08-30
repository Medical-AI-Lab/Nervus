#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import pandas as pd
import metrics as mt
import logger


log = logger.get_logger('eval')

parser = argparse.ArgumentParser(description='Options for eval')
parser.add_argument('--eval_datetime', type=str, default=None, help='date time for evaluation(Default: None)')
args = parser.parse_args()


def _get_latest_test_datetime():
    date_names = [path for path in Path('./results/sets/').glob('*') if re.search(r'\d+', str(path))]
    latest = max(date_names, key=lambda date_name: date_name.stat().st_mtime).name
    return latest


def set_eval(task):
    if task == 'classification':
        return mt.make_roc, 'ROC'
    elif task == 'regression':
        return mt.make_yy, 'YY'
    elif task == 'deepsurv':
        return mt.make_c_index, 'C_Index'
    else:
        log.error(f"Invalid task: {task}.")
        exit()


def update_summary(df_summary):
    summary_dir = Path('./results/summary')
    summary_path = Path(summary_dir, 'summary.csv')
    if summary_path.exists():
        df_prev = pd.read_csv(summary_path)
        df_updated = pd.concat([df_prev, df_summary], axis=0)
    else:
        summary_dir.mkdir(parents=True, exist_ok=True)
        df_updated = df_summary
    df_updated.to_csv(summary_path, index=False)


#
# Main
#
# Check date time
if args.eval_datetime is None:
    args.eval_datetime = _get_latest_test_datetime()

# Check task
parameter_path = Path('./results/sets', args.eval_datetime, 'parameter.csv')
df_args = pd.read_csv(parameter_path)
task = df_args.loc[df_args['option'] == 'task', 'parameter'].item()

# Collect likelihood
likelihood_paths = list(Path('./results/sets/', args.eval_datetime, 'likelihoods').glob('likelihood_*.csv'))
likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
make_eval, _metrics = set_eval(task)

log.info(f"\nCalculating {_metrics} for {args.eval_datetime}.\n")
for likelihood_path in likelihood_paths:
    log.info(f"Read {likelihood_path.name}.")
    df_summary = make_eval(args.eval_datetime, likelihood_path)
    log.info('')
    update_summary(df_summary)

log.info('Updated summary.')
log.info('Done.\n')
