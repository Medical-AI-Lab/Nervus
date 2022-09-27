#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import pandas as pd
from lib import set_eval, set_logger
from lib import Logger as logger


def _get_latest_eval_datetime():
    date_names = [path for path in Path('./results/sets/').glob('*') if re.search(r'\d+', str(path))]
    latest = max(date_names, key=lambda date_name: date_name.stat().st_mtime).name
    return latest


def check_eval_options():
    parser = argparse.ArgumentParser(description='Options for eval')
    parser.add_argument('--eval_datetime', type=str, default=None, help='date time for evaluation(Default: None)')
    args = parser.parse_args()

    # Check date time
    if args.eval_datetime is None:
        args.eval_datetime = _get_latest_eval_datetime()
    return args


def _collect_likelihood(eval_datetime):
    likelihood_paths = list(Path('./results/sets/', eval_datetime, 'likelihoods').glob('likelihood_*.csv'))
    assert likelihood_paths != [], f"No likelihood for {eval_datetime}."
    likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return likelihood_paths


def _check_task(eval_datetime):
    parameter_path = Path('./results/sets', eval_datetime, 'parameter.csv')
    df_args = pd.read_csv(parameter_path)
    task = df_args.loc[df_args['option'] == 'task', 'parameter'].item()
    return task


def main(args):
    eval_datetime = args.eval_datetime
    likelihood_paths = _collect_likelihood(eval_datetime)
    task = _check_task(eval_datetime)
    task_eval = set_eval(task)

    logger.logger.info(f"Calculating metrics of {task} for {eval_datetime}.\n")
    for likelihood_path in likelihood_paths:
        logger.logger.info(likelihood_path.name)
        task_eval.make_metrics(eval_datetime, likelihood_path)
        logger.logger.info('')
    logger.logger.info('\nUpdated summary.')


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nEvaluation started.\n')

    args = check_eval_options()
    main(args)

    logger.logger.info('Evaluation done.\n')
