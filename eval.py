#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import pandas as pd
from lib import set_eval, set_logger
from lib import Logger as logger
from typing import List


def _get_latest_eval_datetime(eval_dir: str) -> str:
    """
        Return the latest directory to be evaluated.

        Returns:
            str: directory name indicating date name, eg. 2022-01-02-13-30-20
        """
    date_names = [path for path in Path(eval_dir, 'results/sets').glob('*') if re.search(r'\d+', str(path))]
    latest = max(date_names, key=lambda date_name: date_name.stat().st_mtime).name
    return latest


def check_eval_options() -> argparse.Namespace:
    """
    Parse options.

    Returns:
        argparse.Namespace: oprions
    """
    parser = argparse.ArgumentParser(description='Options for evaluation')
    parser.add_argument('--task',          type=str, default=None, help='task at training. (Default: None)')
    parser.add_argument('--eval_dir',      type=str, default='baseset', help='directory contaning likekihood (Default: baseset)')
    parser.add_argument('--eval_datetime', type=str, default=None,      help='date time for evaluation(Default: None)')
    args = parser.parse_args()

    if args.eval_datetime is None:
        setattr(args, 'eval_datetime', _get_latest_eval_datetime(args.eval_dir))
    return args


def _collect_likelihood(likelihood_dirpath: Path) -> List[Path]:
    """
    Return list of likelihood paths.

    Args:
        likelihood_dirpath (Path): path to directory of likelihoods

    Returns:
        List[Path]: list of likelihood paths
    """
    likelihood_paths = list(likelihood_dirpath.glob('*.csv'))
    assert likelihood_paths != [], f"No likelihood in {likelihood_paths}."
    likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return likelihood_paths


def _check_task(eval_datetime) -> str:
    """
    Return task, ie. classificatio, regression, or deepsurv.

    Args:
        eval_datetime (str): date time to be evaluated

    Returns:
        str: task
    """
    parameter_path = Path(args.weight_dir, 'results/sets', eval_datetime, 'parameter.csv')
    df_args = pd.read_csv(parameter_path, index_col=0)
    task = df_args.loc['task', 'parameter']
    return task


def main(args):
    _likelihood_dirpath = Path(args.eval_dir, 'results/sets', args.eval_datetime, 'likelihoods')
    likelihood_paths = _collect_likelihood(_likelihood_dirpath)

    # task = _check_task(args.eval_datetime)
    task_eval = set_eval(args)

    logger.logger.info(f"Calculating metrics of {args.task} for {args.eval_datetime}.\n")
    for likelihood_path in likelihood_paths:
        logger.logger.info(likelihood_path.name)
        task_eval.make_metrics(args.eval_datetime, likelihood_path)
        logger.logger.info('')
    logger.logger.info('\nUpdated summary.')


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nEvaluation started.\n')

    args = check_eval_options()
    main(args)

    logger.logger.info('Evaluation done.\n')
