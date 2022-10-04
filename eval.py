#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd
from lib import set_eval, set_logger
from lib import Logger as logger
from typing import List


def check_eval_options() -> argparse.Namespace:
    """
    Parse options.

    Returns:
        argparse.Namespace: oprions
    """
    parser = argparse.ArgumentParser(description='Options for evaluation')
    parser.add_argument('--eval_dir',      type=str, default='baseset', help='directory contaning likekihood (Default: baseset)')
    parser.add_argument('--eval_csv_name', type=str, default=None,      help='csv name likekihood (Default: None)')
    parser.add_argument('--eval_datetime', type=str, default=None,      help='date time for evaluation(Default: None)')
    args = parser.parse_args()

    # Parse eval_datetime
    setattr(args, 'eval_datetime', _get_path_eval_datetime(args.eval_dir, args.eval_datetime))
    return args


def _get_path_eval_datetime(eval_dir: str, eval_datetime: str = None) -> Path:
    if eval_datetime is None:
        _pattern = '*/sets/' + '*' + '/likelihoods'
    else:
        _pattern = '*/sets/' + eval_datetime + '/likelihoods'
    _paths = list(path for path in Path(eval_dir, 'results').glob(_pattern))
    assert (_paths != []), f"No likelihood in eval_datetime(={eval_datetime}) below in {eval_dir}."
    eval_datetime_path = max(_paths, key=lambda datename: datename.stat().st_mtime).parent
    return eval_datetime_path


def _check_task(eval_datetime_dirpath: Path) -> str:
    """
    Return task, ie. classification, regression, or deepsurv
    after finding parameter.csv in eval_datetime_dirpath.

    Args:
        eval_datetime (Path): date time to be evaluated

    Returns:
        str: task
    """
    # eval_datetime_dirpath =
    # PosixPath('baseset/results/int_cla_multi_output_multi_class/sets/2022-10-04-10-16-27')
    _baseset_dirpath = eval_datetime_dirpath.parents[3]  # Path('baseset')
    _datetime = eval_datetime_dirpath.name               # '2022-10-04-10-16-27'
    _parameter_paths = Path(_baseset_dirpath, 'results').glob('*/sets/' + _datetime + '/parameter.csv')
    parameter_path = list(_parameter_paths)[0]  # should be one
    df_args = pd.read_csv(parameter_path, index_col=0)
    task = df_args.loc['task', 'parameter']
    return task


def _collect_likelihood(eval_datetime_dirpath: Path) -> List[Path]:
    """
    Return list of likelihood paths.

    Args:
        likelihood_dirpath (Path): path to directory of likelihoods

    Returns:
        List[Path]: list of likelihood paths
    """
    _likelihood_dirpath = Path(eval_datetime_dirpath, 'likelihoods')
    likelihood_paths = list(_likelihood_dirpath.glob('*.csv'))
    assert likelihood_paths != [], f"No likelihood in {eval_datetime_dirpath}."
    likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return likelihood_paths


def main(args):
    likelihood_paths = _collect_likelihood(args.eval_datetime)
    task = _check_task(args.eval_datetime)
    task_eval = set_eval(task)
    logger.logger.info(f"Calculating metrics of {task} for {args.eval_datetime}.\n")
    for likelihood_path in likelihood_paths:
        logger.logger.info(likelihood_path.name)
        task_eval.make_metrics(likelihood_path)
        logger.logger.info('')
    logger.logger.info('\nUpdated summary.')


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nEvaluation started.\n')

    args = check_eval_options()
    main(args)

    logger.logger.info('Evaluation done.\n')
