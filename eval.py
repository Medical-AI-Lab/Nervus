#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
from lib import set_eval, BaseLogger
from typing import List


logger = BaseLogger.get_logger(__name__)


class EvalOptions:
    """
    Class for options.
    """
    def __init__(self) -> None:
        """
        Options for evaluation.
        """
        self.parser = argparse.ArgumentParser(description='Options for evaluation')
        self.parser.add_argument('--likelihood', type=str, default=None,
                                help='path to a directory which contains likelihoods, or path to a likelihood file. If None, the latest directory is selected automatically. (Default: None)')
        self.args = self.parser.parse_args()

    def get_args(self) -> argparse.Namespace:
        """
        Return arguments.

        Returns:
            argparse.Namespace: arguments
        """
        return self.args


def _get_latest_likelihood_dir() -> str:
    """
    Return path to the latest directory of likelihoods.

    Returns:
        str: path to the latest directory of likelihoods
    """
    _likelihood_dirs = list(Path('results').glob('*/trials/*/likelihoods'))
    assert (_likelihood_dirs != []), 'No directory of likelihood.'
    _likelihood_dir = max(_likelihood_dirs, key=lambda likelihood_dir: Path(likelihood_dir).stat().st_mtime)
    return str(_likelihood_dir)


def _collect_likelihood_paths(likelihood_dir: str) -> List[Path]:
    """
    Return list of paths to likelihoods in likelihood_dir.

    Args:
        likelihood_dir (str): path to directory of likelihoods

    Returns:
        List[Path]: list of paths to likelihoods in likelihood_dir
    """
    _likelihood_paths = list(Path(likelihood_dir).glob('*.csv'))
    assert _likelihood_paths != [], f"No likelihood in {likelihood_dir}."
    _likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return _likelihood_paths


def _check_task(datetime_name: str) -> str:
    """
    Return task done on datetime_name.

    Args:
        datetime_dir (str): directory of datetime at training

    Returns:
        str: task
    """
    _parameter_path = list(Path('results').glob('*/trials/' + datetime_name + '/parameters.json'))[0]
    # If you specify the likelihood of an external dataset,
    # the project name is different from project name in training,
    # but the datetime_name is the same as the datetime in training, which is always one.
    with open(_parameter_path) as f:
        _parameters = json.load(f)
    task = _parameters['task']
    return task


def _eval_parse(args: argparse.Namespace) -> argparse.Namespace:
    """
    Parse parameters required at eval.

    Args:
        args (argparse.Namespace): arguments

    Returns:
        argparse.Namespace: parsed arguments
    """
    # Collect likelihood paths
    if args.likelihood is None:
        args.likelihood = _get_latest_likelihood_dir()
        args.likelihood_paths = _collect_likelihood_paths(args.likelihood)
        _likelihood_dir = str(args.likelihood)
    elif Path.is_dir(Path(args.likelihood)):
        args.likelihood_paths = _collect_likelihood_paths(args.likelihood)
        _likelihood_dir = str(args.likelihood)
    elif Path.is_file(Path(args.likelihood)):
        args.likelihood_paths = [Path(args.likelihood)]
        _likelihood_dir = str(Path(args.likelihood).parents[0])
    else:
        raise ValueError(f"Invalid likelihood path: {args.likelihood}.")

    # Get datetime of test
    _datetime = Path(_likelihood_dir).parents[0].name
    args.datetime = _datetime

    # Check task
    args.task = _check_task(args.datetime)

    return args


def set_eval_options() -> argparse.Namespace:
    """
    Set options for evaluation.

    Returns:
        argparse.Namespace: arguments
    """
    opt = EvalOptions()
    _args = opt.get_args()
    args = _eval_parse(_args)
    return args


def main(args):
    task_eval = set_eval(args.task)

    logger.info(f"Metrics of {args.task}.\n")

    for likelihood_path in args.likelihood_paths:
        logger.info(f"Calculating ...")
        logger.info(f"Load likelihood: {likelihood_path}")

        task_eval.make_metrics(likelihood_path)
        logger.info('\n')

    logger.info('Updated summary.')


if __name__ == '__main__':
    try:
        logger.info('\nEvaluation started.\n')

        args = set_eval_options()
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nEvaluation finished.\n')

